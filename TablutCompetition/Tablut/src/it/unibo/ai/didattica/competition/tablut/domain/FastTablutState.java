package it.unibo.ai.didattica.competition.tablut.domain;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random; // Import aggiunto
import java.io.IOException;

import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * Stato del gioco ottimizzato per le performance.
 * Implementa la logica completa di movimento e cattura delle regole Ashton Tablut.
 * Zobrist Hashing per Transposition Table.
 */
public class FastTablutState extends State implements Serializable {

    private static final long serialVersionUID = 1L;
    private static final int BOARD_SIZE = 9;

    // Mappatura interna dei pedoni su tipi primitivi (byte)
    public static final byte E = 0, W = 1, B = 2, K = 3, T = 4;

    public byte[] fastBoard;

    // Metadati per accesso immediato
    public int kingRow = -1;
    public int kingCol = -1;
    public int whitePawnsCount = 0;
    public int blackPawnsCount = 0;

    private long zobristKey;

    // Tabella [riga][colonna][tipo_pedone]
    // tipo_pedone: 0=E, 1=W, 2=B, 3=K, 4=T
    private static final long[][][] zobristTable = new long[BOARD_SIZE][BOARD_SIZE][5];
    private static final long zobristTurnBlack;
    private static final Random rand = new Random(42); // Usa un seed fisso per la riproducibilità

    static {
        // Inizializza la tabella con valori casuali
        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                for (int p = 0; p < 5; p++) {
                    zobristTable[r][c][p] = rand.nextLong();
                }
            }
        }
        zobristTurnBlack = rand.nextLong();
    }

    private static final java.util.Map<Pawn, Byte> PAWN_TO_BYTE = new java.util.HashMap<>();
    static {
        PAWN_TO_BYTE.put(Pawn.EMPTY, E);
        PAWN_TO_BYTE.put(Pawn.WHITE, W);
        PAWN_TO_BYTE.put(Pawn.BLACK, B);
        PAWN_TO_BYTE.put(Pawn.KING, K);
        PAWN_TO_BYTE.put(Pawn.THRONE, T);
    }

    // Punti sensibili del tabellone (coordinate di matrice [r, c])
    public static final List<int[]> CITADELS = Collections.unmodifiableList(List.of(
            new int[]{0,3}, new int[]{0,4}, new int[]{0,5}, new int[]{1,4},
            new int[]{8,3}, new int[]{8,4}, new int[]{8,5}, new int[]{7,4},
            new int[]{3,8}, new int[]{4,8}, new int[]{5,8}, new int[]{4,7},
            new int[]{3,0}, new int[]{4,0}, new int[]{5,0}, new int[]{4,1}
    ));
    public static final List<int[]> ESCAPES = Collections.unmodifiableList(List.of(
            new int[]{0,1}, new int[]{0,2}, new int[]{0,6}, new int[]{0,7},
            new int[]{8,1}, new int[]{8,2}, new int[]{8,6}, new int[]{8,7},
            new int[]{1,0}, new int[]{2,0}, new int[]{6,0}, new int[]{7,0},
            new int[]{1,8}, new int[]{2,8}, new int[]{6,8}, new int[]{7,8}
    ));
    public static final int[] THRONE = {4, 4};

    // Costruttore privato
    private FastTablutState() {
        this.fastBoard = new byte[BOARD_SIZE * BOARD_SIZE];
    }

    // Metodi di accesso interni semplici
    public byte get(int r, int c) {
        if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) return E; // Ritorna EMPTY per coordinate fuori bordi
        return fastBoard[r * BOARD_SIZE + c];
    }
    public void set(int r, int c, byte pawnType) { fastBoard[r * BOARD_SIZE + c] = pawnType; }

    private boolean isCitadel(int r, int c) {
        if (r == THRONE[0] && c == THRONE[1]) return false;
        for (int[] coord : CITADELS) { if (coord[0] == r && coord[1] == c) return true; }
        return false;
    }

    private boolean containsCoord(List<int[]> list, int r, int c) {
        for (int[] coord : list) { if (coord[0] == r && coord[1] == c) return true; }
        return false;
    }

    public static FastTablutState fromState(State state) {
        FastTablutState fastState = new FastTablutState();
        fastState.turn = state.getTurn();
        fastState.zobristKey = 0L;

        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                Pawn pawn = state.getPawn(r, c);
                byte bytePawn = PAWN_TO_BYTE.getOrDefault(pawn, E);

                fastState.set(r, c, bytePawn);

                fastState.zobristKey ^= zobristTable[r][c][bytePawn];

                if (pawn.equals(Pawn.WHITE)) {
                    fastState.whitePawnsCount++;
                } else if (pawn.equals(Pawn.BLACK)) {
                    fastState.blackPawnsCount++;
                } else if (pawn.equals(Pawn.KING)) {
                    fastState.kingRow = r;
                    fastState.kingCol = c;
                }
            }
        }

        if (fastState.turn.equals(Turn.BLACK)) {
            fastState.zobristKey ^= zobristTurnBlack;
        }

        return fastState;
    }

    @Override
    public FastTablutState clone() {
        FastTablutState newState = new FastTablutState();
        newState.fastBoard = this.fastBoard.clone();
        newState.turn = this.turn;
        newState.kingRow = this.kingRow;
        newState.kingCol = this.kingCol;
        newState.whitePawnsCount = this.whitePawnsCount;
        newState.blackPawnsCount = this.blackPawnsCount;
        newState.zobristKey = this.zobristKey;
        return newState;
    }

    // ZOBRIST: Getter per la chiave
    public long getZobristKey() {
        return this.zobristKey;
    }

    // ----------------------------------------------------------------------
    // LOGICA DI GIOCO OTTIMIZZATA
    // ----------------------------------------------------------------------

    private boolean isPathClear(int r1, int c1, int r2, int c2, byte movingPawn) {
        int dr = Integer.signum(r2 - r1);
        int dc = Integer.signum(c2 - c1);

        for (int r = r1 + dr, c = c1 + dc; r != r2 || c != c2; r += dr, c += dc) {
            byte currentPawn = get(r, c);

            if (currentPawn != E && currentPawn != T) return false;

            // REGOLA ASHTON: Solo il Re può passare sul Trono
            if (currentPawn == T && movingPawn != K) return false;

            // REGOLA ASHTON: Nessuno può passare su una Cittadella (tranne il Re se è la sua mossa finale per fuggire)
            // Questa logica è complessa, la gestiamo nel check di validazione mossa.
            // Per ora, assumiamo che le cittadelle vuote blocchino (come il trono per i soldati).
            if (isCitadel(r, c) && movingPawn != K) return false;

            // Fix per Re che attraversa cittadelle (non consentito)
            if (isCitadel(r, c) && movingPawn == K) return false;
        }
        return true;
    }

    /**
     * Esegue la mossa, la validazione, la cattura e il cambio di turno.
     * @return true se la mossa è valida e lo stato è stato modificato.
     */
    public boolean applyMove(Action a) {
        int rFrom = a.getRowFrom();
        int cFrom = a.getColumnFrom();
        int rTo = a.getRowTo();
        int cTo = a.getColumnTo();

        // 1. Controllo di base e turno
        if (rFrom == rTo && cFrom == cTo) return false; // Mossa nulla
        if (rFrom != rTo && cFrom != cTo) return false; // Mossa diagonale

        byte pawn = get(rFrom, cFrom);
        if (pawn == E || pawn == T) return false; // Muove casella vuota o trono

        if (this.turn.equals(Turn.WHITE) && (pawn != W && pawn != K)) return false; // Turno Bianco, muove Nero
        if (this.turn.equals(Turn.BLACK) && (pawn != B)) return false; // Turno Nero, muove Bianco/Re

        // 2. Controllo di destinazione
        if (get(rTo, cTo) != E) return false; // Destinazione occupata

        // 3. Controllo Cittadelle e Trono (Validazione atterraggio)
        if (rTo == THRONE[0] && cTo == THRONE[1]) {
            if (pawn != K) return false; // Solo il Re può andare sul trono
        } else if (isCitadel(rTo, cTo)) {
            // REGOLA ASHTON: Solo il Re può atterrare su una cittadella, E SOLO se è una casella di fuga
            if (pawn != K) {
                return false; // Pedoni (W/B) non possono MAI atterrare su una cittadella
            } else { // Re (K):
                if (!containsCoord(ESCAPES, rTo, cTo)) {
                    return false; // Il Re non può atterrare su una cittadella NON-escape (es. a4)
                }
            }
        }

        // 4. Controllo del percorso (Climbing)
        if (!isPathClear(rFrom, cFrom, rTo, cTo, pawn)) return false;

        // 5. Esegui la Mossa
        byte oldPawnAtFrom = get(rFrom, cFrom);
        byte newPawnAtFrom = (rFrom == THRONE[0] && cFrom == THRONE[1]) ? T : E;
        set(rFrom, cFrom, newPawnAtFrom);
        // Aggiorna Zobrist per la casella 'from'
        this.zobristKey ^= zobristTable[rFrom][cFrom][oldPawnAtFrom];
        this.zobristKey ^= zobristTable[rFrom][cFrom][newPawnAtFrom];

        byte oldPawnAtTo = get(rTo, cTo); // Sarà E
        byte newPawnAtTo = pawn;
        set(rTo, cTo, newPawnAtTo);
        // Aggiorna Zobrist per la casella 'to'
        this.zobristKey ^= zobristTable[rTo][cTo][oldPawnAtTo];
        this.zobristKey ^= zobristTable[rTo][cTo][newPawnAtTo];


        if (pawn == K) {
            this.kingRow = rTo;
            this.kingCol = cTo;
            if (containsCoord(ESCAPES, rTo, cTo)) {
                this.turn = Turn.WHITEWIN;
                return true;
            }
        }

        // 6. Check Catture
        checkCaptures(rTo, cTo, pawn);

        // 7. Controlla Sconfitta/Vittoria (post-cattura)
        if (this.turn != Turn.WHITEWIN && this.kingRow == -1) {
            this.turn = Turn.BLACKWIN;
        }
        if (this.turn != Turn.WHITEWIN && this.turn != Turn.BLACKWIN) {
            // Controlli aggiuntivi di stallo/vittoria materiale
            if (this.whitePawnsCount == 0 && this.kingRow == -1) { this.turn = Turn.BLACKWIN; }
            if (this.blackPawnsCount == 0) { this.turn = Turn.WHITEWIN; }
        }

        // 8. Cambia Turno (se il gioco non è finito)
        if (this.turn != Turn.WHITEWIN && this.turn != Turn.BLACKWIN) {
            if (this.turn.equals(Turn.WHITE)) this.turn = Turn.BLACK;
            else if (this.turn.equals(Turn.BLACK)) this.turn = Turn.WHITE;

            this.zobristKey ^= zobristTurnBlack;
        }

        return true;
    }

    /**
     * Metodo helper per aggiornare lo Zobrist Key durante una cattura.
     * Deve essere chiamato PRIMA di modificare il tabellone.
     */
    private void updateZobristKeyForRemoval(int r, int c) {
        byte oldPawn = get(r, c);
        this.zobristKey ^= zobristTable[r][c][oldPawn]; // Rimuovi pedone vecchio
        this.zobristKey ^= zobristTable[r][c][E];     // Aggiungi pedone vuoto
    }

    private void checkCaptures(int rLast, int cLast, byte movedPawn) {
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        byte opponent = (movedPawn == B) ? W : B;

        for (int[] dir : directions) {
            int dr = dir[0];
            int dc = dir[1];

            int rOpp = rLast + dr;
            int cOpp = cLast + dc;

            if (rOpp < 0 || rOpp >= BOARD_SIZE || cOpp < 0 || cOpp >= BOARD_SIZE) continue;

            byte opponentPawn = get(rOpp, cOpp);

            // 1. C'è un Soldato Avversario?
            if (opponentPawn == opponent) {

                int rWall = rOpp + dr;
                int cWall = cOpp + dc;

                boolean isWall = false;
                if (rWall < 0 || rWall >= BOARD_SIZE || cWall < 0 || cWall >= BOARD_SIZE) {
                    isWall = false; // Bordo del tabellone NON è un muro per i soldati
                } else {
                    byte wallPawn = get(rWall, cWall);
                    // REGOLA ASHTON: Il muro può essere un alleato (movedPawn o K), il Trono (T) o una Cittadella
                    if (wallPawn == movedPawn || (movedPawn == W && wallPawn == K) || wallPawn == T || isCitadel(rWall, cWall)) {
                        isWall = true;
                    }
                }

                if (isWall) {
                    // Aggiorna hash prima di rimuovere
                    updateZobristKeyForRemoval(rOpp, cOpp);

                    if (get(rOpp, cOpp) == W) whitePawnsCount--;
                    if (get(rOpp, cOpp) == B) blackPawnsCount--;
                    set(rOpp, cOpp, E);
                }
            }
            // 2. Check Cattura RE
            else if (opponentPawn == K) {
                if (movedPawn != B) continue; // Solo il Nero può catturare

                int wallCount = 0;
                int[][] kingDirections = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

                for (int[] kDir : kingDirections) {
                    int rKWall = rOpp + kDir[0];
                    int cKWall = cOpp + kDir[1];

                    // REGOLA ASHTON: Il bordo del tabellone NON conta come muro per il Re
                    if (rKWall < 0 || rKWall >= BOARD_SIZE || cKWall < 0 || cKWall >= BOARD_SIZE) {
                        continue;
                    }

                    byte kWall = get(rKWall, cKWall);
                    // Muro per il Re è: Pedone Nero (B), Trono (T), o Cittadella
                    if (kWall == B || kWall == T || isCitadel(rKWall, cKWall)) {
                        wallCount++;
                    }
                }

                boolean isKingCaptured = false;
                boolean isKingOnThrone = (rOpp == THRONE[0] && cOpp == THRONE[1]);
                boolean isKingAdjThrone = !isKingOnThrone &&
                        (Math.abs(rOpp - THRONE[0]) + Math.abs(cOpp - THRONE[1]) == 1);

                if (isKingOnThrone && wallCount == 4) {
                    isKingCaptured = true; // 4 lati sul Trono
                } else if (isKingAdjThrone && wallCount == 3) {
                    isKingCaptured = true; // 3 lati se adiacente al Trono
                } else if (!isKingOnThrone && !isKingAdjThrone && wallCount == 2) {
                    // REGOLA ASHTON: Se il re è in campo aperto (non sul trono o adiacente),
                    // bastano 2 Neri (o 1 Nero e 1 Cittadella/Trono) per catturarlo.
                    // La logica precedente (wallCount) copre già Nero+Cittadella/Trono.
                    // Dobbiamo verificare la cattura 2-lati Nero-Nero.

                    // Controlliamo se è una cattura N-K-N (orizzontale o verticale)
                    int rWall = rOpp + dr;
                    int cWall = cOpp + dc;
                    int rWallOpp = rOpp - dr; // Muro opposto
                    int cWallOpp = cOpp - dc; // Muro opposto

                    if(get(rWall, cWall) == B && get(rWallOpp, cWallOpp) == B) {
                        isKingCaptured = true;
                    }
                }


                isKingCaptured = false; // Reset

                // Controlla i 4 lati del Re (rOpp, cOpp)
                byte north = get(rOpp - 1, cOpp);
                byte south = get(rOpp + 1, cOpp);
                byte west = get(rOpp, cOpp - 1);
                byte east = get(rOpp, cOpp + 1);

                // Muri ostili (Nero, Trono, Cittadella)
                boolean northWall = (north == B || (rOpp - 1 == THRONE[0] && cOpp == THRONE[1]) || isCitadel(rOpp - 1, cOpp));
                boolean southWall = (south == B || (rOpp + 1 == THRONE[0] && cOpp == THRONE[1]) || isCitadel(rOpp + 1, cOpp));
                boolean westWall = (west == B || (rOpp == THRONE[0] && cOpp - 1 == THRONE[1]) || isCitadel(rOpp, cOpp - 1));
                boolean eastWall = (east == B || (rOpp == THRONE[0] && cOpp + 1 == THRONE[1]) || isCitadel(rOpp, cOpp + 1));

                if (isKingOnThrone) {
                    if (northWall && southWall && westWall && eastWall) isKingCaptured = true;
                } else if (isKingAdjThrone) {
                    // Adiacente, es. (3,4) (sopra il trono)
                    if (rOpp == THRONE[0] - 1 && cOpp == THRONE[1]) {
                        if (northWall && westWall && eastWall) isKingCaptured = true;
                    }
                    // (5,4) (sotto il trono)
                    else if (rOpp == THRONE[0] + 1 && cOpp == THRONE[1]) {
                        if (southWall && westWall && eastWall) isKingCaptured = true;
                    }
                    // (4,3) (a ovest del trono)
                    else if (rOpp == THRONE[0] && cOpp == THRONE[1] - 1) {
                        if (northWall && southWall && westWall) isKingCaptured = true;
                    }
                    // (4,5) (a est del trono)
                    else if (rOpp == THRONE[0] && cOpp == THRONE[1] + 1) {
                        if (northWall && southWall && eastWall) isKingCaptured = true;
                    }
                } else {
                    // Caso generale: cattura a 2 lati
                    if (northWall && southWall) isKingCaptured = true; // Cattura verticale
                    if (westWall && eastWall) isKingCaptured = true;   // Cattura orizzontale
                }


                if (isKingCaptured) {
                    // Aggiorna hash prima di rimuovere
                    updateZobristKeyForRemoval(rOpp, cOpp);

                    set(rOpp, cOpp, E);
                    this.kingRow = -1;
                    this.turn = Turn.BLACKWIN;
                }
            }
        }
    }

    /**
     * Genera tutte le mosse legali per il turno corrente.
     * @return Una lista di oggetti Action.
     */
    public List<Action> generateLegalMoves() {
        List<Action> legalMoves = new ArrayList<>();
        byte myPawnType = this.turn.equals(Turn.WHITE) ? W : B;

        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                byte pawn = get(r, c);

                // Controlla se la pedina appartiene al giocatore di turno
                if (pawn == myPawnType || (pawn == K && this.turn.equals(Turn.WHITE))) {

                    int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

                    // Itera per le 4 direzioni
                    for (int[] dir : directions) {
                        int dr = dir[0];
                        int dc = dir[1];

                        // Itera per la lunghezza della mossa
                        for (int steps = 1; steps < BOARD_SIZE; steps++) {
                            int rTo = r + dr * steps;
                            int cTo = c + dc * steps;

                            // Fuori dal tabellone, cambia direzione
                            if (rTo < 0 || rTo >= BOARD_SIZE || cTo < 0 || cTo >= BOARD_SIZE) break;

                            // 1. Controllo ostacoli/destinazione occupata
                            if (get(rTo, cTo) != E) break;

                            // 2. Controllo Percorso (necessario per Trono/Cittadelle bloccanti)
                            // NOTA: isPathClear controlla da (r,c) a (rTo, cTo) ESCLUDENDO (rTo, cTo)
                            if (!isPathClear(r, c, rTo, cTo, pawn)) break;

                            // 3. Controllo Restrizioni di Arrivo (come in applyMove)
                            if (rTo == THRONE[0] && cTo == THRONE[1]) {
                                if (pawn != K) break; // Solo il Re può atterrare sul trono
                            } else if (isCitadel(rTo, cTo)) {
                                if (pawn != K) {
                                    break; // Pedoni (W/B) non possono atterrare su cittadelle
                                } else { // Re (K):
                                    if (!containsCoord(ESCAPES, rTo, cTo)) {
                                        break; // Re non può atterrare su Cittadella NON-Escape
                                    }
                                }
                            }

                            // Se tutti i controlli passano, la mossa è legale
                            try {
                                String from = getBox(r, c);
                                String to = getBox(rTo, cTo);
                                legalMoves.add(new Action(from, to, this.getTurn()));
                            } catch (IOException e) { /* Ignora (non dovrebbe mai accadere) */ }
                        }
                    }
                }
            }
        }
        return legalMoves;
    }


    // --- Metodi di compatibilità Framework ---

    private Pawn byteToPawn(byte b) {
        if (b == W) return Pawn.WHITE;
        if (b == B) return Pawn.BLACK;
        if (b == K) return Pawn.KING;
        if (b == T) return Pawn.THRONE;
        return Pawn.EMPTY;
    }

    @Override
    public Pawn getPawn(int row, int column) {
        return byteToPawn(get(row, column));
    }

    /**
     * Sovrascrive removePawn per aggiornare anche lo Zobrist Key.
     * Usato principalmente dal framework esterno (es. Tester),
     * la logica interna usa checkCaptures.
     */
    @Override
    public void removePawn(int row, int column) {
        byte old = get(row, column);
        if (old == E || old == T) return; // Non rimuovere caselle vuote

        // MODIFICA ZOBRIST: Aggiorna hash prima di rimuovere
        updateZobristKeyForRemoval(row, column);

        if (old == W) whitePawnsCount--;
        if (old == B) blackPawnsCount--;
        if (old == K) kingRow = -1;

        set(row, column, E);
    }

    @Override
    public Pawn[][] getBoard() {
        Pawn[][] board = new Pawn[BOARD_SIZE][BOARD_SIZE];
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                board[i][j] = getPawn(i, j);
            }
        }
        return board;
    }

    public String getBox(int row, int column) {
        char col = (char) (column + 97);
        return col + "" + (row + 1);
    }

    /**
     * Sostituisce l'hashCode() di default con lo Zobrist Key.
     */
    @Override
    public int hashCode() {
        return (int) (this.zobristKey ^ (this.zobristKey >>> 32));
    }

    /**
     * Sostituisce l'equals() di default per usare lo Zobrist Key e il turno.
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || !(obj instanceof FastTablutState)) return false;

        FastTablutState other = (FastTablutState) obj;

        // Confronto veloce
        if (this.zobristKey != other.zobristKey) return false;
        if (this.turn != other.turn) return false;

        // Controllo di sicurezza completo (opzionale se credi ciecamente nell'hash)
        // return Arrays.equals(fastBoard, other.fastBoard);
        return true;
    }
}