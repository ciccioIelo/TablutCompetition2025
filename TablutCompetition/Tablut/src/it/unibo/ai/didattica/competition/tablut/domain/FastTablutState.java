package it.unibo.ai.didattica.competition.tablut.domain;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.io.IOException;

import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * Stato del gioco ottimizzato per le performance (Fase 1 della Roadmap).
 * Implementa la logica completa di movimento e cattura delle regole Ashton Tablut.
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
        if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) return E;
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

        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                Pawn pawn = state.getPawn(r, c);
                byte bytePawn = PAWN_TO_BYTE.getOrDefault(pawn, E);

                fastState.set(r, c, bytePawn);

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
        return newState;
    }

    // ----------------------------------------------------------------------
    // LOGICA DI GIOCO OTTIMIZZATA (FASE 1)
    // ----------------------------------------------------------------------

    private boolean isPathClear(int r1, int c1, int r2, int c2, byte movingPawn) {
        int dr = Integer.signum(r2 - r1);
        int dc = Integer.signum(c2 - c1);

        // Controlla ogni casella *tra* la partenza e l'arrivo
        for (int r = r1 + dr, c = c1 + dc; r != r2 || c != c2; r += dr, c += dc) {
            byte currentPawn = get(r, c);

            // Blocco 1: Qualsiasi pedina (escluso il Trono che è un 'T' ma è vuoto)
            if (currentPawn != E && currentPawn != T) return false;

            // Blocco 2: Non puoi scavalcare il Trono o una Cittadella
            if (currentPawn == T && movingPawn != K) return false;
            if (isCitadel(r, c) && get(r, c) == E && movingPawn != K) return false;
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

        if (rFrom == rTo && cFrom == cTo) return false;
        if (rFrom != rTo && cFrom != cTo) return false; // Diagonale

        byte pawn = get(rFrom, cFrom);
        if (pawn == E || pawn == T) return false;

        // 1. Validazione del pezzo
        if (this.turn.equals(Turn.WHITE) && (pawn != W && pawn != K)) return false;
        if (this.turn.equals(Turn.BLACK) && (pawn != B)) return false;

        // 2. Controllo di destinazione
        if (get(rTo, cTo) != E) return false;
        if (rTo == THRONE[0] && cTo == THRONE[1] && pawn != K) return false;

        // 3. Controllo Cittadelle
        if (isCitadel(rTo, cTo) && pawn != K) {
            if (!isCitadel(rFrom, cFrom)) return false; // Non può entrare se non viene da una Cittadella
            if (isCitadel(rFrom, cFrom) && (Math.abs(rTo - rFrom) > 1 || Math.abs(cTo - cFrom) > 1)) return false; // Max 1 passo se in Cittadella
        }

        // 4. Controllo del percorso (Climbing)
        if (!isPathClear(rFrom, cFrom, rTo, cTo, pawn)) return false;

        // 5. Esegui la Mossa
        set(rFrom, cFrom, (rFrom == THRONE[0] && cFrom == THRONE[1]) ? T : E);
        set(rTo, cTo, pawn);

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

        // 7. Controlla Sconfitta/Vittoria
        if (this.turn != Turn.WHITEWIN && this.kingRow == -1) {
            this.turn = Turn.BLACKWIN;
        }
        if (this.turn != Turn.WHITEWIN && this.turn != Turn.BLACKWIN) {
            if (this.whitePawnsCount == 0 && this.kingRow == -1) { this.turn = Turn.BLACKWIN; }
            if (this.blackPawnsCount == 0) { this.turn = Turn.WHITEWIN; }
        }

        // 8. Cambia Turno
        if (this.turn != Turn.WHITEWIN && this.turn != Turn.BLACKWIN) {
            if (this.turn.equals(Turn.WHITE)) this.turn = Turn.BLACK;
            else if (this.turn.equals(Turn.BLACK)) this.turn = Turn.WHITE;
        }

        return true;
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

            // 1. C'è un Re o un Soldato Avversario?
            if (get(rOpp, cOpp) == opponent) {

                int rWall = rOpp + dr;
                int cWall = cOpp + dc;

                boolean isWall = false;
                if (rWall < 0 || rWall >= BOARD_SIZE || cWall < 0 || cWall >= BOARD_SIZE) {
                    isWall = true; // Bordo del tabellone
                } else {
                    byte wallPawn = get(rWall, cWall);
                    // Parete: tuo pezzo, Trono (T), Cittadella, o la casella da cui proviene il mosso (rLast, cLast)
                    if (wallPawn == movedPawn || wallPawn == K || (rWall == THRONE[0] && cWall == THRONE[1]) || isCitadel(rWall, cWall)) {
                        isWall = true;
                    }
                }

                if (isWall) {
                    // Cattura pezzo
                    if (get(rOpp, cOpp) == W) whitePawnsCount--;
                    if (get(rOpp, cOpp) == B) blackPawnsCount--;
                    set(rOpp, cOpp, E);
                }
            }
            // 2. Check Cattura RE
            else if (get(rOpp, cOpp) == K) {
                if (movedPawn != B) continue; // Solo il Nero può catturare

                int wallCount = 0;
                int[][] kingDirections = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

                for (int[] kDir : kingDirections) {
                    int rKWall = rOpp + kDir[0];
                    int cKWall = cOpp + kDir[1];

                    if (rKWall < 0 || rKWall >= BOARD_SIZE || cKWall < 0 || cKWall >= BOARD_SIZE) {
                        wallCount++; // Bordo
                        continue;
                    }

                    byte kWall = get(rKWall, cKWall);
                    // Parete: Pezzo Nero (B), Trono (T), Cittadella
                    if (kWall == B || (rKWall == THRONE[0] && cKWall == THRONE[1]) || isCitadel(rKWall, cKWall)) {
                        wallCount++;
                    }
                }

                boolean isKingCaptured = false;

                if (rOpp == THRONE[0] && cOpp == THRONE[1] && wallCount == 4) {
                    isKingCaptured = true; // 4 lati sul Trono
                } else if (wallCount >= 3 && (rOpp == THRONE[0] || cOpp == THRONE[1])) {
                    isKingCaptured = true; // 3 lati se adiacente al Trono/Cittadella
                } else if (wallCount == 4) {
                    isKingCaptured = true; // 4 lati ovunque
                }

                if (isKingCaptured) {
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

                if (pawn == myPawnType || (pawn == K && this.turn.equals(Turn.WHITE))) {

                    int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

                    for (int[] dir : directions) {
                        int dr = dir[0];
                        int dc = dir[1];

                        for (int steps = 1; steps < BOARD_SIZE; steps++) {
                            int rTo = r + dr * steps;
                            int cTo = c + dc * steps;

                            if (rTo < 0 || rTo >= BOARD_SIZE || cTo < 0 || cTo >= BOARD_SIZE) break;

                            // 1. Controllo ostacoli intermedi (Climbing)
                            if (steps > 1 && get(rTo, cTo) != E) {
                                // Se la destinazione è occupata, e non è il primo passo, fermati
                                if (get(r + dr * (steps - 1), c + dc * (steps - 1)) != E) break;
                            }

                            // Controllo: la casella di arrivo deve essere libera o una destinazione valida
                            if (get(rTo, cTo) != E) break;

                            // 2. Controllo Restrizioni di Arrivo
                            if (rTo == THRONE[0] && cTo == THRONE[1] && pawn != K) break;
                            if (isCitadel(rTo, cTo) && pawn != K) {
                                if (!isCitadel(r, c)) break; // Non può entrare da fuori Cittadella
                                if (isCitadel(r, c) && steps > 1) break; // Massima 1 casella se interno
                            }

                            // 3. Controllo Percorso (necessario per trono/cittadelle attraversate)
                            if (!isPathClear(r, c, rTo, cTo, pawn)) break;

                            // Se tutti i controlli passano, la mossa è legale
                            try {
                                String from = getBox(r, c);
                                String to = getBox(rTo, cTo);
                                legalMoves.add(new Action(from, to, this.getTurn()));
                            } catch (IOException e) { /* Ignora */ }

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

    @Override
    public void removePawn(int row, int column) {
        byte old = get(row, column);
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

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + Arrays.hashCode(fastBoard);
        result = prime * result + ((this.turn == null) ? 0 : this.turn.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        FastTablutState other = (FastTablutState) obj;
        if (!Arrays.equals(fastBoard, other.fastBoard)) return false;
        return turn == other.turn;
    }
}