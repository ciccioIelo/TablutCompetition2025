package it.unibo.ai.didattica.competition.tablut.client;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.*;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.FastTablutState;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * Contiene la logica di ricerca Alpha-Beta, l'euristica e la gestione della Transposition Table.
 * Implementa la Fase 1 (Motore Veloce) e la Fase 2 (Parallelizzazione Root Node e Timeout).
 */
public class AlphaBetaEngineCopy {

    private static final int BOARD_SIZE = 9;

    // --- VALORI ESTREMI E MARGINI ---
    private static final int MAX_VALUE = 100000;
    private static final int MIN_VALUE = -100000;
    private static final int HEURISTIC_MAX = 50000;
    private static final int HEURISTIC_MIN = -50000;
    private static final int INITIAL_ALPHA = MIN_VALUE - 1000;
    private static final int INITIAL_BETA = MAX_VALUE + 1000;

    // VARIABILE PER I PESI (INIETTABILI - FASE 3.1)
    private final double[] weights;

    // --- TRANSPOSITION TABLE e Helper Classes ---
    private Map<Long, TranspositionEntry> transpositionTable;
    private static final int EXACT_SCORE = 0;
    private static final int LOWER_BOUND = 1;
    private static final int UPPER_BOUND = 2;

    private class TranspositionEntry {
        final int score, depth, nodeType;
        public TranspositionEntry(int score, int depth, int nodeType) {
            this.score = score; this.depth = depth; this.nodeType = nodeType;
        }
        public int getScore() { return score; }
        public int getDepth() { return depth; }
    }

    public class AlphaBetaResult {
        private final int score;
        private final Action action;
        public AlphaBetaResult(int score, Action action) {
            this.score = score; this.action = action;
        }
        public AlphaBetaResult(int score) { this(score, null); }
        public int getScore() { return score; }
        public Action getAction() { return action; }
    }

    private class ActionScore {
        final Action action;
        final int score;
        public ActionScore(Action action, int score) {
            this.action = action; this.score = score;
        }
    }

    private class ActionScoreComparator implements Comparator<ActionScore> {
        private final Turn playerToMove;
        public ActionScoreComparator(Turn playerToMove) { this.playerToMove = playerToMove; }

        @Override
        public int compare(ActionScore a, ActionScore b) {
            return playerToMove.equals(Turn.WHITE) ? Integer.compare(b.score, a.score) : Integer.compare(a.score, b.score);
        }
    }

    private final Turn player;
    private final ExecutorService executorService;
    private static final int N_CPUS = Runtime.getRuntime().availableProcessors();


    // COSTRUTTORE AGGIORNATO: ACCETTA I PESI
    public AlphaBetaEngineCopy(Turn player, double[] weights) {
        this.player = player;
        this.weights = weights; // Usa i pesi iniettati
        this.transpositionTable = new HashMap<>();
        this.executorService = Executors.newFixedThreadPool(N_CPUS);
    }

    // ----------------------------------------------------------------------
    // 1. GENERAZIONE E ORDINAMENTO DELLE MOSSE (METODI REINTEGRATI)
    // ----------------------------------------------------------------------

    private List<Action> getLegalMoves(FastTablutState state) {
        return state.generateLegalMoves();
    }

    /**
     * Reintegra il metodo di ordinamento basato sull'euristica dello stato successivo.
     */
    private List<Action> sortMovesByHeuristic(FastTablutState currentState, List<Action> moves) {
        if (moves.isEmpty()) return Collections.emptyList();

        List<ActionScore> scoredMoves = new ArrayList<>();

        for (Action action : moves) {
            // Per l'ordinamento, simuliamo la mossa in uno stato temporaneo per valutarla
            FastTablutState nextStateForEval = currentState.clone();
            if (nextStateForEval.applyMove(action)) {
                // Valuta lo stato dopo l'applicazione della mossa
                int score = evaluateState(nextStateForEval);
                scoredMoves.add(new ActionScore(action, score));
            }
        }

        if(scoredMoves.isEmpty()) return moves;

        Collections.sort(scoredMoves, new ActionScoreComparator(currentState.getTurn()));

        List<Action> orderedActions = new ArrayList<>();
        for (ActionScore as : scoredMoves) {
            orderedActions.add(as.action);
        }

        return orderedActions;
    }

    // ----------------------------------------------------------------------
    // 2. LOGICA MINIMAX (ITERATIVE DEEPENING e PARALLELIZZAZIONE)
    // ----------------------------------------------------------------------

    public Action getBestMove(FastTablutState currentState, int timeoutSeconds) {

        long startTime = System.currentTimeMillis();
        final long timeLimit = startTime + (timeoutSeconds * 1000L);

        this.transpositionTable.clear();

        List<Action> legalMoves = this.getLegalMoves(currentState);

        if (legalMoves.isEmpty()) {
            System.out.println("ID: Nessuna mossa legale disponibile. Ritorno null.");
            return null;
        }

        Action bestMoveAtCurrentDepth = legalMoves.get(0);
        int bestScoreAtCurrentDepth = evaluateState(currentState);

        int currentDepth = 1;

        while (true) {

            if (System.currentTimeMillis() >= timeLimit) { break; }

            long timeRemaining = timeLimit - System.currentTimeMillis();
            if (timeRemaining <= 100) break;

            int currentIterationBestScore = (this.player.equals(Turn.WHITE)) ? MIN_VALUE : MAX_VALUE;
            Action currentIterationBestMove = bestMoveAtCurrentDepth;

            final int searchDepth = currentDepth; // Variabile final per la lambda

            List<Action> orderedMoves = sortMovesByHeuristic(currentState, legalMoves);

            List<Future<AlphaBetaResult>> futures = new ArrayList<>();

            for (Action action : orderedMoves) {

                Callable<AlphaBetaResult> task = () -> {
                    FastTablutState nextState = currentState.clone();
                    if (!nextState.applyMove(action)) {
                        return new AlphaBetaResult(this.player.equals(Turn.WHITE) ? MIN_VALUE : MAX_VALUE, action);
                    }

                    if (this.player.equals(Turn.WHITE)) {
                        return minValue(nextState, INITIAL_ALPHA, INITIAL_BETA, searchDepth, searchDepth, timeLimit);
                    } else {
                        return maxValue(nextState, INITIAL_ALPHA, INITIAL_BETA, searchDepth, searchDepth, timeLimit);
                    }
                };

                futures.add(executorService.submit(task));
            }

            try {
                int moveIndex = 0;
                for (Future<AlphaBetaResult> future : futures) {

                    timeRemaining = timeLimit - System.currentTimeMillis();
                    if (timeRemaining <= 0) {
                        throw new TimeoutException();
                    }

                    AlphaBetaResult result = future.get(timeRemaining, TimeUnit.MILLISECONDS);
                    Action action = orderedMoves.get(moveIndex);
                    int currentScore = result.getScore();

                    if (this.player.equals(Turn.WHITE)) {
                        if (currentScore > currentIterationBestScore) {
                            currentIterationBestScore = currentScore;
                            currentIterationBestMove = action;
                        }
                    } else {
                        if (currentScore < currentIterationBestScore) {
                            currentIterationBestScore = currentScore;
                            currentIterationBestMove = action;
                        }
                    }

                    moveIndex++;
                }

                bestMoveAtCurrentDepth = currentIterationBestMove;
                bestScoreAtCurrentDepth = currentIterationBestScore;

                System.out.println("ID: Profondità D=" + currentDepth + " COMPLETATA.");
                currentDepth++;

            } catch (TimeoutException | InterruptedException e) {
                System.out.println("ID: Timeout/Interruzione. Uso il miglior risultato da D=" + (currentDepth - 1) + ".");
                for (Future<AlphaBetaResult> future : futures) {
                    future.cancel(true);
                }
                break;
            } catch (ExecutionException e) {
                System.err.println("Errore durante l'esecuzione del thread: " + e.getMessage());
                break;
            }
        }

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.println("--- LOG FINALE (Iterative Deepening) ---");
        System.out.println("INFO: Tempo totale trascorso: " + (totalTime / 1000.0) + "s.");
        System.out.println("INFO: Massima profondità completata: D=" + (currentDepth - 1) + ".");
        System.out.println("INFO: Mossa scelta: " + bestMoveAtCurrentDepth.toString());
        System.out.println("INFO: Punteggio finale della mossa: " + bestScoreAtCurrentDepth);
        System.out.println("------------------------");

        return bestMoveAtCurrentDepth;
    }

    // ----------------------------------------------------------------------
    // 3. MAX VALUE (White) e 4. MIN VALUE (Black)
    // ----------------------------------------------------------------------

    /**
        * Funzione MAX (Giocatore Bianco) potenziata con TT e Ordinamento Mosse Ricorsivo.
    */
    private AlphaBetaResult maxValue(FastTablutState state, int alpha, int beta, int depthRemaining, int currentMaxDepth, long timeLimit) {
        // --- 1. Controllo Timeout e Stati Terminali ---
        if (System.currentTimeMillis() >= timeLimit) { 
            throw new RuntimeException("Timeout reached in Minimax"); 
        }

        if (state.getTurn().equals(Turn.WHITEWIN)) { 
            return new AlphaBetaResult(MAX_VALUE + depthRemaining, null); // Preferisce vittorie veloci
        }
        if (state.getTurn().equals(Turn.BLACKWIN)) { 
            return new AlphaBetaResult(MIN_VALUE - depthRemaining, null); // Preferisce sconfitte lente
        }
        if (depthRemaining == 0) { 
            // Raggiunta la profondità massima, restituisci l'euristica
            return new AlphaBetaResult(evaluateState(state), null); 
        }

        // Salva l'alpha originale per determinare il tipo di nodo per la TT
        int oldAlpha = alpha;
        
        // --- 2. ACCORGIMENTO: TT Lookup (Controllo Cache) ---
        long stateKey = state.hashCode();
        TranspositionEntry entry = transpositionTable.get(stateKey);

        if (entry != null && entry.getDepth() >= depthRemaining) {
            if (entry.nodeType == EXACT_SCORE) {
                // Trovato punteggio esatto, restituiscilo
                return new AlphaBetaResult(entry.getScore(), null);
            } else if (entry.nodeType == LOWER_BOUND) {
                // Il punteggio salvato è un limite inferiore, restringe la finestra
                alpha = Math.max(alpha, entry.getScore()); 
            } else if (entry.nodeType == UPPER_BOUND) {
                // Il punteggio salvato è un limite superiore, restringe la finestra
                beta = Math.min(beta, entry.getScore());
            }

            // Se la finestra si chiude, abbiamo un taglio
            if (alpha >= beta) {
                return new AlphaBetaResult(entry.getScore(), null);
            }
        }

        // --- 3. Generazione e Ordinamento Mosse ---
        List<Action> possibleActions = this.getLegalMoves(state);

        // --- BUGFIX: Gestione "Nessuna Mossa" ---
        if (possibleActions.isEmpty()) {
            // Il Bianco non può muovere, è una sconfitta
            return new AlphaBetaResult(MIN_VALUE - depthRemaining, null); 
        }

        // --- 4. ACCORGIMENTO: Ordinamento Mosse Ricorsivo ---
        List<Action> orderedActions = this.sortMovesByHeuristic(state, possibleActions);
        
        int maxScore = MIN_VALUE;
        Action bestMove = orderedActions.get(0); // Inizializza con una mossa valida

        // --- 5. Loop di Ricerca ---
        for (Action action : orderedActions) {
            FastTablutState nextState = state.clone();
            if (!nextState.applyMove(action)) { continue; } // Mossa non valida (dovrebbe essere già filtrata)

            AlphaBetaResult result = minValue(nextState, alpha, beta, depthRemaining - 1, currentMaxDepth, timeLimit);
            
            if (result.getScore() > maxScore) {
                maxScore = result.getScore();
                bestMove = action; // (Necessario solo al nodo radice, ma utile per il debug)
            }
            
            alpha = Math.max(alpha, maxScore); // Aggiorna il limite inferiore

            if (alpha >= beta) {
                // Taglio Beta: Il Nero ha una mossa migliore altrove
                break; 
            }
        }

        // --- 6. ACCORGIMENTO: TT Store (Salva in Cache) ---
        int nodeType;
        if (maxScore <= oldAlpha) {
            nodeType = UPPER_BOUND; // Non siamo riusciti a migliorare alpha, punteggio è <= oldAlpha
        } else if (maxScore >= beta) {
            nodeType = LOWER_BOUND; // Abbiamo causato un taglio beta, punteggio è >= beta
        } else {
            nodeType = EXACT_SCORE; // Trovato un punteggio esatto nella finestra
        }
        
        transpositionTable.put(stateKey, new TranspositionEntry(maxScore, depthRemaining, nodeType));
        
        return new AlphaBetaResult(maxScore, bestMove); // Ritorna il punteggio e la mossa
    }

    private AlphaBetaResult minValue(FastTablutState state, int alpha, int beta, int depthRemaining,
            int currentMaxDepth, long timeLimit) {
        // --- 1. Controllo Timeout e Stati Terminali ---
        if (System.currentTimeMillis() >= timeLimit) {
            throw new RuntimeException("Timeout reached in Minimax");
        }

        // I controlli terminali sono identici (la prospettiva del punteggio è assoluta)
        if (state.getTurn().equals(Turn.WHITEWIN)) {
            return new AlphaBetaResult(MAX_VALUE + depthRemaining, null);
        }
        if (state.getTurn().equals(Turn.BLACKWIN)) {
            return new AlphaBetaResult(MIN_VALUE - depthRemaining, null);
        }
        if (depthRemaining == 0) {
            return new AlphaBetaResult(evaluateState(state), null);
        }

        // Salva la beta originale per determinare il tipo di nodo per la TT
        int oldBeta = beta;
        // --- 2. ACCORGIMENTO: TT Lookup (Controllo Cache) ---
        // (Assicurati di usare il tipo di chiave corretto, es. Long o Integer)
        long stateKey = state.hashCode(); // o la tua chiave Zobrist
        TranspositionEntry entry = transpositionTable.get(stateKey);

        if (entry != null && entry.getDepth() >= depthRemaining) {
            // La logica di lookup è identica a maxValue
            if (entry.nodeType == EXACT_SCORE) {
                return new AlphaBetaResult(entry.getScore(), null);
            } else if (entry.nodeType == LOWER_BOUND) {
                alpha = Math.max(alpha, entry.getScore());
            } else if (entry.nodeType == UPPER_BOUND) {
                beta = Math.min(beta, entry.getScore());
            }
            if (alpha >= beta) {
                return new AlphaBetaResult(entry.getScore(), null);
            }
        }

        // --- 3. Generazione e Ordinamento Mosse ---
        List<Action> possibleActions = this.getLegalMoves(state);

        // --- BUGFIX: Gestione "Nessuna Mossa" ---
        if (possibleActions.isEmpty()) {
            // Il Nero (Min) non può muovere, è una VITTORIA per il Bianco (Max)
            return new AlphaBetaResult(MAX_VALUE + depthRemaining, null);
        }

        // --- 4. ACCORGIMENTO: Ordinamento Mosse Ricorsivo ---
        List<Action> orderedActions = this.sortMovesByHeuristic(state, possibleActions);

        int minScore = MAX_VALUE; // Inizializza al valore più ALTO
        Action bestMove = orderedActions.get(0); // Inizializza con una mossa valida

        // --- 5. Loop di Ricerca ---
        for (Action action : orderedActions) {
            FastTablutState nextState = state.clone();
            if (!nextState.applyMove(action)) {
                continue;
            }

            // Chiama la funzione opposta (maxValue)
            AlphaBetaResult result = maxValue(nextState, alpha, beta, depthRemaining - 1, currentMaxDepth, timeLimit);

            if (result.getScore() < minScore) {
                minScore = result.getScore();
                bestMove = action;
            }

            beta = Math.min(beta, minScore); // Aggiorna il limite superiore

            if (minScore <= alpha) {
                // Taglio Alpha: Il Bianco ha già una mossa migliore
                break;
            }
        }

        // --- 6. ACCORGIMENTO: TT Store (Salva in Cache) ---
        // La logica di store è speculare a maxValue
        int nodeType;
        if (minScore <= alpha) {
            nodeType = UPPER_BOUND; // Abbiamo causato un taglio alpha, il punteggio è <= alpha
        } else if (minScore >= oldBeta) {
            nodeType = LOWER_BOUND; // Non siamo riusciti a migliorare beta, il punteggio è >= oldBeta
        } else {
            nodeType = EXACT_SCORE; // Trovato un punteggio esatto nella finestra
        }

        transpositionTable.put(stateKey, new TranspositionEntry(minScore, depthRemaining, nodeType));

        return new AlphaBetaResult(minScore, bestMove);
    }

    // ----------------------------------------------------------------------
    // 5. FUNZIONE EURISTICA (USA I PESI MEMBRI)
    // ----------------------------------------------------------------------

    private boolean containsCoord(List<int[]> list, int r, int c) {
        for (int[] coord : list) { if (coord[0] == r && coord[1] == c) return true; }
        return false;
    }

    private int getMinEscapeDistance(int kingR, int kingC) {
        int minDistance = Integer.MAX_VALUE;
        for (int[] escape : FastTablutState.ESCAPES) {
            int distance = Math.abs(kingR - escape[0]) + Math.abs(kingC - escape[1]);
            minDistance = Math.min(minDistance, distance);
        }
        return minDistance;
    }

    private int evaluateState(FastTablutState state) {
        if (state.getTurn().equals(Turn.WHITEWIN)) { return MAX_VALUE; }
        if (state.getTurn().equals(Turn.BLACKWIN)) { return MIN_VALUE; }

        int whiteCount = state.whitePawnsCount;
        int blackCount = state.blackPawnsCount;
        int kingR = state.kingRow;
        int kingC = state.kingCol;

        if (kingR == -1) { return MIN_VALUE; }

        double kingPositionScore = evalKingPos(state, kingR, kingC);
        double escapeDistancePenalty = this.weights[11] * getMinEscapeDistance(kingR, kingC);
        double materialScore = this.weights[9] * whiteCount + this.weights[10] * blackCount;

        double totalScore = this.weights[7] * materialScore +
                this.weights[8] * (kingPositionScore + escapeDistancePenalty);

        int finalScore = (int) Math.round(totalScore);
        finalScore = Math.min(finalScore, HEURISTIC_MAX);
        finalScore = Math.max(finalScore, HEURISTIC_MIN);

        return finalScore;
    }

    private double evalKingPos(FastTablutState state, int kingR, int kingC) {
        double score = 0;
        int[][] directions = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

        for (int[] dir : directions) {
            int dr = dir[0];
            int dc = dir[1];

            for (int steps = 1; steps < BOARD_SIZE; steps++) {
                int r = kingR + dr * steps;
                int c = kingC + dc * steps;

                if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) { break; }

                byte currentPawn = state.get(r, c);
                boolean isAdjacent = (steps == 1);

                if (currentPawn == FastTablutState.E && containsCoord(FastTablutState.ESCAPES, r, c)) {
                    score += this.weights[0]; break;
                }

                if (currentPawn == FastTablutState.E && isCitadelOrThrone(r, c)) {
                    if (r == FastTablutState.THRONE[0] && c == FastTablutState.THRONE[1]) {
                        score += this.weights[2];
                    } else if (containsCoord(FastTablutState.CITADELS, r, c)) {
                        score += this.weights[1];
                    }
                    continue;
                }

                if (currentPawn == FastTablutState.B) {
                    score += isAdjacent ? this.weights[6] : this.weights[5]; break;
                }

                if (currentPawn == FastTablutState.W || currentPawn == FastTablutState.K) {
                    score += isAdjacent ? this.weights[4] : this.weights[3]; break;
                }
            }
        }
        return score;
    }

    private boolean isCitadelOrThrone(int r, int c) {
        if (r == FastTablutState.THRONE[0] && c == FastTablutState.THRONE[1]) return true;
        return containsCoord(FastTablutState.CITADELS, r, c);
    }
}