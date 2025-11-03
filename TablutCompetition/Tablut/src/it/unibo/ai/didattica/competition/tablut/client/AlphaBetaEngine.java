package it.unibo.ai.didattica.competition.tablut.client;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.*; // Import per la Parallelizzazione
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.FastTablutState;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * Contiene la logica di ricerca Alpha-Beta, l'euristica e la gestione della Transposition Table.
 * Implementa la Fase 1 (Motore Veloce) e la Fase 2 (Parallelizzazione Root Node e Timeout).
 */
public class AlphaBetaEngine {

    private static final int BOARD_SIZE = 9;

    // --- VALORI ESTREMI E MARGINI ---
    private static final int MAX_VALUE = 100000;
    private static final int MIN_VALUE = -100000;
    private static final int HEURISTIC_MAX = 50000;
    private static final int HEURISTIC_MIN = -50000;
    private static final int INITIAL_ALPHA = MIN_VALUE - 1000;
    private static final int INITIAL_BETA = MAX_VALUE + 1000;

    // --- PESI EURISTICI BILANCIATI (Estratti dall'agente originale) ---
    private static final double[] WEIGHTS = {
            5000.0, -300.0, -500.0, -50.0, -150.0, 100.0, -800.0, 1.0, 1.0, 80.0, -60.0, -200.0
    };

    // --- TRANSPOSITION TABLE e Helper Classes ---
    private Map<String, TranspositionEntry> transpositionTable;
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


    public AlphaBetaEngine(Turn player) {
        this.player = player;
        this.transpositionTable = new HashMap<>();
        this.executorService = Executors.newFixedThreadPool(N_CPUS);
    }

    // ----------------------------------------------------------------------
    // 1. GENERAZIONE E ORDINAMENTO DELLE MOSSE
    // ----------------------------------------------------------------------

    private List<Action> getLegalMoves(FastTablutState state) {
        return state.generateLegalMoves();
    }

    private List<Action> sortMovesByHeuristic(FastTablutState currentState, List<Action> moves) {
        if (moves.isEmpty()) return Collections.emptyList();

        List<ActionScore> scoredMoves = new ArrayList<>();

        for (Action action : moves) {
            FastTablutState nextStateForEval = currentState.clone();
            if (nextStateForEval.applyMove(action)) {
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
        final long timeLimit = startTime + (timeoutSeconds * 1000L); // timeLimit is implicitly final

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

            // CATTURA IL VALORE CORRENTE di currentDepth in una variabile final
            final int searchDepth = currentDepth;

            // Ordina le mosse per la Root Node (MVS - Most Valuable Strategy)
            List<Action> orderedMoves = sortMovesByHeuristic(currentState, legalMoves);

            // --- PARALLELIZZAZIONE DEL ROOT NODE (Fase 2) ---
            List<Future<AlphaBetaResult>> futures = new ArrayList<>();

            for (Action action : orderedMoves) {

                // Crea un Task Callable per ogni mossa candidata
                Callable<AlphaBetaResult> task = () -> {
                    FastTablutState nextState = currentState.clone();
                    if (!nextState.applyMove(action)) {
                        return new AlphaBetaResult(this.player.equals(Turn.WHITE) ? MIN_VALUE : MAX_VALUE, action);
                    }

                    // Usiamo searchDepth (final) anziché currentDepth
                    if (this.player.equals(Turn.WHITE)) {
                        return minValue(nextState, INITIAL_ALPHA, INITIAL_BETA, searchDepth, searchDepth, timeLimit);
                    } else {
                        return maxValue(nextState, INITIAL_ALPHA, INITIAL_BETA, searchDepth, searchDepth, timeLimit);
                    }
                };

                futures.add(executorService.submit(task));
            }

            try {
                // Raccogli i risultati con la gestione del timeout
                int moveIndex = 0;
                for (Future<AlphaBetaResult> future : futures) {

                    timeRemaining = timeLimit - System.currentTimeMillis();
                    if (timeRemaining <= 0) {
                        throw new TimeoutException();
                    }

                    AlphaBetaResult result = future.get(timeRemaining, TimeUnit.MILLISECONDS);
                    Action action = orderedMoves.get(moveIndex);
                    int currentScore = result.getScore();

                    // Aggiorna il miglior punteggio (Logica Root Node)
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

                // Se completata senza eccezioni, aggiorna il risultato globale
                bestMoveAtCurrentDepth = currentIterationBestMove;
                bestScoreAtCurrentDepth = currentIterationBestScore;

                System.out.println("ID: Profondita' D=" + currentDepth + " COMPLETATA.");
                currentDepth++; // Aggiorna currentDepth per l'iterazione successiva

            } catch (TimeoutException | InterruptedException e) {
                // Timeout o Interruzione: Ferma i thread rimanenti
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

        // Log finale (con il punteggio richiesto)
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
    // 3. MAX VALUE (White) e 4. MIN VALUE (Black) (Logica ricorsiva con Interruzione)
    // ----------------------------------------------------------------------

    private AlphaBetaResult maxValue(FastTablutState state, int alpha, int beta, int depthRemaining, int currentMaxDepth, long timeLimit) {
        if (System.currentTimeMillis() >= timeLimit) { throw new RuntimeException("Timeout reached in Minimax"); }

        if (state.getTurn().equals(Turn.WHITEWIN)) { return new AlphaBetaResult(MAX_VALUE + depthRemaining, null); }
        if (state.getTurn().equals(Turn.BLACKWIN)) { return new AlphaBetaResult(MIN_VALUE - depthRemaining, null); }
        if (depthRemaining == 0) { return new AlphaBetaResult(evaluateState(state), null); }

        int maxScore = MIN_VALUE;
        List<Action> possibleActions = this.getLegalMoves(state);

        if (possibleActions.isEmpty()) { return new AlphaBetaResult(evaluateState(state), null); }

        for (Action action : possibleActions) {
            FastTablutState nextState = state.clone();
            if (!nextState.applyMove(action)) { continue; }

            AlphaBetaResult result = minValue(nextState, alpha, beta, depthRemaining - 1, currentMaxDepth, timeLimit);
            maxScore = Math.max(maxScore, result.getScore());

            if (maxScore >= beta) { return new AlphaBetaResult(maxScore, action); }
            alpha = Math.max(alpha, maxScore);
        }
        return new AlphaBetaResult(maxScore, null);
    }

    private AlphaBetaResult minValue(FastTablutState state, int alpha, int beta, int depthRemaining, int currentMaxDepth, long timeLimit) {
        if (System.currentTimeMillis() >= timeLimit) { throw new RuntimeException("Timeout reached in Minimax"); }

        if (state.getTurn().equals(Turn.WHITEWIN)) { return new AlphaBetaResult(MAX_VALUE + depthRemaining, null); }
        if (state.getTurn().equals(Turn.BLACKWIN)) { return new AlphaBetaResult(MIN_VALUE - depthRemaining, null); }
        if (depthRemaining == 0) { return new AlphaBetaResult(evaluateState(state), null); }

        int minScore = MAX_VALUE;
        List<Action> possibleActions = this.getLegalMoves(state);

        if (possibleActions.isEmpty()) { return new AlphaBetaResult(evaluateState(state), null); }

        for (Action action : possibleActions) {
            FastTablutState nextState = state.clone();
            if (!nextState.applyMove(action)) { continue; }

            AlphaBetaResult result = maxValue(nextState, alpha, beta, depthRemaining - 1, currentMaxDepth, timeLimit);
            minScore = Math.min(minScore, result.getScore());

            if (minScore <= alpha) { return new AlphaBetaResult(minScore, action); }
            beta = Math.min(beta, minScore);
        }
        return new AlphaBetaResult(minScore, null);
    }

    // ----------------------------------------------------------------------
    // 5. FUNZIONE EURISTICA e Helper Methods (omessi per brevità)
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
        double escapeDistancePenalty = WEIGHTS[11] * getMinEscapeDistance(kingR, kingC);
        double materialScore = WEIGHTS[9] * whiteCount + WEIGHTS[10] * blackCount;

        double totalScore = WEIGHTS[7] * materialScore +
                WEIGHTS[8] * (kingPositionScore + escapeDistancePenalty);

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
                    score += WEIGHTS[0]; break;
                }

                if (currentPawn == FastTablutState.E && isCitadelOrThrone(r, c)) {
                    if (r == FastTablutState.THRONE[0] && c == FastTablutState.THRONE[1]) {
                        score += WEIGHTS[2];
                    } else if (containsCoord(FastTablutState.CITADELS, r, c)) {
                        score += WEIGHTS[1];
                    }
                    continue;
                }

                if (currentPawn == FastTablutState.B) {
                    score += isAdjacent ? WEIGHTS[6] : WEIGHTS[5]; break;
                }

                if (currentPawn == FastTablutState.W || currentPawn == FastTablutState.K) {
                    score += isAdjacent ? WEIGHTS[4] : WEIGHTS[3]; break;
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