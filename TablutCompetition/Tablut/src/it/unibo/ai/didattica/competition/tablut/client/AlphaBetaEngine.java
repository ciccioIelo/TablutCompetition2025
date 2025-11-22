package it.unibo.ai.didattica.competition.tablut.client;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.*;
import java.util.stream.Collectors;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.FastTablutState;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * Contiene la logica di ricerca Alpha-Beta, l'euristica e la gestione della Transposition Table.
 * * MODIFICATO (Fase 1): Utilizza Zobrist Hashing (long) per la Transposition Table.
 * * MODIFICATO (Fase 2): Implementata Quiescence Search.
 * * MODIFICATO (Fase 3): Euristica migliorata con "King Safety".
 * * MODIFICATO (Fase 4): Move Ordering basato sulla mossa migliore della TT.
 * * MODIFICATO (Fase 5): Aggiunto limite di profondità alla Quiescence Search.
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

    // --------- INIZIO MODIFICA FASE 5 ---------
    /**
     * Profondità massima per la ricerca di quiete.
     * Limita la ricerca di catture. Impostato a 2 per evitare timeout.
     */
    private static final int MAX_QUIESCENCE_DEPTH = 2;
    // --------- FINE MODIFICA FASE 5 ---------


    // VARIABILE PER I PESI (INIETTABILI)
    private final double[] weights;

    // --- TRANSPOSITION TABLE e Helper Classes ---
    private Map<Long, TranspositionEntry> transpositionTable;
    private static final int EXACT_SCORE = 0;
    private static final int LOWER_BOUND = 1;
    private static final int UPPER_BOUND = 2;

    // Array statico per le 8 direzioni adiacenti (incluse diagonali)
    private static final int[][] ADJACENT_DIRECTIONS = {
            {-1, -1}, {-1, 0}, {-1, 1},
            { 0, -1},          { 0, 1},
            { 1, -1}, { 1, 0}, { 1, 1}
    };

    private class TranspositionEntry {
        final int score, depth, nodeType;
        final Action bestMove; // Mossa che ha generato questo punteggio

        public TranspositionEntry(int score, int depth, int nodeType, Action bestMove) {
            this.score = score; this.depth = depth; this.nodeType = nodeType;
            this.bestMove = bestMove; // Salva la mossa
        }
        public int getScore() { return score; }
        public int getDepth() { return depth; }
        public Action getBestMove() { return bestMove; } // Getter per la mossa
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
    public AlphaBetaEngine(Turn player, double[] weights) {
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
        final long timeLimit = startTime + (timeoutSeconds * 1000L) - 200L;

        this.transpositionTable.clear();

        List<Action> legalMoves = this.getLegalMoves(currentState);

        if (legalMoves.isEmpty()) {
            System.out.println("ID: Nessuna mossa legale disponibile. Ritorno null.");
            return null;
        }
        if (legalMoves.size() == 1) {
            System.out.println("ID: Solo una mossa legale disponibile. Ritorno: " + legalMoves.get(0));
            return legalMoves.get(0);
        }


        Action bestMoveAtCurrentDepth = legalMoves.get(0);
        int bestScoreAtCurrentDepth = evaluateState(currentState);

        int currentDepth = 1;

        while (true) {

            if (System.currentTimeMillis() >= timeLimit) {
                System.out.println("ID: Tempo limite raggiunto prima di iniziare D=" + currentDepth);
                break;
            }

            long timeRemaining = timeLimit - System.currentTimeMillis();
            if (timeRemaining <= 0) break; // Controllo extra

            int currentIterationBestScore = (this.player.equals(Turn.WHITE)) ? MIN_VALUE : MAX_VALUE;
            Action currentIterationBestMove = bestMoveAtCurrentDepth;

            final int searchDepth = currentDepth; // Variabile final per la lambda

            final Action bestActionFromLastIteration = bestMoveAtCurrentDepth;
            legalMoves.sort((a1, a2) -> {
                if (a1.equals(bestActionFromLastIteration)) return -1;
                if (a2.equals(bestActionFromLastIteration)) return 1;
                return 0;
            });
            List<Action> orderedMoves = sortMovesByHeuristic(currentState, legalMoves);


            List<Future<AlphaBetaResult>> futures = new ArrayList<>();

            for (Action action : orderedMoves) {

                Callable<AlphaBetaResult> task = () -> {
                    if (System.currentTimeMillis() >= timeLimit) {
                        throw new TimeoutException("Timeout prima dell'esecuzione del task");
                    }

                    FastTablutState nextState = currentState.clone();
                    if (!nextState.applyMove(action)) {
                        return new AlphaBetaResult(this.player.equals(Turn.WHITE) ? MIN_VALUE : MAX_VALUE, action);
                    }

                    AlphaBetaResult result;
                    if (this.player.equals(Turn.WHITE)) {
                        result = minValue(nextState, INITIAL_ALPHA, INITIAL_BETA, searchDepth - 1, searchDepth, timeLimit);
                    } else {
                        result = maxValue(nextState, INITIAL_ALPHA, INITIAL_BETA, searchDepth - 1, searchDepth, timeLimit);
                    }
                    return new AlphaBetaResult(result.getScore(), action);
                };

                futures.add(executorService.submit(task));
            }

            try {
                for (Future<AlphaBetaResult> future : futures) {

                    timeRemaining = timeLimit - System.currentTimeMillis();
                    if (timeRemaining <= 0) {
                        throw new TimeoutException("Timeout durante l'attesa dei futures");
                    }

                    AlphaBetaResult result = future.get(timeRemaining, TimeUnit.MILLISECONDS);

                    Action action = result.getAction();
                    int currentScore = result.getScore();

                    if (action == null) continue;

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
                }

                if (System.currentTimeMillis() < timeLimit) {
                    bestMoveAtCurrentDepth = currentIterationBestMove;
                    bestScoreAtCurrentDepth = currentIterationBestScore;

                    //System.out.println("ID: Profondità D=" + currentDepth + " COMPLETATA. Mossa: " + bestMoveAtCurrentDepth.toString() + " Punteggio: " + bestScoreAtCurrentDepth);
                    currentDepth++;
                } else {
                    //System.out.println("ID: Timeout durante il completamento di D=" + currentDepth + ". Uso D=" + (currentDepth - 1));
                    break;
                }


            } catch (TimeoutException | InterruptedException | CancellationException e) {
                System.out.println("ID: Timeout/Interruzione/Cancellazione. Uso il miglior risultato da D=" + (currentDepth - 1) + ".");
                for (Future<AlphaBetaResult> future : futures) {
                    future.cancel(true);
                }
                break;
            } catch (ExecutionException e) {
                System.err.println("Errore durante l'esecuzione del thread: " + e.getMessage());
                break;
            }
        }

        //long totalTime = System.currentTimeMillis() - startTime;
        // System.out.println("--- LOG FINALE (Iterative Deepening) ---");
        // System.out.println("INFO: Tempo totale trascorso: " + (totalTime / 1000.0) + "s.");
        // System.out.println("INFO: Massima profondità completata: D=" + (currentDepth - 1) + ".");
        // System.out.println("INFO: Mossa scelta: " + bestMoveAtCurrentDepth.toString());
        // System.out.println("INFO: Punteggio finale della mossa: " + bestScoreAtCurrentDepth);
        // System.out.println("------------------------");

        return bestMoveAtCurrentDepth;
    }

    // ----------------------------------------------------------------------
    // 3. MAX VALUE (White) e 4. MIN VALUE (Black)
    // ----------------------------------------------------------------------

    private AlphaBetaResult maxValue(FastTablutState state, int alpha, int beta, int depthRemaining, int currentMaxDepth, long timeLimit) {
        if (System.currentTimeMillis() >= timeLimit) {
            throw new RuntimeException("Timeout reached in MaxValue");
        }

        if (state.getTurn().equals(Turn.WHITEWIN)) {
            return new AlphaBetaResult(MAX_VALUE + depthRemaining, null);
        }
        if (state.getTurn().equals(Turn.BLACKWIN)) {
            return new AlphaBetaResult(MIN_VALUE - depthRemaining, null);
        }

        // --------- INIZIO MODIFICA FASE 5 ---------
        if (depthRemaining == 0) {
            // Chiama quiescence con la profondità massima di quiete
            return quiescenceSearch(state, alpha, beta, timeLimit, MAX_QUIESCENCE_DEPTH);
        }
        // --------- FINE MODIFICA FASE 5 ---------

        int oldAlpha = alpha;

        long stateKey = state.getZobristKey();
        TranspositionEntry entry = transpositionTable.get(stateKey);

        Action ttBestMove = null;

        if (entry != null && entry.getDepth() >= depthRemaining) {
            ttBestMove = entry.getBestMove();

            if (entry.nodeType == EXACT_SCORE) {
                return new AlphaBetaResult(entry.getScore(), ttBestMove);
            } else if (entry.nodeType == LOWER_BOUND) {
                alpha = Math.max(alpha, entry.getScore());
            } else if (entry.nodeType == UPPER_BOUND) {
                beta = Math.min(beta, entry.getScore());
            }
            if (alpha >= beta) {
                return new AlphaBetaResult(entry.getScore(), ttBestMove);
            }
        }

        List<Action> possibleActions = this.getLegalMoves(state);

        if (possibleActions.isEmpty()) {
            return new AlphaBetaResult(MIN_VALUE - depthRemaining, null);
        }

        int maxScore = MIN_VALUE;
        Action bestMove = possibleActions.get(0); // Default

        if (ttBestMove != null) {
            FastTablutState nextState = state.clone();
            if (nextState.applyMove(ttBestMove)) {
                AlphaBetaResult result = minValue(nextState, alpha, beta, depthRemaining - 1, currentMaxDepth, timeLimit);

                if (result.getScore() > maxScore) {
                    maxScore = result.getScore();
                    bestMove = ttBestMove;
                }
                alpha = Math.max(alpha, maxScore);
            }
        }

        if (alpha < beta) {
            for (Action action : possibleActions) {
                if (action.equals(ttBestMove)) continue;

                FastTablutState nextState = state.clone();
                if (!nextState.applyMove(action)) { continue; }

                AlphaBetaResult result = minValue(nextState, alpha, beta, depthRemaining - 1, currentMaxDepth, timeLimit);

                if (result.getScore() > maxScore) {
                    maxScore = result.getScore();
                    bestMove = action;
                }

                alpha = Math.max(alpha, maxScore);

                if (alpha >= beta) {
                    break;
                }
            }
        }

        int nodeType;
        if (maxScore <= oldAlpha) {
            nodeType = UPPER_BOUND;
        } else if (maxScore >= beta) {
            nodeType = LOWER_BOUND;
        } else {
            nodeType = EXACT_SCORE;
        }

        transpositionTable.put(stateKey, new TranspositionEntry(maxScore, depthRemaining, nodeType, bestMove));

        return new AlphaBetaResult(maxScore, bestMove);
    }

    private AlphaBetaResult minValue(FastTablutState state, int alpha, int beta, int depthRemaining,
                                     int currentMaxDepth, long timeLimit) {
        if (System.currentTimeMillis() >= timeLimit) {
            throw new RuntimeException("Timeout reached in MinValue");
        }

        if (state.getTurn().equals(Turn.WHITEWIN)) {
            return new AlphaBetaResult(MAX_VALUE + depthRemaining, null);
        }
        if (state.getTurn().equals(Turn.BLACKWIN)) {
            return new AlphaBetaResult(MIN_VALUE - depthRemaining, null);
        }

        // --------- INIZIO MODIFICA FASE 5 ---------
        if (depthRemaining == 0) {
            // Chiama quiescence con la profondità massima di quiete
            return quiescenceSearch(state, alpha, beta, timeLimit, MAX_QUIESCENCE_DEPTH);
        }
        // --------- FINE MODIFICA FASE 5 ---------


        int oldBeta = beta;
        long stateKey = state.getZobristKey();
        TranspositionEntry entry = transpositionTable.get(stateKey);

        Action ttBestMove = null;

        if (entry != null && entry.getDepth() >= depthRemaining) {
            ttBestMove = entry.getBestMove();

            if (entry.nodeType == EXACT_SCORE) {
                return new AlphaBetaResult(entry.getScore(), ttBestMove);
            } else if (entry.nodeType == LOWER_BOUND) {
                alpha = Math.max(alpha, entry.getScore());
            } else if (entry.nodeType == UPPER_BOUND) {
                beta = Math.min(beta, entry.getScore());
            }
            if (alpha >= beta) {
                return new AlphaBetaResult(entry.getScore(), ttBestMove);
            }
        }

        List<Action> possibleActions = this.getLegalMoves(state);

        if (possibleActions.isEmpty()) {
            return new AlphaBetaResult(MAX_VALUE + depthRemaining, null);
        }

        int minScore = MAX_VALUE;
        Action bestMove = possibleActions.get(0);

        if (ttBestMove != null) {
            FastTablutState nextState = state.clone();
            if (nextState.applyMove(ttBestMove)) {
                AlphaBetaResult result = maxValue(nextState, alpha, beta, depthRemaining - 1, currentMaxDepth, timeLimit);

                if (result.getScore() < minScore) {
                    minScore = result.getScore();
                    bestMove = ttBestMove;
                }
                beta = Math.min(beta, minScore);
            }
        }

        if (beta > alpha) {
            for (Action action : possibleActions) {
                if (action.equals(ttBestMove)) continue;

                FastTablutState nextState = state.clone();
                if (!nextState.applyMove(action)) {
                    continue;
                }

                AlphaBetaResult result = maxValue(nextState, alpha, beta, depthRemaining - 1, currentMaxDepth, timeLimit);

                if (result.getScore() < minScore) {
                    minScore = result.getScore();
                    bestMove = action;
                }

                beta = Math.min(beta, minScore);

                if (minScore <= alpha) {
                    break;
                }
            }
        }

        int nodeType;
        if (minScore <= alpha) {
            nodeType = UPPER_BOUND;
        } else if (minScore >= oldBeta) {
            nodeType = LOWER_BOUND;
        } else {
            nodeType = EXACT_SCORE;
        }

        transpositionTable.put(stateKey, new TranspositionEntry(minScore, depthRemaining, nodeType, bestMove));

        return new AlphaBetaResult(minScore, bestMove);
    }


    // ----------------------------------------------------------------------
    // 5. FUNZIONE DI QUIETE (MODIFICATA FASE 5)
    // ----------------------------------------------------------------------

    /**
     * Ricerca solo le mosse "non tranquille" (catture) per stabilizzare la valutazione.
     * Ora include un limite di profondità.
     */
    private AlphaBetaResult quiescenceSearch(FastTablutState state, int alpha, int beta, long timeLimit, int depth) { // Aggiunto 'depth'
        if (System.currentTimeMillis() >= timeLimit) {
            throw new RuntimeException("Timeout reached in Quiescence");
        }

        if (state.getTurn().equals(Turn.WHITEWIN)) {
            return new AlphaBetaResult(MAX_VALUE);
        }
        if (state.getTurn().equals(Turn.BLACKWIN)) {
            return new AlphaBetaResult(MIN_VALUE);
        }

        // --------- INIZIO MODIFICA FASE 5 ---------
        // Se la profondità di quiete è esaurita, ci fermiamo e valutiamo
        if (depth == 0) {
            return new AlphaBetaResult(evaluateState(state));
        }
        // --------- FINE MODIFICA FASE 5 ---------


        int standPatScore = evaluateState(state);

        if (state.getTurn().equals(Turn.WHITE)) { // MAX (Bianco)
            if (standPatScore >= beta) {
                return new AlphaBetaResult(standPatScore);
            }
            alpha = Math.max(alpha, standPatScore);
        } else { // MIN (Nero)
            if (standPatScore <= alpha) {
                return new AlphaBetaResult(standPatScore);
            }
            beta = Math.min(beta, standPatScore);
        }

        List<Action> captureMoves = getCaptureMoves(state);

        if (captureMoves.isEmpty()) {
            return new AlphaBetaResult(standPatScore); // Posizione tranquilla
        }

        // TODO: Ordinare le mosse di cattura (MVV-LVA)

        for (Action action : captureMoves) {
            FastTablutState nextState = state.clone();
            if (!nextState.applyMove(action)) continue;

            // MODIFICA FASE 5: Passa depth - 1
            AlphaBetaResult result = quiescenceSearch(nextState, alpha, beta, timeLimit, depth - 1);

            if (state.getTurn().equals(Turn.WHITE)) { // MAX
                int score = result.getScore();
                if (score > alpha) {
                    alpha = score;
                }
                if (alpha >= beta) {
                    break;
                }
            } else { // MIN
                int score = result.getScore();
                if (score < beta) {
                    beta = score;
                }
                if (beta <= alpha) {
                    break;
                }
            }
        }

        return new AlphaBetaResult(state.getTurn().equals(Turn.WHITE) ? alpha : beta);
    }

    /**
     * Metodo helper per identificare solo le mosse che risultano in una cattura.
     */
    private List<Action> getCaptureMoves(FastTablutState state) {
        List<Action> allMoves = state.generateLegalMoves();
        int initialPawns = state.whitePawnsCount + state.blackPawnsCount + (state.kingRow != -1 ? 1 : 0);

        return allMoves.stream().filter(action -> {
            FastTablutState tempState = state.clone();
            if (!tempState.applyMove(action)) return false;

            int finalPawns = tempState.whitePawnsCount + tempState.blackPawnsCount + (tempState.kingRow != -1 ? 1 : 0);

            return (finalPawns < initialPawns ||
                    tempState.getTurn().equals(Turn.BLACKWIN) ||
                    tempState.getTurn().equals(Turn.WHITEWIN));
        }).collect(Collectors.toList());
    }


    // ----------------------------------------------------------------------
    // 6. FUNZIONE EURISTICA (MODIFICATA FASE 3)
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

        if (kingR == -1) { return MIN_VALUE; } // Il re è stato catturato

        double kingPositionScore = evalKingPos(state, kingR, kingC);
        double escapeDistancePenalty = this.weights[11] * getMinEscapeDistance(kingR, kingC);
        double materialScore = this.weights[9] * whiteCount + this.weights[10] * blackCount;

        // (Fase 3)
        double kingSafetyScore = evalKingSafety(state, kingR, kingC);

        double totalScore = this.weights[7] * materialScore +
                this.weights[8] * (kingPositionScore + escapeDistancePenalty) +
                kingSafetyScore; // Aggiunto il nuovo punteggio

        int finalScore = (int) Math.round(totalScore);
        finalScore = Math.min(finalScore, HEURISTIC_MAX);
        finalScore = Math.max(finalScore, HEURISTIC_MIN);

        return finalScore;
    }

    /**
     * Valuta le vie di fuga principali del Re.
     */
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
                    score += this.weights[0]; // [0] Via di Fuga Libera
                    break;
                }

                if (currentPawn == FastTablutState.E && isCitadelOrThrone(r, c)) {
                    if (r == FastTablutState.THRONE[0] && c == FastTablutState.THRONE[1]) {
                        score += this.weights[2]; // [2] Penalità per blocco da Trono vuoto
                    } else if (containsCoord(FastTablutState.CITADELS, r, c)) {
                        score += this.weights[1]; // [1D] Penalità per blocco da Cittadella vuota
                    }
                    continue;
                }

                if (currentPawn == FastTablutState.B) {
                    score += isAdjacent ? this.weights[6] : this.weights[5]; // [6] Blocco adiacente Nero, [5] Blocco lontano Nero
                    break;
                }

                if (currentPawn == FastTablutState.W || currentPawn == FastTablutState.K) {
                    score += isAdjacent ? this.weights[4] : this.weights[3]; // [4] Blocco adiacente Bianco, [3] Blocco lontano Bianco
                    break;
                }
            }
        }
        return score;
    }

    /**
     * Valuta la sicurezza immediata del Re controllando le 8 caselle adiacenti. (Fase 3)
     */
    private double evalKingSafety(FastTablutState state, int kingR, int kingC) {
        double score = 0;
        for (int[] dir : ADJACENT_DIRECTIONS) {
            int r = kingR + dir[0];
            int c = kingC + dir[1];

            if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) continue;

            byte pawn = state.get(r, c);
        }
        return score;
    }

    private boolean isCitadelOrThrone(int r, int c) {
        if (r == FastTablutState.THRONE[0] && c == FastTablutState.THRONE[1]) return true;
        return containsCoord(FastTablutState.CITADELS, r, c);
    }
}