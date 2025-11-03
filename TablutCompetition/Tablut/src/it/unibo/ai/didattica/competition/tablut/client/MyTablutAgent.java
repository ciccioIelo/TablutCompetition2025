package it.unibo.ai.didattica.competition.tablut.client;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections; 
import java.util.Comparator;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.GameTablut;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

import it.unibo.ai.didattica.competition.tablut.exceptions.*;


/**
 * Agente Tablut basato sull'algoritmo Alpha-Beta Pruning e Ricerca in Profondita' Iterativa con Ordinamento delle Mosse.
 * VERSIONE CORRETTA: Euristica bilanciata senza KING_BASE_VALUE.
 */
public class MyTablutAgent extends TablutClient {

    private static final int BOARD_SIZE = 9; 
    
    // --- IMPOSTAZIONE LOG ---
    // !!! SETTATO A TRUE PER VEDERE TUTTI I LOG DI DEBUG NELL'ALPHABETA !!!
    private static final boolean ENABLE_DIAGNOSTIC_LOGS = false; 
    
    // --- VALORI ESTREMI E MARGINI ---
    private static final int MAX_VALUE = 100000; 
    private static final int MIN_VALUE = -100000;

    // Margini di sicurezza per l'euristica (NON per gli stati terminali)
    private static final int HEURISTIC_MAX = 50000; 
    private static final int HEURISTIC_MIN = -50000;
    
    private static final int INITIAL_ALPHA = MIN_VALUE - 1000; 
    private static final int INITIAL_BETA = MAX_VALUE + 1000; 
    
    private final GameTablut rules = new GameTablut(500000); 
    private final int timeoutInSeconds;
    
    // --- PESI EURISTICI BILANCIATI ---
    private static final double[] WEIGHTS = {
        // [0] Via di Fuga Libera (MASSIMA PRIORITÀ per il Bianco)
        5000.0,
        
        // [1] Penalità per blocco da Cittadella vuota
        -300.0,
        
        // [2] Penalità per blocco da Trono vuoto
        -500.0,
        
        // [3] Penalità per blocco lontano da pedina bianca (alleato lontano blocca via)
        -50.0,
        
        // [4] Penalità per blocco adiacente da pedina bianca (alleato vicino blocca via)
        -150.0,
        
        // [5] Bonus per blocco lontano da pedina nera (nemico lontano, meno minaccia)
        100.0,
        
        // [6] Penalità per blocco adiacente da pedina nera (FORTE MINACCIA)
        -800.0,
        
        // [7] Peso Bilanciamento Materiale (moltiplicatore generale)
        1.0,
        
        // [8] Peso Posizionale Re (moltiplicatore per kingScore)
        1.0,
        
        // [9] Peso Pedine Bianche (Materiale) - Valore per ogni pedina bianca
        80.0,
        
        // [10] Peso Pedine Nere (Materiale) - Valore per ogni pedina nera (negativo)
        -60.0,
        
        // [11] Distanza Manhattan da casella di fuga (peso negativo = penalizza distanza alta)
        // Più il Re è lontano, peggio è per il Bianco
        -200.0
    };

    // Coordinate del tabellone (per l'euristica)
    private static final List<int[]> CITADELS = List.of(
        new int[]{0,3}, new int[]{0,4}, new int[]{0,5}, new int[]{1,4},
        new int[]{8,3}, new int[]{8,4}, new int[]{8,5}, new int[]{7,4},
        new int[]{3,8}, new int[]{4,8}, new int[]{5,8}, new int[]{4,7},
        new int[]{3,0}, new int[]{4,0}, new int[]{5,0}, new int[]{4,1}
    );
    private static final List<int[]> ESCAPES = List.of(
        new int[]{0,1}, new int[]{0,2}, new int[]{0,6}, new int[]{0,7},
        new int[]{8,1}, new int[]{8,2}, new int[]{8,6}, new int[]{8,7},
        new int[]{1,0}, new int[]{2,0}, new int[]{6,0}, new int[]{7,0},
        new int[]{1,8}, new int[]{2,8}, new int[]{6,8}, new int[]{7,8}
    );
    private static final int[] THRONE = {4, 4};
    
    // --- TRANSPOSITION TABLE ---
    private Map<String, TranspositionEntry> transpositionTable;
    
    private static final int EXACT_SCORE = 0;
    private static final int LOWER_BOUND = 1; 
    private static final int UPPER_BOUND = 2; 


    private class TranspositionEntry {
        final int score;
        final int depth;
        final int nodeType; 

        public TranspositionEntry(int score, int depth, int nodeType) {
            this.score = score;
            this.depth = depth;
            this.nodeType = nodeType;
        }
        public int getScore() { return score; }
        public int getDepth() { return depth; }
        public int getNodeType() { return nodeType; }
    }


    public class AlphaBetaResult {
        private final int score;
        private final Action action;

        public AlphaBetaResult(int score, Action action) {
            this.score = score;
            this.action = action;
        }

        public AlphaBetaResult(int score) {
            this(score, null);
        }

        public int getScore() { 
            return score; 
        }
        
        public Action getAction() { 
            return action; 
        }
    }

    private class ActionScore {
        final Action action;
        final int score;
        public ActionScore(Action action, int score) {
            this.action = action;
            this.score = score;
        }
        public Action getAction() { return action; }
    }

    private class ActionScoreComparator implements Comparator<ActionScore> {
        private final Turn playerToMove;

        public ActionScoreComparator(Turn playerToMove) {
            this.playerToMove = playerToMove;
        }

        @Override
        public int compare(ActionScore a, ActionScore b) {
            if (playerToMove.equals(Turn.WHITE)) {
                return Integer.compare(b.score, a.score); 
            }
            else {
                return Integer.compare(a.score, b.score);
            }
        }
    }

    public MyTablutAgent(String player, String name, int timeout) throws IOException {
        super(player, name, timeout);
        this.timeoutInSeconds = timeout;
        this.transpositionTable = new HashMap<>();
        System.out.println("Agente " + name + " (" + player + ") inizializzato.");
    }
    
    private boolean containsCoord(List<int[]> list, int r, int c) {
        for (int[] coord : list) {
            if (coord[0] == r && coord[1] == c) return true;
        }
        return false;
    }
    
    // ----------------------------------------------------------------------
    // 1. GENERAZIONE E FILTRAGGIO DELLE MOSSE LEGALI 
    // ----------------------------------------------------------------------
    
    private List<Action> getLegalMoves(StateTablut state) {
        List<Action> legalMoves = new ArrayList<>();
        List<Action> candidateMoves = generateAllPossibleMoves(state);
        
        boolean isBlackTurn = state.getTurn().equals(Turn.BLACK);
        
        for (Action candidate : candidateMoves) {
            StateTablut tempState = (StateTablut) state.clone(); 
            
            try {
                rules.checkMove(tempState, candidate);
                legalMoves.add(candidate);
            } catch (Exception e) {
                // Mossa non legale, viene scartata
            }
        }
        
        return legalMoves;
    }
    
    private boolean isCitadelOrThrone(int r, int c) {
        if (r == 4 && c == 4) return true; 
        
        if ((r == 0 && (c == 3 || c == 4 || c == 5)) ||
            (r == 8 && (c == 3 || c == 4 || c == 5)) ||
            (c == 0 && (r == 3 || r == 4 || r == 5)) ||
            (c == 8 && (r == 3 || r == 4 || r == 5)) ||
            (r == 1 && c == 4) || (r == 7 && c == 4) ||
            (c == 1 && r == 4) || (c == 7 && r == 4)) {
            return true;
        }
        
        return false;
    }

    private List<Action> generateAllPossibleMoves(StateTablut state) {
        List<Action> candidates = new ArrayList<>();
        Turn currentTurn = state.getTurn();

        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                Pawn pawn = state.getPawn(r, c);

                boolean isMyPawn = (currentTurn.equals(Turn.WHITE) && (pawn.equals(Pawn.WHITE) || pawn.equals(Pawn.KING))) ||
                                   (currentTurn.equals(Turn.BLACK) && pawn.equals(Pawn.BLACK));

                if (isMyPawn) {
                    
                    int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
                    
                    for (int[] dir : directions) {
                        int dr = dir[0];
                        int dc = dir[1];

                        for (int steps = 1; steps < BOARD_SIZE; steps++) {
                            int nextR = r + dr * steps;
                            int nextC = c + dc * steps;

                            if (nextR >= 0 && nextR < BOARD_SIZE && nextC >= 0 && nextC < BOARD_SIZE) {
                                
                                Pawn destinationPawn = state.getPawn(nextR, nextC);
                                
                                if (!destinationPawn.equals(Pawn.EMPTY)) {
                                    break; 
                                }

                                if (isCitadelOrThrone(nextR, nextC)) {
                                    boolean isThrone = (nextR == 4 && nextC == 4);
                                    
                                    if (pawn.equals(Pawn.WHITE) || pawn.equals(Pawn.BLACK) || (pawn.equals(Pawn.KING) && isThrone)) {
                                        break; 
                                    }
                                }
                                
                                try {
                                    String from = state.getBox(r, c); 
                                    String to = state.getBox(nextR, nextC);
                                    Action candidate = new Action(from, to, currentTurn);
                                    candidates.add(candidate);
                                } catch (Exception e) {
                                }
                                
                            } else {
                                break; 
                            }
                        }
                    }
                }
            }
        }
        return candidates;
    }
    
    private List<Action> sortMovesByHeuristic(StateTablut currentState, List<Action> moves) {
        List<ActionScore> scoredMoves = new ArrayList<>();
        
        for (Action action : moves) {
            StateTablut nextState;
            try {
                StateTablut stateAfterMove = (StateTablut) currentState.clone();
                rules.checkMove(stateAfterMove, action);
                nextState = stateAfterMove; 
                
                int score = evaluateState(nextState);
                scoredMoves.add(new ActionScore(action, score));
            } catch (Exception e) {
                continue;
            }
        }

        Collections.sort(scoredMoves, new ActionScoreComparator(currentState.getTurn())); 
        
        List<Action> orderedActions = new ArrayList<>();
        for (ActionScore as : scoredMoves) {
            orderedActions.add(as.getAction());
        }
        
        return orderedActions;
    }
    
    private List<Action> reorderMoves(List<Action> allMoves, Action preferredMove) {
        List<Action> ordered = new ArrayList<>();
        if (preferredMove != null) {
            boolean added = false;
            for(Action a : allMoves) {
                if (a.equals(preferredMove)) {
                    ordered.add(a);
                    added = true;
                    break; 
                }
            }
            
            for (Action a : allMoves) {
                if (!a.equals(preferredMove)) {
                    ordered.add(a);
                }
            }
            
            if(!added) { 
                 return allMoves;
            }

        } else {
             return allMoves;
        }
        return ordered;
    }

    // ----------------------------------------------------------------------
    // 2. LOGICA MINIMAX (ITERATIVE DEEPENING)
    // ----------------------------------------------------------------------
    
    public Action getBestMove(StateTablut currentState, int timeoutSeconds) {
        
        long startTime = System.currentTimeMillis();
        long timeLimit = startTime + (timeoutSeconds * 1000L) - 100L; 
        
        this.transpositionTable.clear();

        List<Action> legalMoves = this.getLegalMoves(currentState); 
        
        if (legalMoves.isEmpty()) {
            System.err.println("AGENTE BLOCCATO: Nessuna mossa legale disponibile. Ritorno null.");
            return null;
        }

        Action fallbackMove = legalMoves.get(0); 
        Action bestMoveAtCurrentDepth = fallbackMove; 
        
        int bestScoreAtCurrentDepth = evaluateState(currentState); 
        
        int currentDepth = 1;
        
        int consecutiveZeroScores = 0;
        final int STALL_LIMIT = 5; 

        System.out.println("ID: Avvio ricerca Alpha-Beta con limite di tempo: " + timeoutSeconds + "s.");
        System.out.println("DEBUG: Punteggio di fallback iniziale: " + bestScoreAtCurrentDepth);

        while (true) {
            
            System.out.println("ID: Avvio ricerca a profondità D=" + currentDepth); 
            
            int alpha = INITIAL_ALPHA;
            int beta = INITIAL_BETA;
            
            int currentIterationBestScore = (this.getPlayer().equals(Turn.WHITE)) ? MIN_VALUE : MAX_VALUE;
            Action currentIterationBestMove = bestMoveAtCurrentDepth; 
            
            List<Action> orderedMoves;
            
            // LOGICA DI ORDINAMENTO DELLE MOSSE E CALCOLO DEI PUNTEGGI INIZIALI (D=0)
            if (currentDepth > 1 && bestMoveAtCurrentDepth != null) {
                 orderedMoves = reorderMoves(legalMoves, bestMoveAtCurrentDepth);
                 
                 if (ENABLE_DIAGNOSTIC_LOGS) {
                    System.out.println("ID: Riapplicazione dell'ordinamento con mossa preferita: " + bestMoveAtCurrentDepth.toString());
                 }
                 
            } else {
                 List<ActionScore> initialScores = new ArrayList<>();
                 
                 // Calcolo e log dei punteggi euristici (D=0) per la prima iterazione
                 for (Action action : legalMoves) {
                     StateTablut nextState;
                     try {
                         StateTablut stateAfterMove = (StateTablut) currentState.clone();
                         rules.checkMove(stateAfterMove, action);
                         nextState = stateAfterMove; 
                         
                         int score = evaluateState(nextState);
                         initialScores.add(new ActionScore(action, score));
                     } catch (Exception e) {
                         continue;
                     }
                 }
                 
                 Collections.sort(initialScores, new ActionScoreComparator(currentState.getTurn())); 
                 
                 if (ENABLE_DIAGNOSTIC_LOGS) {
                     System.out.println("ID: Punteggi iniziali (D=0) per tutte le mosse:");
                     for (ActionScore as : initialScores) {
                         System.out.println("    [D=0] Mossa: " + as.getAction().toString() + " | Punteggio: " + as.score);
                     }
                 }
                 // Ricrea la lista ordinata
                 orderedMoves = new ArrayList<>();
                 for (ActionScore as : initialScores) {
                     orderedMoves.add(as.getAction());
                 }
                 
                 if (!orderedMoves.isEmpty()) {
                     currentIterationBestMove = orderedMoves.get(0); 
                 }
            }
            if (orderedMoves.isEmpty()) { orderedMoves = legalMoves; }

            try {
                for (Action action : orderedMoves) {
                    
                    if (System.currentTimeMillis() >= timeLimit) {
                        System.out.println("ID: TIMEOUT alla radice prima di analizzare la prossima mossa. Profondità D=" + currentDepth + " interrotta.");
                        throw new RuntimeException("Timeout reached in Minimax"); 
                    }
                    
                    StateTablut nextState;
                    try {
                        StateTablut stateAfterMove = (StateTablut) currentState.clone();
                        nextState = (StateTablut) rules.checkMove(stateAfterMove, action); 
                    } catch (Exception e) {
                        continue; 
                    }

                    int currentScore;
                    int depth = currentDepth;
                    
                    if (this.getPlayer().equals(Turn.WHITE)) {
                        AlphaBetaResult result = minValue(nextState, alpha, beta, depth, currentDepth, timeLimit);
                        currentScore = result.getScore();
                        
                        if (currentScore > currentIterationBestScore) {
                            currentIterationBestScore = currentScore;
                            currentIterationBestMove = action;
                        }
                        alpha = Math.max(alpha, currentIterationBestScore); 

                    } else {
                        AlphaBetaResult result = maxValue(nextState, alpha, beta, depth, currentDepth, timeLimit);
                        currentScore = result.getScore();
                        
                        if (currentScore < currentIterationBestScore) {
                            currentIterationBestScore = currentScore;
                            currentIterationBestMove = action;
                        }
                        beta = Math.min(beta, currentIterationBestScore); 
                    }
                    
                    if (alpha >= beta) {
                        if (ENABLE_DIAGNOSTIC_LOGS) {
                             System.out.println("ID: !!! TAGLIO RADICE !!! Profondità D=" + currentDepth + " Alpha >= Beta: " + alpha + " >= " + beta);
                        }
                        break;
                    }
                    
                    // !!! NUOVO LOG AGGIUNTO QUI !!!
                    if (ENABLE_DIAGNOSTIC_LOGS) {
                         System.out.println("ID: [D=" + currentDepth + " Radice] Analisi di " + action.toString() + " completata. Punteggio: " + currentScore + " | Nuovo Alpha: " + alpha + " Beta: " + beta);
                    }
                }
                
                bestMoveAtCurrentDepth = currentIterationBestMove;
                bestScoreAtCurrentDepth = currentIterationBestScore;

                // Rilevamento stallo rimosso (score non sarà più sempre vicino a un valore fisso)
                
                System.out.println("ID: Profondità D=" + currentDepth + " COMPLETATA. Mossa: " + bestMoveAtCurrentDepth.toString() + " | Score: " + bestScoreAtCurrentDepth);
                currentDepth++;
                
            } catch (RuntimeException e) {
                if (e.getMessage().contains("Timeout reached")) {
                    System.out.println("ID: Raggiunto il limite di tempo. Interrompo Iterative Deepening.");
                    break; 
                } else {
                    throw e; 
                }
            }
        } 
        
        long totalTime = System.currentTimeMillis() - startTime;
        System.out.println("--- LOG FINALE (Iterative Deepening) ---");
        System.out.println("INFO: Tempo totale trascorso: " + (totalTime / 1000.0) + "s.");
        System.out.println("INFO: Massima profondità completata: D=" + (currentDepth - 1) + "."); 
        System.out.println("INFO: Mossa scelta (dalla massima profondità completata): " + bestMoveAtCurrentDepth.toString());
        System.out.println("INFO: Punteggio della mossa: " + bestScoreAtCurrentDepth); 
        System.out.println("------------------------");
        
        return bestMoveAtCurrentDepth;
    }
    
    // ----------------------------------------------------------------------
    // 3. MAX VALUE (CON TRANSPOSITION TABLE E LOG)
    // ----------------------------------------------------------------------

    private AlphaBetaResult maxValue(StateTablut state, int alpha, int beta, int depthRemaining, int currentMaxDepth, long timeLimit) {
        
        if (System.currentTimeMillis() >= timeLimit) {
            throw new RuntimeException("Timeout reached in Minimax"); 
        }

        if (state.getTurn().equals(Turn.WHITEWIN)) { return new AlphaBetaResult(MAX_VALUE + depthRemaining, null); } 
        if (state.getTurn().equals(Turn.BLACKWIN)) { return new AlphaBetaResult(MIN_VALUE - depthRemaining, null); } 
        //if (state.getTurn().equals(Turn.DRAW)) { return new AlphaBetaResult(0, null); } 
        
        if (depthRemaining == 0) { 
            return new AlphaBetaResult(evaluateState(state), null);
        }
        
        String stateHash = state.toString();
        TranspositionEntry entry = transpositionTable.get(stateHash);
        
        if (entry != null && entry.getDepth() >= depthRemaining) {
            if (entry.getNodeType() == EXACT_SCORE) {
                return new AlphaBetaResult(entry.getScore(), null);
            } else if (entry.getNodeType() == LOWER_BOUND) {
                alpha = Math.max(alpha, entry.getScore());
            } else if (entry.getNodeType() == UPPER_BOUND) {
                beta = Math.min(beta, entry.getScore());
            }
            if (alpha >= beta) {
                return new AlphaBetaResult(entry.getScore(), null);
            }
        }
        
        int maxScore = MIN_VALUE;
        int originalAlpha = alpha; 

        List<Action> possibleActions = this.getLegalMoves(state);
        
        if (possibleActions.isEmpty()) {
             return new AlphaBetaResult(evaluateState(state), null);
        }
        
        List<Action> orderedActions = sortMovesByHeuristic(state, possibleActions);
        
        for (Action action : orderedActions) {
            
            if (ENABLE_DIAGNOSTIC_LOGS) {
                // Stampa l'esplorazione del nodo
                String pad = "  ".repeat(currentMaxDepth - depthRemaining);
                System.out.println(pad + "[MAX D=" + depthRemaining + " | A=" + alpha + " B=" + beta + "] Esploro: " + action.toString());
            }
            
            StateTablut nextState;
            try {
                StateTablut stateAfterMove = (StateTablut) state.clone();
                nextState = (StateTablut) rules.checkMove(stateAfterMove, action); 
            } catch (Exception e) {
                continue; 
            }
            
            AlphaBetaResult result = minValue(nextState, alpha, beta, depthRemaining - 1, currentMaxDepth, timeLimit);
            int score = result.getScore(); 

            maxScore = Math.max(maxScore, score);
            
            if (maxScore >= beta) {
                if (ENABLE_DIAGNOSTIC_LOGS) {
                    String pad = "  ".repeat(currentMaxDepth - depthRemaining);
                    // Log del taglio
                    System.out.println(pad + "!!! TAGLIO BETA !!! [D=" + depthRemaining + "] Mossa: " + action.toString() + " con score " + maxScore + " >= Beta " + beta);
                }
                transpositionTable.put(stateHash, new TranspositionEntry(maxScore, depthRemaining, LOWER_BOUND));
                return new AlphaBetaResult(maxScore, action); 
            }
            alpha = Math.max(alpha, maxScore);
        }

        int nodeType = (maxScore > originalAlpha) ? EXACT_SCORE : UPPER_BOUND;
        transpositionTable.put(stateHash, new TranspositionEntry(maxScore, depthRemaining, nodeType));
        
        return new AlphaBetaResult(maxScore, null); 
    }

    // ----------------------------------------------------------------------
    // 4. MIN VALUE (CON TRANSPOSITION TABLE E LOG)
    // ----------------------------------------------------------------------
    
    private AlphaBetaResult minValue(StateTablut state, int alpha, int beta, int depthRemaining, int currentMaxDepth, long timeLimit) {
        
    	if (ENABLE_DIAGNOSTIC_LOGS) {
            String pad = "  ".repeat(currentMaxDepth - depthRemaining);
            System.out.println(pad + "<-- INGRESSO MIN (Nero) D=" + depthRemaining + " -->");
        }
    	
    	if (System.currentTimeMillis() >= timeLimit) {
            throw new RuntimeException("Timeout reached in Minimax"); 
        }

        if (state.getTurn().equals(Turn.WHITEWIN)) { return new AlphaBetaResult(MAX_VALUE + depthRemaining, null); } 
        if (state.getTurn().equals(Turn.BLACKWIN)) { return new AlphaBetaResult(MIN_VALUE - depthRemaining, null); } 
        //if (state.getTurn().equals(Turn.DRAW)) { return new AlphaBetaResult(0, null); } 
        
        if (depthRemaining == 0) { 
            return new AlphaBetaResult(evaluateState(state), null);
        }
        
        String stateHash = state.toString();
        TranspositionEntry entry = transpositionTable.get(stateHash);
        
        if (entry != null && entry.getDepth() >= depthRemaining) {
            if (entry.getNodeType() == EXACT_SCORE) {
                return new AlphaBetaResult(entry.getScore(), null);
            } else if (entry.getNodeType() == LOWER_BOUND) {
                alpha = Math.max(alpha, entry.getScore());
            } else if (entry.getNodeType() == UPPER_BOUND) {
                beta = Math.min(beta, entry.getScore());
            }
            if (alpha >= beta) {
                return new AlphaBetaResult(entry.getScore(), null);
            }
        }
        
        int minScore = MAX_VALUE;
        int originalBeta = beta; 

        List<Action> possibleActions = this.getLegalMoves(state);
        
        if (possibleActions.isEmpty()) {
             return new AlphaBetaResult(evaluateState(state), null);
        }
        
        List<Action> orderedActions = sortMovesByHeuristic(state, possibleActions);
        for (Action action : orderedActions) {
            
            if (ENABLE_DIAGNOSTIC_LOGS) {
                // Stampa l'esplorazione del nodo
                String pad = "  ".repeat(currentMaxDepth - depthRemaining);
                System.out.println(pad + "[MIN D=" + depthRemaining + " | A=" + alpha + " B=" + beta + "] Esploro: " + action.toString());
            }
            
            StateTablut nextState;
            try {
                StateTablut stateAfterMove = (StateTablut) state.clone();
                nextState = (StateTablut) rules.checkMove(stateAfterMove, action); 
            } catch (Exception e) {
                continue; 
            }
            
            AlphaBetaResult result = maxValue(nextState, alpha, beta, depthRemaining - 1, currentMaxDepth, timeLimit);
            int score = result.getScore(); 

            minScore = Math.min(minScore, score);
            
            if (minScore <= alpha) {
                if (ENABLE_DIAGNOSTIC_LOGS) {
                    String pad = "  ".repeat(currentMaxDepth - depthRemaining);
                    // Log del taglio
                    System.out.println(pad + "!!! TAGLIO ALPHA !!! [D=" + depthRemaining + "] Mossa: " + action.toString() + " con score " + minScore + " <= Alpha " + alpha);
                }
                transpositionTable.put(stateHash, new TranspositionEntry(minScore, depthRemaining, UPPER_BOUND));
                return new AlphaBetaResult(minScore, action); 
            }
            beta = Math.min(beta, minScore);
        }

        int nodeType = (minScore < originalBeta) ? EXACT_SCORE : LOWER_BOUND;
        transpositionTable.put(stateHash, new TranspositionEntry(minScore, depthRemaining, nodeType));
        
        return new AlphaBetaResult(minScore, null);
    }
    
    // ----------------------------------------------------------------------
    // 5. FUNZIONE EURISTICA (KING-CENTRIC CON TIE-BREAKER SULLA FUGA)
    // ----------------------------------------------------------------------
    
    /**
     * Calcola la minima distanza di Manhattan dal Re a un qualsiasi quadrante di fuga.
     */
    private int getMinEscapeDistance(int kingR, int kingC) {
        int minDistance = Integer.MAX_VALUE;
        for (int[] escape : ESCAPES) {
            int distance = Math.abs(kingR - escape[0]) + Math.abs(kingC - escape[1]);
            minDistance = Math.min(minDistance, distance);
        }
        return minDistance;
    }
    
    private int evaluateState(StateTablut state) {
        // Stati terminali - valori assoluti molto alti
        if (state.getTurn().equals(Turn.WHITEWIN)) { 
            return MAX_VALUE; 
        } 
        if (state.getTurn().equals(Turn.BLACKWIN)) { 
            return MIN_VALUE; 
        } 
        //if (state.getTurn().equals(Turn.DRAW)) { 
        //    return 0; // Pareggio = neutro
        //}
        
        int whiteCount = 0; 
        int blackCount = 0; 
        int kingR = -1, kingC = -1;
        
        // Conta i pezzi e trova il Re
        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                Pawn piece = state.getPawn(r, c);
                if (piece.equals(Pawn.WHITE)) {
                    whiteCount++;
                } else if (piece.equals(Pawn.BLACK)) {
                    blackCount++;
                } else if (piece.equals(Pawn.KING)) {
                    kingR = r;
                    kingC = c;
                }
            }
        }
        
        // Se il Re non c'è (non dovrebbe mai accadere in uno stato non terminale)
        if (kingR == -1) {
            return MIN_VALUE;
        }
        
        // 1. Valutazione posizionale del Re (vie di fuga, minacce, protezione)
        double kingPositionScore = evalKingPos(state, kingR, kingC);
        
        // 2. Tie-Breaker: Distanza minima da una casella di fuga
        //    WEIGHTS[11] è negativo (-200), quindi più il Re è lontano, 
        //    più il punteggio diminuisce (sfavorevole per il Bianco)
        int minEscapeDistance = getMinEscapeDistance(kingR, kingC);
        double escapeDistancePenalty = WEIGHTS[11] * minEscapeDistance;
        
        // 3. Componente materiale (numero di pedine)
        double materialScore = WEIGHTS[9] * whiteCount + WEIGHTS[10] * blackCount;
        
        // 4. Calcolo finale
        //    - WEIGHTS[7] (=1.0) è il moltiplicatore per il materiale
        //    - WEIGHTS[8] (=1.0) è il moltiplicatore per la posizione del Re
        double totalScore = WEIGHTS[7] * materialScore + 
                            WEIGHTS[8] * (kingPositionScore + escapeDistancePenalty);
        
        // Conversione a int e clamp
        int finalScore = (int) Math.round(totalScore);
        finalScore = Math.min(finalScore, HEURISTIC_MAX);
        finalScore = Math.max(finalScore, HEURISTIC_MIN);
        
        return finalScore;
    }

    private double evalKingPos(StateTablut state, int kingR, int kingC) {
        double score = 0;
        
        // Direzioni: sinistra, destra, su, giù
        int[][] directions = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
        
        for (int[] dir : directions) {
            int dr = dir[0];
            int dc = dir[1];
            
            // Esplora in ogni direzione fino a incontrare un ostacolo
            for (int steps = 1; steps < BOARD_SIZE; steps++) {
                int r = kingR + dr * steps;
                int c = kingC + dc * steps;

                // Fuori dal tabellone
                if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) {
                    break; 
                }
                
                Pawn currentPawn = state.getPawn(r, c);
                boolean isAdjacent = (steps == 1);

                // CASO 1: Via di fuga libera (MASSIMA PRIORITÀ)
                if (currentPawn.equals(Pawn.EMPTY) && containsCoord(ESCAPES, r, c)) {
                    score += WEIGHTS[0]; // +5000
                    break; // Trovata via di fuga, non continuare in questa direzione
                }
                
                // CASO 2: Trono o Cittadella vuoti (bloccano parzialmente ma sono attraversabili)
                if (currentPawn.equals(Pawn.EMPTY) && isCitadelOrThrone(r, c)) { 
                    if (r == THRONE[0] && c == THRONE[1]) {
                        score += WEIGHTS[2]; // -500 (Trono vuoto)
                    } else {
                        score += WEIGHTS[1]; // -300 (Cittadella vuota)
                    }
                    // NON fare break, il Re potrebbe attraversarle
                    continue;
                }

                // CASO 3: Pedina Nera (MINACCIA)
                if (currentPawn.equals(Pawn.BLACK)) {
                    if (isAdjacent) {
                        score += WEIGHTS[6]; // -800 (minaccia adiacente GRAVE)
                    } else {
                        score += WEIGHTS[5]; // +100 (minaccia lontana, meno pericolosa)
                    }
                    break; // Pedina nera blocca la direzione
                }
                
                // CASO 4: Pedina Bianca (PROTEZIONE o BLOCCO)
                if (currentPawn.equals(Pawn.WHITE) || currentPawn.equals(Pawn.KING)) {
                    if (isAdjacent) {
                        score += WEIGHTS[4]; // -150 (alleato vicino blocca via)
                    } else {
                        score += WEIGHTS[3]; // -50 (alleato lontano blocca via)
                    }
                    break; // Pedina bianca blocca la direzione
                }
                
                // CASO 5: Casella vuota normale
                // Non fa nulla, continua a esplorare
            }
        }
        return score;
    }
    
    // ----------------------------------------------------------------------
    // 6. METODO RUN() E MAIN 
    // ----------------------------------------------------------------------
    
    @Override
    public void run() {
        try {
            this.declareName();
        } catch (Exception e) {
            e.printStackTrace();
        }

        State state = this.getCurrentState(); 
        
        System.out.println("You are player " + this.getPlayer().toString() + "!");

        while (true) {
            try {
                this.read();
            } catch (ClassNotFoundException | IOException e1) {
                e1.printStackTrace();
                System.exit(1);
            }
            
            state = this.getCurrentState();
            System.out.println("Current state:"); 
            System.out.println(state.toString());
            try {
                Thread.sleep(100); 
            } catch (InterruptedException e) {
            }

            StateTablut currentState = (StateTablut) state;

            if (currentState.getTurn().equals(StateTablut.Turn.WHITEWIN) || 
                currentState.getTurn().equals(StateTablut.Turn.BLACKWIN) ||
                currentState.getTurn().equals(StateTablut.Turn.DRAW)) {
                
                String outcome = "";
                if (currentState.getTurn().equals(StateTablut.Turn.WHITEWIN)) {
                    outcome = this.getPlayer().equals(Turn.WHITE) ? "YOU WIN (WHITE)!" : "YOU LOSE (BLACK)!";
                } else if (currentState.getTurn().equals(StateTablut.Turn.BLACKWIN)) {
                    outcome = this.getPlayer().equals(Turn.BLACK) ? "YOU WIN (BLACK)!" : "YOU LOSE (WHITE)!";
                } else {
                    outcome = "DRAW!";
                }
                System.out.println(outcome);
                System.exit(0);
            }
            
            if (this.getPlayer().equals(currentState.getTurn())) {
                
                // Tempo limite ridotto a 57 secondi per i log, 
                // così da avere almeno 2 secondi di margine.
                Action bestAction = getBestMove(currentState, this.timeoutInSeconds - 2); 
                
                if (bestAction != null) {
                    System.out.println("Mossa scelta: " + bestAction.toString());
                    try {
                        this.write(bestAction);
                    } catch (ClassNotFoundException | IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    System.err.println("Impossibile trovare una mossa valida (agente bloccato o errore).");
                }

            } else { 
                System.out.println("Waiting for your opponent move... ");
            }
        }

    }
    
    public static void main(String[] args) throws IOException {
        String role = args[0]; 
        String name = "MyTablutAgent"; 
        int timeout = 59; 
        
        if (args.length >= 2) {
            try {
                timeout = Integer.parseInt(args[1]);
            } catch (NumberFormatException e) {
                System.err.println("Timeout non valido, uso il default: 59");
                timeout = 59; 
            }
        }
        
        String ip = "localhost";
        if (args.length >= 3) {
            ip = args[2];
        }
        
        System.out.println("Inizializzazione di " + name + " come " + role + " con timeout " + timeout + "s @ " + ip);
        
        TablutClient client = new MyTablutAgent(role, name, timeout);
        client.run();
    }
}