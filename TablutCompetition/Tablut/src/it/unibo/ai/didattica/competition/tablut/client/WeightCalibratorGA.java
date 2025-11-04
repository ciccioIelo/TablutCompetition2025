package it.unibo.ai.didattica.competition.tablut.client;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.FastTablutState;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * Classe per l'Algoritmo Genetico (GA) che ottimizza i pesi euristici
 * utilizzando una strategia di Co-Evoluzione (Self-Play) su un pool di avversari.
 * (Fase 3 della Roadmap)
 */
public class WeightCalibratorGA {

    // --- PARAMETRI GA (Modificabili per il Tuning) ---
    private static final int POPULATION_SIZE = 50;
    private static final int NUM_GENERATIONS = 1;
    private static final int NUM_MATCHES_PER_CHROMOSOME = 5; // Simula 5 partite contro avversari casuali dal pool
    private static final double MUTATION_RATE = 0.05;
    private static final double MUTATION_STEP = 100.0;

    // Tempo di simulazione aggressivo per forzare D=2/3
    private static final int MATCH_TIMEOUT_SECONDS = 0; // Usiamo 0 secondi per forzare il timeout dinamico
    private static final long SINGLE_MOVE_TIME_MILLIS = 1000L; // 500 ms per mossa per mirare a D=2

    private final ExecutorService executorService;
    private final Random random = new Random();
    private PrintWriter logWriter;

    // Struttura di memorizzazione per i set di pesi e il loro fitness
    private static class Chromosome {
        final double[] weights;
        double fitnessScore = 0.0;

        public Chromosome(double[] weights) {
            this.weights = weights;
        }

        public double getFitnessScore() { return fitnessScore; }
        public double[] getWeights() { return weights; }

        public void addResult(Turn result) {
            if (result.equals(Turn.WHITEWIN) || result.equals(Turn.BLACKWIN)) {
                fitnessScore += 3.0;
            } else if (result.equals(Turn.DRAW)) {
                fitnessScore += 1.0;
            }
        }
    }

    public WeightCalibratorGA() {
        int numThreads = Runtime.getRuntime().availableProcessors();
        this.executorService = Executors.newFixedThreadPool(numThreads);
        System.out.println("GA Engine initialized with " + numThreads + " threads.");
    }

    // ----------------------------------------------------------------------
    // LOGICA PRINCIPALE DEL GA
    // ----------------------------------------------------------------------

    public double[] runGA() throws InterruptedException, ExecutionException, IOException {

        // Inizializza il writer per il log
        logWriter = new PrintWriter(new FileWriter("ga_evolution_log.txt", false));
        logWriter.println("GENERATION\tBEST_FITNESS\tAVG_FITNESS\tBEST_WEIGHTS");
        logWriter.flush();

        List<Chromosome> population = initializePopulation();
        List<Chromosome> bestChromosomesHistory = new ArrayList<>();
        Chromosome overallBestChromosome = null;

        for (int gen = 0; gen < NUM_GENERATIONS; gen++) {
            System.out.println("\n--- Generation " + gen + " ---");
            long genStartTime = System.currentTimeMillis();

            // 1. Valutazione Fitness (Co-Evoluzione)
            population = evaluatePopulation(population, bestChromosomesHistory);

            // 2. Ordinamento e Selezione (Elitismo)
            population.sort(Comparator.comparingDouble(Chromosome::getFitnessScore).reversed());
            Chromosome currentBest = population.get(0);

            // Aggiorna il miglior cromosoma generale
            if (overallBestChromosome == null || currentBest.getFitnessScore() > overallBestChromosome.getFitnessScore()) {
                overallBestChromosome = new Chromosome(currentBest.getWeights());
                overallBestChromosome.fitnessScore = currentBest.getFitnessScore();
            }

            double avgFitness = population.stream().mapToDouble(Chromosome::getFitnessScore).average().orElse(0.0);

            // Log su console
            System.out.printf("Time: %.2fs | Best Fitness: %.2f | Avg Fitness: %.2f\n",
                    (System.currentTimeMillis() - genStartTime) / 1000.0,
                    currentBest.getFitnessScore(),
                    avgFitness
            );

            // Log su file
            logEvolution(gen, currentBest.getFitnessScore(), avgFitness, currentBest.getWeights());

            // Aggiungi i migliori alla storia (per Co-Evoluzione)
            bestChromosomesHistory.add(currentBest);
            if (bestChromosomesHistory.size() > 20) {
                bestChromosomesHistory.remove(0);
            }

            // Seleziona i genitori (es. 50% migliore)
            List<Chromosome> parents = population.subList(0, POPULATION_SIZE / 2);

            // 3. Creazione Nuova Generazione (Crossover e Mutazione)
            List<Chromosome> newPopulation = new ArrayList<>();
            newPopulation.add(overallBestChromosome); // Mantieni il migliore assoluto (Elitismo)

            while (newPopulation.size() < POPULATION_SIZE) {
                Chromosome parent1 = parents.get(random.nextInt(parents.size()));
                Chromosome parent2 = parents.get(random.nextInt(parents.size()));

                double[] childWeights = crossover(parent1.getWeights(), parent2.getWeights());
                mutate(childWeights);

                newPopulation.add(new Chromosome(childWeights));
            }

            population = newPopulation;
        }

        executorService.shutdownNow();
        logWriter.close();
        return overallBestChromosome.getWeights();
    }

    // ... (omissis: OPERATORI GA: initializePopulation, crossover, mutate)

    private List<Chromosome> initializePopulation() {
        List<Chromosome> population = new ArrayList<>();
        for (int i = 0; i < POPULATION_SIZE; i++) {
            double[] weights = HeuristicWeights.INITIAL_WEIGHTS.clone();
            if (i > 0) mutate(weights);
            population.add(new Chromosome(weights));
        }
        return population;
    }

    private double[] crossover(double[] p1, double[] p2) {
        double[] child = new double[p1.length];
        int crossoverPoint = random.nextInt(p1.length);

        for (int i = 0; i < p1.length; i++) {
            child[i] = (i < crossoverPoint) ? p1[i] : p2[i];
        }
        return child;
    }

    private void mutate(double[] weights) {
        for (int i = 0; i < weights.length; i++) {
            if (random.nextDouble() < MUTATION_RATE) {
                double change = (random.nextDouble() * MUTATION_STEP * 2) - MUTATION_STEP;
                weights[i] += change;
            }
        }
    }

    // ----------------------------------------------------------------------
    // LOGGING
    // ----------------------------------------------------------------------

    private void logEvolution(int generation, double bestFitness, double avgFitness, double[] bestWeights) {
        StringBuilder sb = new StringBuilder();
        sb.append(generation).append("\t")
                .append(String.format(Locale.US, "%.4f", bestFitness)).append("\t")
                .append(String.format(Locale.US, "%.4f", avgFitness)).append("\t");

        sb.append("{");
        for (int i = 0; i < bestWeights.length; i++) {
            sb.append(String.format(Locale.US, "%.2f", bestWeights[i]));
            if (i < bestWeights.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("}");

        logWriter.println(sb.toString());
        logWriter.flush();
    }

    // ----------------------------------------------------------------------
    // FUNZIONE DI FITNESS (Simulazione Partite)
    // ----------------------------------------------------------------------

    private List<Chromosome> evaluatePopulation(List<Chromosome> population, List<Chromosome> opponentsPool)
            throws InterruptedException, ExecutionException {

        List<Future<?>> futures = new ArrayList<>();
        List<Chromosome> opponents = getOpponentsForEvaluation(population, opponentsPool);

        for (Chromosome chromosome : population) {
            chromosome.fitnessScore = 0.0; // Reset Fitness

            Callable<Void> task = () -> {
                for (int i = 0; i < NUM_MATCHES_PER_CHROMOSOME; i++) {
                    Chromosome opponent = opponents.get(random.nextInt(opponents.size()));

                    // Match 1: Cromo vs Opponent
                    Turn result = simulateMatch(chromosome.getWeights(), opponent.getWeights());
                    chromosome.addResult(result);

                    // Match 2: Opponent vs Cromo (Ruoli Invertiti)
                    result = simulateMatch(opponent.getWeights(), chromosome.getWeights());
                    chromosome.addResult(invertResult(result));
                }
                return null;
            };
            futures.add(executorService.submit(task));
        }

        for (Future<?> future : futures) {
            future.get();
        }

        return population;
    }

    private List<Chromosome> getOpponentsForEvaluation(List<Chromosome> currentPopulation, List<Chromosome> history) {
        List<Chromosome> opponents = new ArrayList<>(currentPopulation);
        if (!history.isEmpty()) {
            opponents.addAll(history);
        }
        return opponents;
    }

    /**
     * Simula una singola partita tra due set di pesi, forzando un breve timeout per mossa (D=2/3).
     */
    private Turn simulateMatch(double[] whiteWeights, double[] blackWeights) {
        final int MAX_MOVES = 200;

        // Inizializza i motori con i pesi specifici
        AlphaBetaEngine whiteEngine = new AlphaBetaEngine(Turn.WHITE, whiteWeights);
        AlphaBetaEngine blackEngine = new AlphaBetaEngine(Turn.BLACK, blackWeights);

        FastTablutState currentState = FastTablutState.fromState(new it.unibo.ai.didattica.competition.tablut.domain.StateTablut());
        currentState.setTurn(Turn.WHITE);

        for (int i = 0; i < MAX_MOVES; i++) {
            Action move = null;

            // Tempo limite per la singola mossa
            long moveTimeLimit = System.currentTimeMillis() + SINGLE_MOVE_TIME_MILLIS;

            try {
                if (currentState.getTurn().equals(Turn.WHITE)) {
                    // Passa il tempo limite assoluto (per D=2, circa 500ms)
                    move = whiteEngine.getBestMove(currentState, (int)SINGLE_MOVE_TIME_MILLIS / 1000);
                } else if (currentState.getTurn().equals(Turn.BLACK)) {
                    move = blackEngine.getBestMove(currentState, (int)SINGLE_MOVE_TIME_MILLIS / 1000);
                }
            } catch (Exception e) {
                // Errore/Timeout interno nella ricerca (dovrebbe essere catturato come null)
            }

            if (move == null) {
                return Turn.DRAW;
            }

            // Applica la mossa e verifica le condizioni terminali
            FastTablutState nextState = currentState.clone();
            nextState.applyMove(move);
            currentState = nextState;

            if (currentState.getTurn().equals(Turn.WHITEWIN)) return Turn.WHITEWIN;
            if (currentState.getTurn().equals(Turn.BLACKWIN)) return Turn.BLACKWIN;
            if (currentState.getTurn().equals(Turn.DRAW)) return Turn.DRAW;
        }

        return Turn.DRAW; // Superato MAX_MOVES
    }

    private Turn invertResult(Turn result) {
        if (result.equals(Turn.WHITEWIN)) return Turn.BLACKWIN;
        if (result.equals(Turn.BLACKWIN)) return Turn.WHITEWIN;
        return Turn.DRAW;
    }

    // ----------------------------------------------------------------------
    // MAIN PER ESECUZIONE OFFLINE
    // ----------------------------------------------------------------------

    public static void main(String[] args) {
        System.out.println("--- Starting Genetic Algorithm Tuning ---");
        System.out.println("Settings: Pop=" + POPULATION_SIZE +
                ", Gens=" + NUM_GENERATIONS +
                ", Matches/Cromo=" + (NUM_MATCHES_PER_CHROMOSOME * 2) +
                ", Move Time=" + SINGLE_MOVE_TIME_MILLIS + "ms");
        WeightCalibratorGA ga = new WeightCalibratorGA();
        try {
            double[] optimizedWeights = ga.runGA();
            System.out.println("\n*** OPTIMIZATION COMPLETE ***");
            System.out.println("Final Optimized Weights:");
            System.out.print("{");
            for (int i = 0; i < optimizedWeights.length; i++) {
                System.out.printf(Locale.US, "%.4f", optimizedWeights[i]);
                if (i < optimizedWeights.length - 1) {
                    System.out.print(", ");
                }
            }
            System.out.println("}");
        } catch (Exception e) {
            System.err.println("GA failed due to an exception:");
            e.printStackTrace();
        }
    }
}