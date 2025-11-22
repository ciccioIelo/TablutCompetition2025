package it.unibo.ai.didattica.competition.tablut.client;

/**
 * Contenitore statico per i pesi euristici.
 * Questo array rappresenta la struttura del "Cromosoma" per l'Algoritmo Genetico (GA).
 */
public class HeuristicWeightsCopy {

    // Struttura dei pesi:
    // [0] Via di Fuga Libera
    // [1] Penalità per blocco da Cittadella vuota
    // [2] Penalità per blocco da Trono vuoto
    // [3] Penalità per blocco lontano da pedina bianca
    // [4] Penalità per blocco adiacente da pedina bianca
    // [5] Bonus per blocco lontano da pedina nera
    // [6] Penalità per blocco adiacente da pedina nera (FORTE MINACCIA)
    // [7] Peso Bilanciamento Materiale (moltiplicatore materiale)
    // [8] Peso Posizionale Re (moltiplicatore posizionale)
    // [9] Peso Pedine Bianche (Materiale)
    // [10] Peso Pedine Nere (Materiale)
    // [11] Distanza Manhattan da casella di fuga

    public static final double[] INITIAL_WEIGHTS = {
            5000.0, -300.0, -500.0, -50.0, -150.0, 100.0, -800.0, 1.0, 1.0, 80.0, -60.0, -200.0
    };

    private final double[] weights;

    public HeuristicWeightsCopy(double[] weights) {
        // Clono l'array per evitare modifiche esterne involontarie
        this.weights = weights.clone();
    }

    public double[] getWeights() {
        return this.weights;
    }
}