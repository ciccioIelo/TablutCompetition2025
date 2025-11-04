package it.unibo.ai.didattica.competition.tablut.client;

import java.io.IOException;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.FastTablutState;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;


/**
 * Agente Tablut modulare: gestisce I/O e delega la logica di ricerca (Alpha-Beta)
 * all'AlphaBetaEngine, iniettando i pesi.
 */
public class MyTablutAgent extends TablutClient {

    private final AlphaBetaEngine aiEngine;
    private final int timeoutInSeconds;

    public MyTablutAgent(String player, String name, int timeout) throws IOException {
        super(player, name, timeout);
        this.timeoutInSeconds = timeout;

        // FASE 3.1: INIEZIONE DEI PESI INIZIALI
        double[] initialWeights = HeuristicWeights.INITIAL_WEIGHTS;
        this.aiEngine = new AlphaBetaEngine(this.getPlayer(), initialWeights);

        System.out.println("Agente " + name + " (" + player + ") inizializzato con motore AI modulare.");
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

    @Override
    public void run() {
        try {
            this.declareName();
        } catch (Exception e) {
            e.printStackTrace();
        }

        State currentState = this.getCurrentState();

        System.out.println("You are player " + this.getPlayer().toString() + "!");

        while (true) {
            try {
                this.read();
            } catch (ClassNotFoundException | IOException e1) {
                e1.printStackTrace();
                System.exit(1);
            }

            currentState = this.getCurrentState();
            System.out.println("Current state:");
            System.out.println(currentState.toString());

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

                FastTablutState fastState = FastTablutState.fromState(currentState);
                Action bestAction = aiEngine.getBestMove(fastState, this.timeoutInSeconds - 2);

                if (bestAction != null) {
                    System.out.println("Mossa scelta: " + bestAction.toString());
                    try {
                        this.write(bestAction);
                    } catch (ClassNotFoundException | IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    System.err.println("Impossibile trovare una mossa valida (motore AI bloccato o nessuna mossa legale).");
                }

            } else {
                System.out.println("Waiting for your opponent move... ");
            }
        }
    }
}