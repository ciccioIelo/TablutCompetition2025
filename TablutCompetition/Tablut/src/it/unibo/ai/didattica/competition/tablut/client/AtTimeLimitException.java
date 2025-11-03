package it.unibo.ai.didattica.competition.tablut.client;

/**
 * Eccezione specifica per interrompere la ricorsione Minimax/Alpha-Beta 
 * quando il limite di tempo del turno viene raggiunto. 
 * Basato sulla pratica MinimaxPruneTree.java.
 */
public class AtTimeLimitException extends RuntimeException {
    public AtTimeLimitException() {
        super("Time limit reached");
    }
}