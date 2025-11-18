"""
Agente A - "Aha" (Heuristic Connect-4 Agent)
-------------------------------------------

Este agente implementa una política basada en heurísticas deterministas para jugar
Connect-4. Su objetivo principal es cumplir con el requisito mínimo del reto: 
ganar consistentemente a un jugador aleatorio.

La estrategia está diseñada para ser:
✓ Simple
✓ Sin entrenamiento
✓ Muy eficiente
✓ Extremadamente fuerte contra agentes aleatorios

Heurísticas principales:
1. Si puede ganar este turno → gana.
2. Si el oponente puede ganar en el siguiente turno → bloquea.
3. Juega priorizando columnas centrales (3, 2, 4, 1, 5, 0, 6), ya que estadísticamente
   generan más oportunidades de conectar 4.
4. Si no aplica ninguna regla, juega la primera columna válida.

Este enfoque, aunque es sencillo, resulta suficiente para lograr una tasa de victoria
cercana al 100% contra un agente que juega aleatoriamente.
"""

import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import override


class Aha(Policy):
    """
    Agente heurístico para Connect-4.

    No utiliza aprendizaje ni modelos probabilísticos;
    en su lugar, toma decisiones analizando posibles jugadas 
    inmediatas del jugador y del oponente.
    """

    @override
    def mount(self) -> None:
        """
        Método de inicialización del agente.

        En agentes entrenables este método se usaría para cargar modelos, pesos,
        entrenar o inicializar estructuras de datos. Como esta política es puramente 
        heurística, no requiere preparación.
        """
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        """
        Decide la acción (columna) a jugar dada la matriz del tablero.

        Parámetros
        ----------
        s : np.ndarray
            Estado del tablero (6x7). 
            -1 representa fichas del jugador A (rojo).
             1 representa fichas del jugador B (amarillo).
             0 espacios vacíos.

        Retorna
        -------
        int
            Índice de la columna donde se realizará la jugada.
        """

        # -----------------------------------------------
        # 1. Identificar columnas disponibles
        # -----------------------------------------------
        available_cols = [c for c in range(7) if s[0, c] == 0]

        # Crear un estado ConnectState a partir del tablero actual
        # Nota: el valor exacto de player no altera la lógica heurística.
        current_state = ConnectState(board=s, player=-1)

        # -----------------------------------------------
        # 2. Intentar ganar inmediatamente
        # -----------------------------------------------
        for col in available_cols:
            simulated = current_state.transition(col)
            if simulated.get_winner() == current_state.player:
                return col  # jugada ganadora

        # -----------------------------------------------
        # 3. Bloquear al oponente si tiene una victoria inmediata
        # -----------------------------------------------
        for col in available_cols:
            simulated = current_state.transition(col)
            opponent_state = ConnectState(simulated.board, -current_state.player)

            # Revisar si el oponente puede ganar en su próximo turno
            opp_available = [c for c in range(7) if simulated.board[0, c] == 0]
            for opp_col in opp_available:
                if opponent_state.transition(opp_col).get_winner() == -current_state.player:
                    return col  # bloquear

        # -----------------------------------------------
        # 4. Priorizar columnas fuertes (centralidad)
        # -----------------------------------------------
        preferred_order = [3, 2, 4, 1, 5, 0, 6]
        for col in preferred_order:
            if col in available_cols:
                return col

        # -----------------------------------------------
        # 5. Última alternativa (nunca debería ocurrir)
        # -----------------------------------------------
        return available_cols[0]
