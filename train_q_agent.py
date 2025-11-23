import numpy as np
from connect4.connect_state import ConnectState

# IMPORTA TU AGENTE (GroupB)
from groups.GroupB.policy import QLearningPolicy

# IMPORTA EL HEURÍSTICO DEL GRUPO A (Aha)
from groups.GroupA.policy import Aha


# ----------------------
#   RIVALES DE ENTRENAMIENTO
# ----------------------

class RandomOpponent:
    """
    Rival completamente aleatorio.
    Solo mira qué columnas están libres y escoge una al azar.
    """
    def act(self, board: np.ndarray) -> int:
        available = [c for c in range(7) if board[0, c] == 0]
        return int(np.random.choice(available))


class HeuristicOpponent:
    """
    Wrapper para usar el agente heurístico Aha como rival de entrenamiento.
    """
    def __init__(self) -> None:
        self.agent = Aha()
        self.agent.mount()  # En este caso no hace nada, pero deja claro el flujo.

    def act(self, board: np.ndarray) -> int:
        return self.agent.act(board)


# ----------------------
#   FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ----------------------

def train_q_agent(
    num_episodes: int = 60000,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.9995,
    save_path: str = "q_table.pkl",
) -> None:
    """
    Entrena al agente Q-learning siempre como jugador -1 (rojo),
    jugando partidas completas contra dos tipos de rivales:

    - RandomOpponent (jugador aleatorio)
    - HeuristicOpponent (agente heurístico Aha)

    Se usa una política ε-greedy:
    - al principio explora mucho (epsilon_start ~ 1.0)
    - con el tiempo explora menos (epsilon → epsilon_min)
    """

    # Nuestro agente Q-learning. En evaluación usará epsilon_eval=0.0 (greedy).
    qagent = QLearningPolicy(epsilon_eval=0.0)
    epsilon = epsilon_start

    for ep in range(1, num_episodes + 1):

        # Escogemos el tipo de rival para TODO el episodio.
        # 70% de las veces: random, 30%: heurístico.
        r = np.random.random()
        if r < 0.70:
            opponent = RandomOpponent()
        else:
            opponent = HeuristicOpponent()

        # Estado inicial: tablero vacío, jugador -1 empieza.
        state = ConnectState()
        done = False

        while not state.is_final():
            board = state.board
            player = state.player  # -1 = Q-learning, 1 = rival

            # --- Turno del Q-learning ---
            if player == -1:
                # ε-greedy: a veces explora, a veces explota la Q-table.
                action, state_key = qagent.select_action_for_training(
                    board, player, epsilon
                )
            # --- Turno del rival ---
            else:
                action = opponent.act(board)
                state_key = None  # En turnos del rival no actualizamos Q.

            # Aplicar la jugada en el entorno
            next_state = state.transition(action)
            next_board = next_state.board
            next_player = next_state.player

            # Calcular recompensa solo desde la perspectiva del jugador que acaba de mover
            reward = 0.0
            winner = next_state.get_winner()
            done = next_state.is_final()

            if done:
                if winner == player:
                    reward = 1.0   # el que jugó acaba de ganar
                elif winner == -player:
                    reward = -1.0  # el que jugó acaba de perder
                else:
                    reward = 0.0   # empate

            # Actualizar Q SOLO cuando jugó nuestro agente (-1)
            if player == -1:
                qagent.update_q(
                    state_key,
                    action,
                    reward,
                    next_board,
                    next_player,
                    alpha,
                    gamma,
                    done,
                )

            # Avanzar al siguiente estado
            state = next_state

        # Actualizar epsilon (menos exploración con el tiempo, hasta epsilon_min)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Log de progreso cada 1000 episodios
        if ep % 1000 == 0:
            print(
                f"Episode {ep}/{num_episodes} "
                f"- epsilon={epsilon:.3f} "
                f"- Q states: {len(qagent.q_table)}"
            )

    # Guardar la Q-table entrenada para usarla después en el torneo.
    qagent.save_q_table(save_path)
    print(f"Entrenamiento terminado. Q-table guardada en {save_path}.")


if __name__ == "__main__":
    train_q_agent()
