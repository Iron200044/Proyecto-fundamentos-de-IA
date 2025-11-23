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
    """Rival completamente aleatorio."""
    def act(self, board: np.ndarray) -> int:
        available = [c for c in range(7) if board[0, c] == 0]
        return int(np.random.choice(available))


class HeuristicOpponent:
    """Wrapper del agente heurístico Aha."""
    def __init__(self) -> None:
        self.agent = Aha()
        self.agent.mount()

    def act(self, board: np.ndarray) -> int:
        return self.agent.act(board)


# ----------------------
#   FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ----------------------

def train_q_agent(
    num_episodes: int = 13000,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.9995,
    save_path: str = "q_table.pkl",
) -> None:
    """
    Entrena al agente Q-learning jugando tanto como rojo (-1) como amarillo (+1),
    contra dos tipos de rivales:
      - RandomOpponent (70% de los episodios)
      - HeuristicOpponent (30% de los episodios)

    Usa una política ε-greedy en entrenamiento:
    - al principio explora mucho (epsilon_start ~ 1.0)
    - con el tiempo explora menos (epsilon → epsilon_min)
    """

    qagent = QLearningPolicy(epsilon_eval=0.0)
    epsilon = epsilon_start

    for ep in range(1, num_episodes + 1):

        # Rival del episodio
        r = np.random.random()
        if r < 0.70:
            opponent = RandomOpponent()
        else:
            opponent = HeuristicOpponent()

        # 50%: Q-agent juega como amarillo (+1), 50% como rojo (-1)
        if np.random.random() < 0.5:
            state = ConnectState(board=None, player=1)
            qagent_plays_as = 1
        else:
            state = ConnectState(board=None, player=-1)
            qagent_plays_as = -1

        done = False

        while not state.is_final():

            board = state.board
            player = state.player  # jugador que va a mover ahora

            # === Turno del Q-learning ===
            if player == qagent_plays_as:
                action, state_key = qagent.select_action_for_training(
                    board, player, epsilon
                )
            else:
                # === Turno del rival ===
                action = opponent.act(board)
                state_key = None  # no actualizamos Q en turnos del rival

            # Aplicar la jugada en el entorno
            next_state = state.transition(action)
            next_board = next_state.board
            next_player = next_state.player

            # Calcular recompensa desde la perspectiva del jugador que acaba de mover
            reward = 0.0
            winner = next_state.get_winner()
            done = next_state.is_final()

            if done:
                if winner == player:
                    # El jugador que acaba de mover ganó
                    reward = 1.0
                elif winner == -player:
                    # El jugador que acaba de mover perdió
                    # Si el que perdió fue nuestro agente, castigo más fuerte
                    if player == qagent_plays_as:
                        reward = -2.0
                    else:
                        reward = -1.0
                else:
                    # Empate
                    reward = 0.0
            else:
                # Reward shaping suave: penalización muy pequeña por cada jugada
                # del agente para incentivar ganar más rápido y evitar movimientos inútiles.
                if player == qagent_plays_as:
                    reward = -0.01

            # === Actualizar Q SOLO cuando jugó nuestro agente ===
            if player == qagent_plays_as and state_key is not None:
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

        # Decaimiento epsilon (menos exploración con el tiempo)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Log de progreso
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
