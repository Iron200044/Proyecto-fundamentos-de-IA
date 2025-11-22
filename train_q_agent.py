import numpy as np
from connect4.connect_state import ConnectState

# IMPORTA TU AGENTE
from groups.GroupB.policy import QLearningPolicy

# IMPORTA EL HEURÍSTICO DEL GRUPO A
from groups.GroupA.policy import Aha


# ----------------------
#   RIVALES DE ENTRENAMIENTO
# ----------------------

class RandomOpponent:
    """Jugador totalmente aleatorio."""
    def act(self, board):
        avail = [c for c in range(7) if board[0, c] == 0]
        return int(np.random.choice(avail))


class HeuristicOpponent:
    """Wrapper para usar el agente Aha como rival."""
    def __init__(self):
        self.agent = Aha()
        self.agent.mount()

    def act(self, board):
        return self.agent.act(board)


class SelfOpponent:
    """Versión ligera del propio Q-learning para self-play."""
    def __init__(self, qagent: QLearningPolicy):
        self.qagent = qagent

    def act(self, board):
        # Self-play usa epsilon ALTO para generar variedad
        return self.qagent._select_action_from_q(board, epsilon=0.4)


# ----------------------
#   FUNCIÓN DE ENTRENAMIENTO
# ----------------------

def train_q_agent(
    num_episodes: int = 60000,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.9995,
    save_path: str = "q_table.pkl",
):

    qagent = QLearningPolicy(epsilon_eval=0.0)
    epsilon = epsilon_start

    for ep in range(1, num_episodes + 1):

        # Selección del rival (IMPORTANTE)
        r = np.random.random()

        if r < 0.60:
            opponent = RandomOpponent()
        elif r < 0.90:
            opponent = HeuristicOpponent()   # <-- ENTRENAMIENTO CONTRA HEURÍSTICO
        else:
            opponent = SelfOpponent(qagent)  # <-- Self-play (pero poco)

        state = ConnectState()
        done = False

        while not state.is_final():

            board = state.board
            player = state.player

            # --- Juega el Q-agent o el oponente ---
            if player == -1:  # Q-agent
                action, state_key = qagent.select_action_for_training(
                    board, player, epsilon
                )
            else:  # Rival
                action = opponent.act(board)
                state_key = None

            # Aplicar jugada
            next_state = state.transition(action)
            next_board = next_state.board
            next_player = next_state.player

            # Recompensa
            reward = 0.0
            winner = next_state.get_winner()
            done = next_state.is_final()

            if done:
                if winner == player:
                    reward = 1.0      # ganar
                elif winner == -player:
                    reward = -1.0     # perder
                else:
                    reward = 0.0      # empate

            # Actualizar Q SOLO si jugó el Q-agent
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

            state = next_state

        # Decaimiento epsilon por episodio
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % 1000 == 0:
            print(
                f"Episode {ep}/{num_episodes} "
                f"- epsilon={epsilon:.3f} "
                f"- Q states: {len(qagent.q_table)}"
            )

    # Guardar Q-table al final
    qagent.save_q_table(save_path)
    print(f"Entrenamiento terminado. Q-table guardada en {save_path}.")


if __name__ == "__main__":
    train_q_agent()
