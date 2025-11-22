# groups/GroupC/policy.py  (ajusta la ruta según tu grupo)
import numpy as np
from connect4.policy import Policy
from typing import override
import os
import pickle


class QLearningPolicy(Policy):
    """
    Agente de Q-learning para Connect 4.

    Durante el torneo:
      - mount() carga la Q-table entrenada (q_table.pkl).
      - act() juega de forma greedy usando esa tabla (epsilon_eval).

    Para entrenamiento usaremos los métodos:
      - encode_state, select_action_for_training, update_q, save_q_table.
    """

    ACTIONS = list(range(7))

    def __init__(self, epsilon_eval: float = 0.0):
        # epsilon que se usa SOLO en torneo (explotación)
        self.epsilon_eval = epsilon_eval
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}
        self._rng = np.random.default_rng()

    # ------------------- Interfaz Policy -------------------

    @override
    def mount(self) -> None:
        """Carga la Q-table entrenada desde disco (si existe)."""
        path = "q_table.pkl"  # archivo en la raíz del proyecto
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
            # print(f"[QLearningPolicy] Q-table cargada con {len(self.q_table)} estados")
        else:
            self.q_table = {}
            # print("[QLearningPolicy] No se encontró q_table.pkl, usando Q-table vacía")

    @override
    def act(self, s: np.ndarray) -> int:
        """
        Elige una acción para el tablero s durante el torneo.
        Usa epsilon_eval (normalmente 0.0 -> greedy).
        """
        return self._select_action_from_q(s, epsilon=self.epsilon_eval)

    # ------------------- Helpers de estado/acciones -------------------

    @staticmethod
    def _infer_player_to_move(board: np.ndarray) -> int:
        """
        Deduce quién debe jugar a partir del tablero:
        - Si fichas rojas (-1) == fichas amarillas (1) -> le toca al primero (-1).
        - Si rojas = amarillas + 1 -> le toca al segundo (1).
        """
        n_neg = int(np.sum(board == -1))
        n_pos = int(np.sum(board == 1))
        return -1 if n_neg == n_pos else 1

    @staticmethod
    def encode_state(board: np.ndarray, player_to_move: int) -> tuple[int, ...]:
        """
        Representación canónica: las fichas del jugador que va a mover son +1
        y las del rival son -1.
        """
        canonical = board * player_to_move
        return tuple(canonical.astype(int).flatten())

    def _get_q_values(self, state_key: tuple[int, ...]) -> np.ndarray:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.ACTIONS), dtype=float)
        return self.q_table[state_key]

    @staticmethod
    def _available_actions(board: np.ndarray) -> list[int]:
        return [c for c in range(board.shape[1]) if board[0, c] == 0]

    # ------------------- Selección de acción -------------------

    def _select_action_from_q(self, board: np.ndarray, epsilon: float) -> int:
        """
        Selección de acción genérica (ε-greedy) usando la Q-table.
        La usamos en act() y también la puedes usar en entrenamiento si quieres.
        """
        available = self._available_actions(board)
        if not available:
            return 0  # debería no pasar porque no se llama act() en estados finales

        player_to_move = self._infer_player_to_move(board)
        state_key = self.encode_state(board, player_to_move)
        q_values = self._get_q_values(state_key).copy()

        # Penalizar acciones inválidas
        for a in self.ACTIONS:
            if a not in available:
                q_values[a] = -1e9

        if self._rng.random() < epsilon:
            return int(self._rng.choice(available))
        return int(np.argmax(q_values))

    # -------- Métodos específicos para ENTRENAR la Q-table --------

    def select_action_for_training(
        self, board: np.ndarray, player_to_move: int, epsilon: float
    ):
        """
        Versión de selección de acción para entrenamiento.
        Devuelve también la clave de estado para poder actualizar Q.
        """
        available = self._available_actions(board)
        if not available:
            return 0, None

        state_key = self.encode_state(board, player_to_move)
        q_values = self._get_q_values(state_key).copy()

        for a in self.ACTIONS:
            if a not in available:
                q_values[a] = -1e9

        if self._rng.random() < epsilon:
            action = int(self._rng.choice(available))
        else:
            action = int(np.argmax(q_values))

        return action, state_key

    def update_q(
        self,
        state_key: tuple[int, ...],
        action: int,
        reward: float,
        next_board: np.ndarray,
        next_player: int,
        alpha: float,
        gamma: float,
        done: bool,
    ) -> None:
        """
        Actualiza Q(s,a) con la regla de Q-learning.
        """
        q_values = self._get_q_values(state_key)
        current_q = q_values[action]

        if done:
            target = reward
        else:
            next_available = self._available_actions(next_board)
            if not next_available:
                target = reward
            else:
                next_key = self.encode_state(next_board, next_player)
                next_q = self._get_q_values(next_key).copy()
                for a in self.ACTIONS:
                    if a not in next_available:
                        next_q[a] = -1e9
                target = reward + gamma * float(np.max(next_q))

        q_values[action] = current_q + alpha * (target - current_q)

    def save_q_table(self, path: str = "q_table.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
