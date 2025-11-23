import numpy as np
import pickle
import os
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class QLearningPolicy(Policy):
    """
    Agente Q-learning para Connect-4.
    - En entrenamiento se actualiza una tabla Q(s,a).
    - En el torneo, simplemente carga esa Q-table y juega greedy.
    """

    ACTIONS = list(range(7))  # columnas 0..6

    def __init__(self, epsilon_eval: float = 0.0):
        # epsilon_eval = 0 → juega greedy en torneo
        self.epsilon_eval = epsilon_eval
        self.q_table = {}
        self._rng = np.random.default_rng()

    # -------------------------
    #   MÉTODOS OBLIGATORIOS
    # -------------------------

    def mount(self, time_out: int = None) -> None:
        """
        Carga la Q-table entrenada desde disco.
        Aceptamos time_out para ser compatibles con Gradescope.
        """
        path = "q_table.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
        else:
            self.q_table = {}

    def act(self, s: np.ndarray) -> int:
        # Acciones disponibles
        available = [c for c in range(7) if s[0, c] == 0]
    
        # QUIÉN JUEGA AHORA
        player = self._infer_player_to_move(s)

        # 1. Si YO puedo ganar → gano
        for c in available:
            if self._can_win_immediate(s, c, player):
                return c

        # 2. Si el random puede ganar en el siguiente → bloqueo
        opp = -player
        for c in available:
            if self._can_win_immediate(s, c, opp):
                return c

        # 3. Si no hay urgencia → usar Q-learning greedy
        return self._select_action_from_q(s, epsilon=self.epsilon_eval)

    def _can_win_immediate(self, board, col, player):
        # Encuentra la fila disponible en esa columna
        rows = np.where(board[:, col] == 0)[0]
        if len(rows) == 0:
            return False

        r = rows[-1]
        temp = board.copy()
        temp[r, col] = player

        # Revisar si esa jugada gana
        winner = ConnectState(board=temp, player=player).get_winner()
        return winner == player

    # -------------------------
    #   UTILIDADES INTERNAS
    # -------------------------

    @staticmethod
    def _infer_player_to_move(board: np.ndarray) -> int:
        """
        Deduce cuál jugador debe mover en base al conteo de fichas.
        -1 = rojo, 1 = amarillo.
        """
        n_neg = int(np.sum(board == -1))
        n_pos = int(np.sum(board == 1))
        return -1 if n_neg == n_pos else 1

    @staticmethod
    def _available_actions(board: np.ndarray):
        """Devuelve columnas libres."""
        return [c for c in range(board.shape[1]) if board[0, c] == 0]

    @staticmethod
    def encode_state(board: np.ndarray, player_to_move: int):
        """
        Representación canónica del estado:
        - Fichas del jugador que mueve se vuelven +1
        - Fichas rivales se vuelven -1
        """
        canonical = board * player_to_move
        return tuple(canonical.astype(int).flatten())

    def _get_q_values(self, state_key):
        """Devuelve los Q-values para un estado; si no existe, lo crea."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.ACTIONS), dtype=float)
        return self.q_table[state_key]

    # -------------------------
    #   SELECCIÓN DE ACCIÓN
    # -------------------------

    def _select_action_from_q(self, board: np.ndarray, epsilon: float) -> int:
        """
        Política ε-greedy genérica. En torneo epsilon=0 → greedy puro.
        """
        available = self._available_actions(board)
        if not available:
            return 0  # caso extremo

        player_to_move = self._infer_player_to_move(board)
        state_key = self.encode_state(board, player_to_move)

        q_values = self._get_q_values(state_key).copy()

        # Penalizar acciones inválidas
        for a in self.ACTIONS:
            if a not in available:
                q_values[a] = -1e9

        # Exploración
        if self._rng.random() < epsilon:
            return int(self._rng.choice(available))

        # Explotación (greedy)
        return int(np.argmax(q_values))

    # ---------------------------------------------------------
    #   MÉTODOS USADOS SOLO EN ENTRENAMIENTO (NO EN TORNEO)
    # ---------------------------------------------------------

    def select_action_for_training(self, board, player_to_move, epsilon):
        """
        Igual que _select_action_from_q pero además devuelve la clave de estado.
        """
        available = self._available_actions(board)
        if not available:
            return 0, None

        state_key = self.encode_state(board, player_to_move)
        q_values = self._get_q_values(state_key).copy()

        # Penalizar inválidas
        for a in self.ACTIONS:
            if a not in available:
                q_values[a] = -1e9

        if self._rng.random() < epsilon:
            action = int(self._rng.choice(available))
        else:
            action = int(np.argmax(q_values))

        return action, state_key

    def update_q(self, state_key, action, reward,
                next_board, next_player, alpha, gamma, done):
        """
        Regla de actualización estándar de Q-learning.
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
                # Penalizar acciones inválidas
                for a in self.ACTIONS:
                    if a not in next_available:
                        next_q[a] = -1e9

                target = reward + gamma * float(np.max(next_q))

        q_values[action] = current_q + alpha * (target - current_q)

    def save_q_table(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
            
            

