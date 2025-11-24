# groups/GroupD/policy.py
"""
Agente D PRO - MCTS puro y optimizado (sin aprendizaje offline)

Descripción
-----------
Esta implementación es un MCTS "puro" y optimizado para Connect-4.
Diseño objetivo:
 - Mantener la interpretación estadística de MCTS (rollouts uniformes).
 - Mejorar rendimiento por segundo (más simulaciones útiles).
 - Evitar heurísticas sesgadoras dentro de los rollouts.
 - Mantener compatibilidad con Gradescope: clase exportada 'Policy'
   implementa mount(time_out) y act(board: np.ndarray) -> int.

Principales características
---------------------------
- Heurísticas rápidas PRE-MCTS: ganar/bloquear inmediato.
- Reutilización de buffer numpy para playouts (evita construir arrays nuevos).
- Rollouts limitados pero suficientemente profundos para reducir ruido.
- Selección mediante UCB1, expansión 1 hijo por simulación, backprop puro.
- Parámetros afinables: c (exploración UCB), playout_depth.
"""

import time
import math
from typing import Optional, List
import numpy as np

from connect4.policy import Policy as BasePolicy
from connect4.connect_state import ConnectState


# -------------------------
# Utilidades vectorizadas
# -------------------------
def infer_player_fast(board: np.ndarray) -> int:
    neg = int(np.count_nonzero(board == -1))
    pos = int(np.count_nonzero(board == 1))
    return -1 if neg <= pos else 1


def make_move_into(dst: np.ndarray, src: np.ndarray, col: int, player: int) -> int:
    """
    Copia src -> dst y aplica move (col, player) en dst.
    Retorna la fila donde se colocó la ficha, o -1 si columna llena.
    (Evita crear + copiar arrays adicionales).
    """
    np.copyto(dst, src)
    rows = np.where(dst[:, col] == 0)[0]
    if rows.size == 0:
        return -1
    dst[rows[-1], col] = player
    return int(rows[-1])


def check_winner_numpy(board: np.ndarray) -> int:
    # Horizontal
    for r in range(6):
        for c in range(4):
            v = board[r, c]
            if v != 0 and v == board[r, c+1] == board[r, c+2] == board[r, c+3]:
                return int(v)
    # Vertical
    for r in range(3):
        for c in range(7):
            v = board[r, c]
            if v != 0 and v == board[r+1, c] == board[r+2, c] == board[r+3, c]:
                return int(v)
    # Diagonal \
    for r in range(3):
        for c in range(4):
            v = board[r, c]
            if v != 0 and v == board[r+1, c+1] == board[r+2, c+2] == board[r+3, c+3]:
                return int(v)
    # Diagonal /
    for r in range(3, 6):
        for c in range(4):
            v = board[r, c]
            if v != 0 and v == board[r-1, c+1] == board[r-2, c+2] == board[r-3, c+3]:
                return int(v)
    return 0


def can_win_immediate(board: np.ndarray, col: int, player: int) -> bool:
    rows = np.where(board[:, col] == 0)[0]
    if rows.size == 0:
        return False
    r = rows[-1]
    b = board.copy()
    b[r, col] = player
    return check_winner_numpy(b) == player


def uniform_playout_into(buf: np.ndarray, starting_player: int, root_player: int, rng: np.random.Generator, max_depth: int) -> float:
    """
    Playout aleatorio sobre `buf` (se modifica `buf` in-place).
    Se asume que `buf` ya contiene el estado desde el que se simula.
    Retorna valor desde perspectiva root_player: 1.0 (victoria root), 0.5 empate, 0.0 derrota root.
    """
    p = starting_player
    for _ in range(max_depth):
        w = check_winner_numpy(buf)
        if w != 0:
            return 1.0 if w == root_player else 0.0
        free = np.where(buf[0] == 0)[0]
        if free.size == 0:
            return 0.5
        col = int(rng.choice(free))
        rows = np.where(buf[:, col] == 0)[0]
        buf[rows[-1], col] = p
        p = -p
    # Si no concluye en max_depth, evaluar por ganador parcial
    w = check_winner_numpy(buf)
    if w == root_player:
        return 1.0
    if w == -root_player:
        return 0.0
    return 0.5


# -------------------------
# Nodo MCTS ligero
# -------------------------
class Node:
    __slots__ = ("board", "player", "parent", "parent_action", "untried_actions", "children", "visits", "wins")

    def __init__(self, board: np.ndarray, player: int, parent: Optional["Node"] = None, parent_action: Optional[int] = None):
        self.board = board  # numpy array (full copy at node creation)
        self.player = int(player)
        self.parent = parent
        self.parent_action = parent_action
        self.untried_actions = list(np.where(board[0] == 0)[0])
        self.children = {}   # action -> Node
        self.visits = 0
        self.wins = 0.0

"""
Nicolas Urrea y Samuel Acero
"""

# -------------------------
# Clase exportada: Policy
# -------------------------
class AgentR2D2(BasePolicy):
    """
    Agente D PRO (exportado como Policy).

    Constructor
    -----------
    c : float
        Coeficiente de exploración UCB1.
    playout_depth : int
        Profundidad máxima de cada playout aleatorio.
    target_sim_per_move : int
        Si se desea, objetivo de simulaciones por movimiento (se usa como límite auxiliar).
    """
    def __init__(self, c: float = 1.25, playout_depth: int = 18, target_sim_per_move: int = 800):
        self.c = float(c)
        self.playout_depth = int(playout_depth)
        self.target_sim_per_move = int(target_sim_per_move)

        self.rng = np.random.default_rng()
        self.time_out = 1.0

        # buffer reutilizable para simulaciones (int8 para menos memoria)
        self._sim_board = np.zeros((6, 7), dtype=np.int8)

    def mount(self, time_out: int = 10):
        """
        Recibimos time_out (secs) de Gradescope; lo convertimos a budget por movimiento.
        Ajuste conservador: usamos un porcentaje del total para asegurar multiples jugadas.
        """
        # Usar 15% del tiempo total, acotar entre 0.1s y 3.0s por movimiento
        self.time_out = min(3.0, max(0.1, float(time_out) * 0.15))

    def act(self, s: np.ndarray) -> int:
        """
        Decide la acción usando MCTS:
        - heurísticas rápidas para ganar/bloquear
        - MCTS con rollouts uniformes (no sesgados)
        - seleccion UCB1, expansión de 1 hijo por simulación, backprop puro
        """
        current_player = infer_player_fast(s)
        free_cols = np.where(s[0] == 0)[0].tolist()

        if not free_cols:
            raise RuntimeError("No legal moves")
        if len(free_cols) == 1:
            return int(free_cols[0])

        # Heurísticas previas al MCTS: ganar o bloquear ya (sin alterar rollouts)
        for c in free_cols:
            if can_win_immediate(s, c, current_player):
                return int(c)
        for c in free_cols:
            if can_win_immediate(s, c, -current_player):
                return int(c)

        # Inicializar root (hacemos copia única para la raíz)
        root = Node(s.copy().astype(np.int8), current_player)
        root_player = current_player

        # Control de tiempo
        start_time = time.time()
        sims = 0

        # Bucle MCTS: hasta agotar tiempo o alcanzar target_sim_per_move
        while (time.time() - start_time) < self.time_out and sims < self.target_sim_per_move:
            node = root
            player = current_player
            path: List[Node] = [node]

            # Copiar tablero raíz en buffer para simulación
            buf = self._sim_board
            np.copyto(buf, node.board)  # primera copia rápida

            # ----------------------------
            # SELECCIÓN + EXPANSIÓN
            # ----------------------------
            while True:
                # Terminal check
                if check_winner_numpy(buf) != 0 or not (buf[0] == 0).any():
                    break

                # Si acciones sin probar: expandir una
                if node.untried_actions:
                    action = node.untried_actions.pop()
                    # aplicar movimiento sobre buffer (copiando desde node.board)
                    np.copyto(buf, node.board)
                    rows = np.where(buf[:, action] == 0)[0]
                    if rows.size == 0:
                        # improbable: columna ya llena, seguir
                        continue
                    buf[rows[-1], action] = player
                    child_board = buf.copy()  # board guardado en nodo hijo
                    child = Node(child_board, -player, parent=node, parent_action=action)
                    node.children[action] = child
                    node = child
                    path.append(node)
                    player = -player
                    break  # tras expansión vamos a rollout

                # Si completamente expandido: seleccionar mejor hijo por UCB1
                if not node.children:
                    break

                best_score = -float("inf")
                best_child = None
                parent_log = math.log(node.visits + 1)
                for a, ch in node.children.items():
                    if ch.visits == 0:
                        best_child = ch
                        break
                    q = ch.wins / ch.visits
                    score = q + self.c * math.sqrt(parent_log / ch.visits)
                    if score > best_score:
                        best_score = score
                        best_child = ch
                if best_child is None:
                    break
                node = best_child
                path.append(node)
                # preparar buffer para siguiente iteración tomando board del child
                np.copyto(buf, node.board)
                player = node.player

            # ----------------------------
            # ROLLOUT (UNIFORME) desde node.board
            # ----------------------------
            np.copyto(buf, node.board)  # asegurar que buf tenga el estado del nodo
            reward = uniform_playout_into(buf, player, root_player, self.rng, max_depth=self.playout_depth)

            # ----------------------------
            # BACKPROPAGATION
            # ----------------------------
            for n in path:
                n.visits += 1
                n.wins += reward

            sims += 1

        # Seleccionar acción con más visitas
        if not root.children:
            # Fallback centrado si no hay expansión
            preferred = [3, 2, 4, 1, 5, 0, 6]
            for c in preferred:
                if c in free_cols:
                    return int(c)
            return int(free_cols[0])

        best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
        return int(best_action)
