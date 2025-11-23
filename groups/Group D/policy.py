"""
Agente D - (MCTS Optimizado con Rollout Mejorado)
----------------------------------------------------------

Esta versión del agente MCTS incluye:

✓ MCTS clásico ultra optimizado  
✓ Heurísticas de victoria/bloqueo inmediato  
✓ Playouts rápidos (fast_playout)  
✓ NUEVO: Rollouts mejorados (smart_playout)  
✓ NUEVO: value_fn para soportar entrenamiento offline  
✓ NUEVO: hashing eficiente del estado del tablero  

smart_playout usa:
1. value_fn si existe → valoración aprendida
2. chequeos heurísticos (victoria inmediata)
3. playout aleatorio rápido como fallback

"""

import math
import numpy as np
import time
from typing import Optional
from connect4.policy import Policy
from connect4.connect_state import ConnectState


# ========================
# NODO DEL ÁRBOL MCTS
# ========================

class Node:
    __slots__ = ('state','parent','parent_action',
                 'untried_actions','children','visits',
                 'wins','_board','_player')
    
    def __init__(self, board: np.ndarray, player: int,
                 parent: Optional["Node"]=None,
                 parent_action: Optional[int]=None):
        
        self._board = board
        self._player = player
        self.state = None
        
        self.parent = parent
        self.parent_action = parent_action
        
        self.untried_actions = list(np.where(board[0] == 0)[0])
        self.children = {}
        self.visits = 0
        self.wins = 0.0

    def get_state(self) -> ConnectState:
        if self.state is None:
            self.state = ConnectState(board=self._board, player=self._player)
        return self.state

    def is_terminal(self) -> bool:
        return len(self.untried_actions) == 0 and len(self.children) == 0 or \
               check_winner_numpy(self._board) != 0 or not (self._board[0] == 0).any()


# ========================
# POLÍTICA MCTS + ROLLOUT MEJORADO
# ========================

class AgentD(Policy):

    def __init__(self, c: float = 1.41):
        self.c = c
        self.rng = np.random.default_rng()
        self.time_out = 1.5
        
        # NUEVO: tabla de valor aprendida (state_hash → value)
        self.value_fn = {}  

    def mount(self, time_out: int = 10):
        # reducir tiempo por seguridad
        self.time_out = min(2.0, max(0.5, float(time_out)*0.15))

    # ========================
    # HASH DEL TABLERO (para aprendizaje)
    # ========================
    def hash_board(self, board: np.ndarray) -> int:
        return hash(board.tobytes())

    # ========================
    # ROLLOUT MEJORADO
    # ========================
    def smart_playout(self, board: np.ndarray, player: int, root_player: int) -> float:
        """
        Rollout mejorado:
        1) Usa value_fn si existe
        2) Heurísticas rápidas de victoria
        3) Fallback: playout aleatorio corto
        """
        # 1. Si el estado aparece en la tabla de valor → usarlo
        h = self.hash_board(board)
        if h in self.value_fn:
            return self.value_fn[h]

        # 2. Heurística rápida: victoria inmediata detectada
        winner = check_winner_numpy(board)
        if winner != 0:
            return 1.0 if winner == root_player else 0.0

        # 3. Heurística: ¿hay jugada ganadora inmediata?
        free = np.where(board[0] == 0)[0]
        for col in free:
            if can_win_immediate(board, col, root_player):
                return 1.0
            if can_win_immediate(board, col, -root_player):
                return 0.0

        # 4. Playout aleatorio corto
        return fast_playout(board, player, root_player, max_depth=6)

    # ========================
    # MÉTODO PRINCIPAL DEL AGENTE
    # ========================
    def act(self, s: np.ndarray) -> int:
        current_player = infer_player_fast(s)
        free_cols = np.where(s[0] == 0)[0]

        if len(free_cols) == 0:
            raise RuntimeError("No legal moves")

        if len(free_cols) == 1:
            return int(free_cols[0])

        # Heurística 1: ganar ahora
        for col in free_cols:
            if can_win_immediate(s, col, current_player):
                return int(col)

        # Heurística 2: bloquear
        opp = -current_player
        for col in free_cols:
            if can_win_immediate(s, col, opp):
                return int(col)

        # Inicializar raíz MCTS
        root = Node(s.copy(), current_player)
        root_player = current_player

        start = time.time()
        simulations = 0
        max_simulations = 120

        while time.time() - start < self.time_out and simulations < max_simulations:

            board_copy = s.copy()
            player = current_player
            node = root
            path = [node]

            # ========================
            # SELECCIÓN + EXPANSIÓN
            # ========================
            while True:
                if check_winner_numpy(board_copy) != 0 or not (board_copy[0] == 0).any():
                    break

                if node.untried_actions:
                    action = node.untried_actions.pop()
                    new_board = make_move_numpy(board_copy, action, player)
                    child = Node(new_board, -player, parent=node, parent_action=action)
                    node.children[action] = child
                    node = child
                    path.append(node)
                    board_copy = new_board
                    player = -player
                    break

                if not node.children:
                    break

                # UCB1
                best_score = -float("inf")
                best_child = None
                best_action = None
                logN = math.log(node.visits+1)

                for a, child in node.children.items():
                    if child.visits == 0:
                        best_child = child
                        best_action = a
                        break

                    q = child.wins / child.visits
                    ucb = q + self.c * math.sqrt(logN / child.visits)
                    if ucb > best_score:
                        best_score = ucb
                        best_child = child
                        best_action = a

                if best_child is None:
                    break

                node = best_child
                path.append(node)
                board_copy = make_move_numpy(board_copy, best_action, player)
                player = -player

            # ========================
            # SIMULACIÓN (ROLLOUT MEJORADO)
            # ========================
            reward = self.smart_playout(board_copy, player, root_player)

            # ========================
            # BACKPROPAGACIÓN
            # ========================
            for n in path:
                n.visits += 1
                n.wins  += reward

            simulations += 1

        # Selección final
        if not root.children:
            preferred = [3,2,4,1,5,0,6]
            for col in preferred:
                if col in free_cols:
                    return int(col)
            return int(free_cols[0])

        best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
        return int(best_action)



# ========================
# FUNCIONES AUXILIARES (SIN CAMBIO)
# ========================

def infer_player_fast(board):
    neg = np.count_nonzero(board == -1)
    pos = np.count_nonzero(board == 1)
    return -1 if neg <= pos else 1

def make_move_numpy(board,col,player):
    b = board.copy()
    row = np.where(b[:,col]==0)[0]
    if len(row)>0:
        b[row[-1],col] = player
    return b

def can_win_immediate(board,col,player):
    row_idx = np.where(board[:,col]==0)[0]
    if len(row_idx)==0:
        return False
    r = row_idx[-1]
    b = board.copy()
    b[r,col] = player
    return check_winner_numpy(b) == player

def check_winner_numpy(board):
    for r in range(6):
        for c in range(4):
            v = board[r,c]
            if v!=0 and v==board[r,c+1]==board[r,c+2]==board[r,c+3]:
                return v
    for r in range(3):
        for c in range(7):
            v = board[r,c]
            if v!=0 and v==board[r+1,c]==board[r+2,c]==board[r+3,c]:
                return v
    for r in range(3):
        for c in range(4):
            v = board[r,c]
            if v!=0 and v==board[r+1,c+1]==board[r+2,c+2]==board[r+3,c+3]:
                return v
    for r in range(3,6):
        for c in range(4):
            v = board[r,c]
            if v!=0 and v==board[r-1,c+1]==board[r-2,c+2]==board[r-3,c+3]:
                return v
    return 0

def fast_playout(board,player,root_player,max_depth=10):
    b = board
    p = player
    for _ in range(max_depth):
        w = check_winner_numpy(b)
        if w!=0:
            return 1.0 if w==root_player else 0.0
        free = np.where(b[0]==0)[0]
        if len(free)==0:
            return 0.5
        col = np.random.choice(free)
        b = make_move_numpy(b,col,p)
        p = -p
    w = check_winner_numpy(b)
    if w==root_player: return 1.0
    if w==-root_player: return 0.0
    return 0.5
