"""
Agente C - OhYes (MCTS Ultra-Optimizado)
----------------------------------------------------------
Optimizado para Gradescope con múltiples partidas en tiempo límite.
"""

import math
import numpy as np
import time
from typing import Optional
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class Node:
    __slots__ = ('state', 'parent', 'parent_action', 'untried_actions', 
                 'children', 'visits', 'wins', '_board', '_player')
    
    def __init__(self, board: np.ndarray, player: int, parent: Optional["Node"] = None, 
                 parent_action: Optional[int] = None):
        self._board = board
        self._player = player
        self.state = None  # Lazy initialization
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
        # Check rápido sin crear ConnectState
        return len(self.untried_actions) == 0 and len(self.children) == 0 or \
               check_winner_numpy(self._board) != 0 or not (self._board[0] == 0).any()


class OhYes(Policy):
    def __init__(self, c: float = 1.41):
        self.c = c
        self.rng = np.random.default_rng()
        self.time_out = 1.5  # Tiempo agresivo por movimiento
        
    def mount(self, time_out: int = 10) -> None:
        # Tiempo muy reducido para permitir muchas partidas
        self.time_out = min(2.0, max(0.5, float(time_out) * 0.15))

    def act(self, s: np.ndarray) -> int:
        current_player = infer_player_fast(s)
        
        # Validación rápida inicial
        free_cols = np.where(s[0] == 0)[0]
        if len(free_cols) == 0:
            raise RuntimeError("No legal actions available")
        
        if len(free_cols) == 1:
            return int(free_cols[0])
        
        # Check si podemos ganar inmediatamente
        for col in free_cols:
            if can_win_immediate(s, col, current_player):
                return int(col)
        
        # Check si debemos bloquear
        opp = -current_player
        for col in free_cols:
            if can_win_immediate(s, col, opp):
                return int(col)
        
        # MCTS reducido
        root = Node(s.copy(), current_player)
        root_player = current_player
        
        start_time = time.time()
        simulations = 0
        max_simulations = 100  # Límite de simulaciones
        
        while (time.time() - start_time < self.time_out) and (simulations < max_simulations):
            board_copy = s.copy()
            player = current_player
            node = root
            path = [node]
            
            # SELECCIÓN + EXPANSIÓN combinadas
            while True:
                if check_winner_numpy(board_copy) != 0 or not (board_copy[0] == 0).any():
                    break
                
                free = np.where(board_copy[0] == 0)[0]
                if len(free) == 0:
                    break
                
                # Si tiene acciones no probadas, expandir
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
                
                # Si está completamente expandido, seleccionar mejor hijo
                if not node.children:
                    break
                
                # UCB1 inline
                best_score = -float("inf")
                best_child = None
                best_action = None
                parent_log = math.log(node.visits + 1)
                
                for action, child in node.children.items():
                    if child.visits == 0:
                        best_child = child
                        best_action = action
                        break
                    q = child.wins / child.visits
                    score = q + self.c * math.sqrt(parent_log / child.visits)
                    if score > best_score:
                        best_score = score
                        best_child = child
                        best_action = action
                
                if best_child is None:
                    break
                
                node = best_child
                path.append(node)
                board_copy = make_move_numpy(board_copy, best_action, player)
                player = -player
            
            # SIMULACIÓN ultra-rápida (máximo 10 movimientos)
            reward = fast_playout(board_copy, player, root_player, max_depth=10)
            
            # BACKPROPAGACIÓN
            for n in path:
                n.visits += 1
                n.wins += reward
            
            simulations += 1
        
        # Selección final
        if not root.children:
            # Heurística: preferir centro
            preferred = [3, 2, 4, 1, 5, 0, 6]
            for col in preferred:
                if col in free_cols:
                    return int(col)
            return int(free_cols[0])
        
        best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
        return int(best_action)


# ========================
# FUNCIONES NUMPY PURAS
# ========================

def infer_player_fast(board: np.ndarray) -> int:
    neg = np.count_nonzero(board == -1)
    pos = np.count_nonzero(board == 1)
    return -1 if neg <= pos else 1


def make_move_numpy(board: np.ndarray, col: int, player: int) -> np.ndarray:
    """Hace una jugada y retorna nuevo board."""
    new_board = board.copy()
    row = np.where(new_board[:, col] == 0)[0]
    if len(row) > 0:
        new_board[row[-1], col] = player
    return new_board


def can_win_immediate(board: np.ndarray, col: int, player: int) -> bool:
    """Check si jugar en col gana inmediatamente."""
    row_idx = np.where(board[:, col] == 0)[0]
    if len(row_idx) == 0:
        return False
    
    row = row_idx[-1]
    test_board = board.copy()
    test_board[row, col] = player
    return check_winner_numpy(test_board) == player


def check_winner_numpy(board: np.ndarray) -> int:
    """Versión ultra-optimizada de check winner."""
    # Horizontal
    for r in range(6):
        for c in range(4):
            val = board[r, c]
            if val != 0:
                if board[r, c] == board[r, c+1] == board[r, c+2] == board[r, c+3]:
                    return val
    
    # Vertical
    for r in range(3):
        for c in range(7):
            val = board[r, c]
            if val != 0:
                if board[r, c] == board[r+1, c] == board[r+2, c] == board[r+3, c]:
                    return val
    
    # Diagonal \
    for r in range(3):
        for c in range(4):
            val = board[r, c]
            if val != 0:
                if board[r, c] == board[r+1, c+1] == board[r+2, c+2] == board[r+3, c+3]:
                    return val
    
    # Diagonal /
    for r in range(3, 6):
        for c in range(4):
            val = board[r, c]
            if val != 0:
                if board[r, c] == board[r-1, c+1] == board[r-2, c+2] == board[r-3, c+3]:
                    return val
    
    return 0


def fast_playout(board: np.ndarray, player: int, root_player: int, max_depth: int = 10) -> float:
    """Playout ultra-rápido con profundidad limitada."""
    current_board = board
    current_player = player
    
    for _ in range(max_depth):
        winner = check_winner_numpy(current_board)
        if winner != 0:
            return 1.0 if winner == root_player else 0.0
        
        free = np.where(current_board[0] == 0)[0]
        if len(free) == 0:
            return 0.5
        
        # Movimiento aleatorio
        col = np.random.choice(free)
        current_board = make_move_numpy(current_board, col, current_player)
        current_player = -current_player
    
    # Si no terminó en max_depth, evaluación heurística
    winner = check_winner_numpy(current_board)
    if winner == root_player:
        return 1.0
    elif winner == -root_player:
        return 0.0
    return 0.5