"""
Agente C - OhYes (MCTS Ultra-Optimizado)
----------------------------------------------------------

Este agente implementa Monte Carlo Tree Search (MCTS) con la estrategia UCT
(Upper Confidence bounds applied to Trees) para jugar Connect-4 de manera
competitiva.

MCTS es un algoritmo de búsqueda que balancea exploración y explotación
mediante simulaciones aleatorias (playouts) para estimar el valor de diferentes
movimientos sin necesidad de evaluar exhaustivamente todo el árbol de juego.

Las 4 fases principales de MCTS son:
1. SELECCIÓN: Navegar por el árbol usando UCB1 hasta encontrar un nodo no completamente expandido
2. EXPANSIÓN: Agregar un nuevo nodo hijo al árbol
3. SIMULACIÓN (Rollout/Playout): Jugar aleatoriamente hasta el final desde el nuevo nodo
4. BACKPROPAGACIÓN: Propagar el resultado hacia arriba actualizando estadísticas

Optimizaciones clave para Gradescope:
- Uso intensivo de numpy para operaciones vectorizadas
- Lazy initialization de objetos ConnectState
- Heurísticas de victoria/bloqueo inmediato
- Límites de tiempo y simulaciones agresivos
- Playouts con profundidad limitada
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
    """
    Representa un nodo en el árbol de búsqueda MCTS.
    
    Cada nodo almacena:
    - El estado del juego (board + player a turno)
    - Estadísticas de visitas y victorias
    - Referencias a nodos padre e hijos
    - Lista de acciones aún no exploradas
    
    Atributos
    ---------
    __slots__ : tuple
        Optimización de memoria: define exactamente qué atributos puede tener el nodo.
        Reduce uso de memoria ~30% y acelera acceso a atributos ~20%.
    
    _board : np.ndarray
        Estado del tablero (6x7) con valores -1, 0, 1
    
    _player : int
        Jugador que debe mover en este nodo (-1 o 1)
    
    state : ConnectState | None
        Estado completo del juego. Se inicializa lazy (solo cuando se necesita)
        para evitar overhead de crear objetos ConnectState innecesariamente.
    
    parent : Node | None
        Referencia al nodo padre en el árbol
    
    parent_action : int | None
        La acción (columna) que llevó del padre a este nodo
    
    untried_actions : list[int]
        Lista mutable de acciones (columnas) que aún no han sido expandidas.
        Se va consumiendo durante la fase de EXPANSIÓN.
    
    children : dict[int, Node]
        Diccionario que mapea acciones (columnas) a nodos hijos.
        Solo contiene acciones que ya fueron expandidas.
    
    visits : int
        Número de veces que este nodo ha sido visitado durante las simulaciones MCTS.
        Usado en la fórmula UCB1 para balancear exploración/explotación.
    
    wins : float
        Suma acumulada de recompensas obtenidas en todas las simulaciones que
        pasaron por este nodo. La recompensa es desde la perspectiva del
        jugador raíz (root_player):
        - 1.0 si root_player ganó
        - 0.5 si fue empate
        - 0.0 si root_player perdió
    """
    
    __slots__ = ('state', 'parent', 'parent_action', 'untried_actions', 
                 'children', 'visits', 'wins', '_board', '_player')
    
    def __init__(self, board: np.ndarray, player: int, parent: Optional["Node"] = None, 
                 parent_action: Optional[int] = None):
        """
        Inicializa un nuevo nodo del árbol MCTS.
        
        Parámetros
        ----------
        board : np.ndarray
            Estado actual del tablero (6x7)
        player : int
            Jugador que debe mover (-1 = rojo, 1 = amarillo)
        parent : Node | None
            Nodo padre (None para la raíz)
        parent_action : int | None
            Acción que llevó del padre a este nodo
        """
        self._board = board
        self._player = player
        self.state = None  # Lazy initialization: solo se crea cuando se necesita
        self.parent = parent
        self.parent_action = parent_action
        
        # Identificar acciones disponibles directamente desde numpy (más rápido que get_free_cols())
        # np.where devuelve índices donde la condición es True
        self.untried_actions = list(np.where(board[0] == 0)[0])
        
        self.children = {}  # Se irá llenando durante expansión
        self.visits = 0     # Contador de visitas para UCB1
        self.wins = 0.0     # Acumulador de recompensas

    def get_state(self) -> ConnectState:
        """
        Obtiene el ConnectState completo, creándolo solo si es necesario (lazy).
        
        Esto evita crear objetos ConnectState innecesariamente durante MCTS,
        ya que la mayoría de operaciones se pueden hacer directamente con numpy.
        
        Retorna
        -------
        ConnectState
            Estado completo del juego
        """
        if self.state is None:
            self.state = ConnectState(board=self._board, player=self._player)
        return self.state

    def is_terminal(self) -> bool:
        """
        Verifica si este nodo representa un estado terminal (juego terminado).
        
        Un nodo es terminal si:
        - No hay acciones disponibles Y no tiene hijos (tablero lleno sin expandir)
        - Hay un ganador
        - El tablero está completamente lleno
        
        Retorna
        -------
        bool
            True si es estado terminal, False en caso contrario
        """
        return len(self.untried_actions) == 0 and len(self.children) == 0 or \
               check_winner_numpy(self._board) != 0 or not (self._board[0] == 0).any()


# ========================
# POLÍTICA MCTS
# ========================

class OhYes(Policy):
    """
    Implementación de Monte Carlo Tree Search (MCTS) para Connect-4.
    
    MCTS construye un árbol de búsqueda de forma asimétrica, dedicando más
    recursos (simulaciones) a las ramas más prometedoras del árbol.
    
    El algoritmo itera continuamente sobre 4 fases hasta agotar el tiempo:
    
    1. **SELECCIÓN**: Desde la raíz, descender por el árbol eligiendo el mejor
       hijo según UCB1 (Upper Confidence Bound) hasta encontrar un nodo que
       no está completamente expandido o es terminal.
    
    2. **EXPANSIÓN**: Si el nodo tiene acciones no probadas, elegir una y
       crear un nuevo nodo hijo.
    
    3. **SIMULACIÓN (Playout)**: Desde el nuevo nodo, jugar aleatoriamente
       hasta el final del juego (o hasta un límite de profundidad).
    
    4. **BACKPROPAGACIÓN**: Propagar el resultado de la simulación hacia arriba,
       actualizando las estadísticas (visits, wins) de todos los nodos en el camino.
    
    Parámetros
    ----------
    c : float
        Constante de exploración para UCB1 (por defecto √2 ≈ 1.41).
        Controla el balance exploración vs explotación:
        - c alto → más exploración (probar acciones menos visitadas)
        - c bajo → más explotación (elegir acciones con mejor win rate)
    """
    
    def __init__(self, c: float = 1.41):
        """
        Inicializa el agente MCTS.
        
        Parámetros
        ----------
        c : float
            Constante de exploración UCB1 (default: 1.41 ≈ √2)
        """
        self.c = c
        self.rng = np.random.default_rng()  # Generador de números aleatorios
        self.time_out = 1.5  # Tiempo base por movimiento (será ajustado en mount)
        
    def mount(self, time_out: int = 10) -> None:
        """
        Configura el agente con el tiempo límite por movimiento.
        
        Gradescope llama este método antes de cada torneo pasando el tiempo
        límite total. Lo reducimos agresivamente (15%) para permitir múltiples
        partidas dentro del límite de 600s de Gradescope.
        
        Parámetros
        ----------
        time_out : int
            Tiempo límite en segundos proporcionado por el sistema
        """
        # Usar solo 15% del tiempo límite para ser conservadores
        # min(2.0, ...) asegura no exceder 2 segundos incluso si time_out es grande
        # max(0.5, ...) asegura mínimo 0.5 segundos para tener tiempo de pensar
        self.time_out = min(2.0, max(0.5, float(time_out) * 0.15))

    def act(self, s: np.ndarray) -> int:
        """
        Decide la mejor acción usando MCTS.
        
        Proceso:
        1. Verificar acciones legales y casos triviales
        2. Aplicar heurísticas de victoria/bloqueo inmediato
        3. Ejecutar loop MCTS hasta agotar tiempo
        4. Retornar la acción más visitada
        
        Parámetros
        ----------
        s : np.ndarray
            Estado del tablero (6x7) con valores:
            -1 = ficha roja, 0 = vacío, 1 = ficha amarilla
        
        Retorna
        -------
        int
            Columna donde jugar (0-6)
        """
        # -----------------------------------------------
        # PASO 1: Inferir quién juega y validar estado
        # -----------------------------------------------
        current_player = infer_player_fast(s)
        
        # Identificar columnas libres directamente con numpy
        free_cols = np.where(s[0] == 0)[0]
        if len(free_cols) == 0:
            raise RuntimeError("No legal actions available")
        
        # Caso trivial: si solo hay una columna disponible, jugarla inmediatamente
        if len(free_cols) == 1:
            return int(free_cols[0])
        
        # -----------------------------------------------
        # PASO 2: Heurísticas rápidas (pre-MCTS)
        # -----------------------------------------------
        # Estas heurísticas son mucho más rápidas que MCTS y evitan
        # perder tiempo buscando cuando hay jugadas obvias.
        
        # Heurística 1: Si podemos ganar inmediatamente, hacerlo
        for col in free_cols:
            if can_win_immediate(s, col, current_player):
                return int(col)
        
        # Heurística 2: Si el oponente puede ganar, bloquearlo
        opp = -current_player
        for col in free_cols:
            if can_win_immediate(s, col, opp):
                return int(col)
        
        # -----------------------------------------------
        # PASO 3: Ejecutar MCTS
        # -----------------------------------------------
        # Inicializar la raíz del árbol de búsqueda
        root = Node(s.copy(), current_player)
        root_player = current_player  # Guardar para calcular recompensas consistentemente
        
        start_time = time.time()
        simulations = 0
        max_simulations = 100  # Límite de simulaciones para evitar loops infinitos
        
        # Loop principal de MCTS: repetir hasta agotar tiempo o simulaciones
        while (time.time() - start_time < self.time_out) and (simulations < max_simulations):
            
            # Preparar para una nueva simulación
            board_copy = s.copy()  # Copia del tablero para modificar durante simulación
            player = current_player
            node = root
            path = [node]  # Guardar el camino recorrido para backpropagación
            
            # -----------------------------------------------
            # FASE 1: SELECCIÓN + EXPANSIÓN (combinadas)
            # -----------------------------------------------
            # Navegamos por el árbol hasta encontrar un nodo para expandir
            while True:
                # Verificar si llegamos a un estado terminal
                if check_winner_numpy(board_copy) != 0 or not (board_copy[0] == 0).any():
                    break
                
                free = np.where(board_copy[0] == 0)[0]
                if len(free) == 0:
                    break
                
                # EXPANSIÓN: Si el nodo tiene acciones no probadas, expandir una
                if node.untried_actions:
                    # Tomar una acción no probada (pop la remueve de la lista)
                    action = node.untried_actions.pop()
                    
                    # Crear el estado resultante de esa acción
                    new_board = make_move_numpy(board_copy, action, player)
                    
                    # Crear nuevo nodo hijo
                    child = Node(new_board, -player, parent=node, parent_action=action)
                    node.children[action] = child
                    
                    # Movernos al hijo recién creado
                    node = child
                    path.append(node)
                    board_copy = new_board
                    player = -player
                    break  # Terminamos selección, vamos a simulación
                
                # SELECCIÓN: Si está completamente expandido, aplicar UCB1
                if not node.children:
                    break  # No hay hijos, terminamos
                
                # -----------------------------------------------
                # Fórmula UCB1 (Upper Confidence Bound 1):
                # -----------------------------------------------
                # UCB1(child) = Q(child) + c * sqrt(ln(N(parent)) / N(child))
                #
                # Donde:
                # - Q(child) = wins/visits = tasa de victoria promedio (EXPLOTACIÓN)
                # - c * sqrt(...) = término de exploración (EXPLORACIÓN)
                # - N(parent) = visitas del padre
                # - N(child) = visitas del hijo
                #
                # El término de exploración crece cuando:
                # - El padre tiene muchas visitas (ln(N) grande)
                # - El hijo tiene pocas visitas (1/N grande)
                #
                # Esto incentiva visitar nodos menos explorados mientras
                # también favorece nodos con buenas tasas de victoria.
                # -----------------------------------------------
                
                best_score = -float("inf")
                best_child = None
                best_action = None
                parent_log = math.log(node.visits + 1)  # Pre-calcular para eficiencia
                
                for action, child in node.children.items():
                    # Priorizar hijos nunca visitados (exploración pura)
                    if child.visits == 0:
                        best_child = child
                        best_action = action
                        break
                    
                    # Calcular UCB1
                    q = child.wins / child.visits  # Tasa de victoria (entre 0 y 1)
                    exploration = self.c * math.sqrt(parent_log / child.visits)
                    score = q + exploration
                    
                    if score > best_score:
                        best_score = score
                        best_child = child
                        best_action = action
                
                if best_child is None:
                    break
                
                # Movernos al mejor hijo según UCB1
                node = best_child
                path.append(node)
                board_copy = make_move_numpy(board_copy, best_action, player)
                player = -player
            
            # -----------------------------------------------
            # FASE 2: SIMULACIÓN (Playout/Rollout)
            # -----------------------------------------------
            # Desde el nodo actual, jugar aleatoriamente hasta el final
            # (o hasta max_depth para limitar tiempo)
            reward = fast_playout(board_copy, player, root_player, max_depth=10)
            
            # -----------------------------------------------
            # FASE 3: BACKPROPAGACIÓN
            # -----------------------------------------------
            # Propagar el resultado hacia arriba por todo el camino recorrido
            for n in path:
                n.visits += 1      # Incrementar contador de visitas
                n.wins += reward   # Acumular recompensa
                # Nota: La recompensa es siempre desde la perspectiva de root_player,
                # por eso no necesitamos invertir el signo al subir.
            
            simulations += 1
        
        # -----------------------------------------------
        # PASO 4: Selección final
        # -----------------------------------------------
        # Después de todas las simulaciones, elegir la acción más visitada.
        # La acción más visitada es generalmente más confiable que la acción
        # con mejor win rate, ya que tiene más datos estadísticos.
        
        if not root.children:
            # Caso raro: no se pudo expandir ningún hijo (tiempo muy limitado)
            # Fallback: usar heurística de columnas centrales
            preferred = [3, 2, 4, 1, 5, 0, 6]
            for col in preferred:
                if col in free_cols:
                    return int(col)
            return int(free_cols[0])
        
        # Elegir la acción (columna) cuyo hijo tiene más visitas
        best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
        return int(best_action)


# ========================
# FUNCIONES AUXILIARES NUMPY
# ========================
# Todas estas funciones trabajan directamente con numpy arrays
# para máxima velocidad, evitando crear objetos ConnectState.

def infer_player_fast(board: np.ndarray) -> int:
    """
    Infiere quién debe jugar basándose en el conteo de fichas.
    
    Convención:
    - Si #fichas_rojas <= #fichas_amarillas → turno de rojo (-1)
    - Si #fichas_rojas > #fichas_amarillas → turno de amarillo (1)
    
    Esto asume que rojo siempre juega primero.
    
    Parámetros
    ----------
    board : np.ndarray
        Tablero actual (6x7)
    
    Retorna
    -------
    int
        -1 para rojo, 1 para amarillo
    """
    neg = np.count_nonzero(board == -1)  # Contar fichas rojas
    pos = np.count_nonzero(board == 1)   # Contar fichas amarillas
    return -1 if neg <= pos else 1


def make_move_numpy(board: np.ndarray, col: int, player: int) -> np.ndarray:
    """
    Ejecuta una jugada directamente en numpy sin crear ConnectState.
    
    La ficha "cae" por gravedad hasta la posición más baja disponible
    en la columna especificada.
    
    Parámetros
    ----------
    board : np.ndarray
        Tablero actual (6x7)
    col : int
        Columna donde colocar la ficha (0-6)
    player : int
        Jugador que coloca la ficha (-1 o 1)
    
    Retorna
    -------
    np.ndarray
        Nuevo tablero con la jugada realizada
    """
    new_board = board.copy()
    # Encontrar todas las filas vacías en la columna
    row = np.where(new_board[:, col] == 0)[0]
    if len(row) > 0:
        # Colocar en la fila más baja (última del array filtrado)
        new_board[row[-1], col] = player
    return new_board


def can_win_immediate(board: np.ndarray, col: int, player: int) -> bool:
    """
    Verifica si jugar en una columna resulta en victoria inmediata.
    
    Usado para las heurísticas de ganar/bloquear antes de MCTS.
    
    Parámetros
    ----------
    board : np.ndarray
        Tablero actual
    col : int
        Columna a probar
    player : int
        Jugador que haría la jugada
    
    Retorna
    -------
    bool
        True si esa jugada gana el juego
    """
    # Verificar que la columna tenga espacio
    row_idx = np.where(board[:, col] == 0)[0]
    if len(row_idx) == 0:
        return False
    
    # Simular la jugada
    row = row_idx[-1]
    test_board = board.copy()
    test_board[row, col] = player
    
    # Verificar si hay victoria
    return check_winner_numpy(test_board) == player


def check_winner_numpy(board: np.ndarray) -> int:
    """
    Verifica si hay un ganador en el tablero usando operaciones numpy puras.
    
    Esta es una versión ultra-optimizada que evita crear objetos ConnectState.
    Verifica las 4 direcciones posibles: horizontal, vertical, y ambas diagonales.
    
    Parámetros
    ----------
    board : np.ndarray
        Tablero a verificar (6x7)
    
    Retorna
    -------
    int
        -1 si gana rojo, 1 si gana amarillo, 0 si no hay ganador
    """
    # Verificar horizontal: 4 en línea en cada fila
    # Solo necesitamos verificar posiciones 0-3 porque a partir de ahí
    # no caben 4 fichas consecutivas
    for r in range(6):
        for c in range(4):  # columnas 0, 1, 2, 3
            val = board[r, c]
            if val != 0:
                # Verificar si las 4 posiciones consecutivas son iguales
                if board[r, c] == board[r, c+1] == board[r, c+2] == board[r, c+3]:
                    return val
    
    # Verificar vertical: 4 en línea en cada columna
    for r in range(3):  # filas 0, 1, 2 (desde arriba hay 4 espacios)
        for c in range(7):
            val = board[r, c]
            if val != 0:
                if board[r, c] == board[r+1, c] == board[r+2, c] == board[r+3, c]:
                    return val
    
    # Verificar diagonal \ (descendente de izquierda a derecha)
    for r in range(3):  # filas 0-2
        for c in range(4):  # columnas 0-3
            val = board[r, c]
            if val != 0:
                if board[r, c] == board[r+1, c+1] == board[r+2, c+2] == board[r+3, c+3]:
                    return val
    
    # Verificar diagonal / (ascendente de izquierda a derecha)
    for r in range(3, 6):  # filas 3-5 (desde abajo)
        for c in range(4):  # columnas 0-3
            val = board[r, c]
            if val != 0:
                if board[r, c] == board[r-1, c+1] == board[r-2, c+2] == board[r-3, c+3]:
                    return val
    
    return 0  # No hay ganador


def fast_playout(board: np.ndarray, player: int, root_player: int, max_depth: int = 10) -> float:
    """
    Ejecuta una simulación rápida (playout/rollout) desde el estado dado.
    
    Esta es la FASE DE SIMULACIÓN de MCTS. Juega aleatoriamente hasta:
    1. Encontrar un ganador
    2. Llenar el tablero (empate)
    3. Alcanzar max_depth movimientos
    
    La profundidad limitada es crucial para velocidad: evita simular
    partidas completas de 40+ movimientos.
    
    Parámetros
    ----------
    board : np.ndarray
        Tablero desde donde iniciar la simulación
    player : int
        Jugador que mueve primero en la simulación
    root_player : int
        Jugador desde cuya perspectiva calcular la recompensa
        (siempre es el jugador que movió en la raíz del árbol)
    max_depth : int
        Número máximo de movimientos a simular (default: 10)
    
    Retorna
    -------
    float
        Recompensa desde la perspectiva de root_player:
        - 1.0 si root_player ganó
        - 0.5 si fue empate
        - 0.0 si root_player perdió
    """
    current_board = board
    current_player = player
    
    # Simular hasta max_depth movimientos
    for _ in range(max_depth):
        # Verificar si hay ganador
        winner = check_winner_numpy(current_board)
        if winner != 0:
            return 1.0 if winner == root_player else 0.0
        
        # Verificar si el tablero está lleno (empate)
        free = np.where(current_board[0] == 0)[0]
        if len(free) == 0:
            return 0.5
        
        # Movimiento aleatorio: elegir columna libre al azar
        col = np.random.choice(free)
        current_board = make_move_numpy(current_board, col, current_player)
        
        # Alternar jugador
        current_player = -current_player
    
    # Si alcanzamos max_depth sin terminar, hacer evaluación final
    winner = check_winner_numpy(current_board)
    if winner == root_player:
        return 1.0
    elif winner == -root_player:
        return 0.0
    return 0.5  # Asumir empate si no terminó