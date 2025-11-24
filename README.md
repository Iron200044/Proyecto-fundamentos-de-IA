# Proyecto – Fundamentos de Inteligencia Artificial  
## Sistema de Agentes para Connect-4 (Heurístico, Q-Learning y MCTS)

Este repositorio contiene el desarrollo completo de cuatro agentes distintos para jugar Connect-4 utilizando enfoques de Inteligencia Artificial basados en heurísticas, aprendizaje por refuerzo y búsqueda Monte Carlo Tree Search (MCTS).  
El objetivo del proyecto fue diseñar, entrenar, evaluar y comparar varios agentes para seleccionar el más competitivo de cara al torneo final del curso.

---

# Agente A – Aha (Heuristic Connect-4 Agent)

El Agente A implementa una política heurística diseñada para ganar de manera determinista contra jugadores aleatorios. No requiere entrenamiento y fue creado como baseline confiable para validación y comparación.

### Estrategia
1. Ganar si existe una jugada inmediata.
2. Bloquear si el oponente puede ganar en su turno siguiente.
3. Priorizar columnas centrales: 3, 2, 4, 1, 5, 0, 6.
4. Seleccionar la primera jugada válida si no se aplica ninguna regla previa.

### Motivos del diseño
- Es simple y extremadamente rápido.
- No necesita entrenamiento.
- Garantiza una tasa de victoria superior al 99 % contra oponentes aleatorios.
- Funciona como baseline estable para evaluar agentes más avanzados.

---

# Agente B – Q-Learning

El Agente B implementa aprendizaje por refuerzo mediante Q-Learning tabular. Aprende jugando miles de partidas simuladas contra distintos rivales.

### Características
- Representación canónica del estado que reduce duplicados.
- Entrenamiento contra:
  - Oponente aleatorio
  - Agente heurístico Aha
  - Jugando como rojo y como amarillo
- Política epsilon-greedy con decaimiento.
- Carga automática de la Q-table en torneo.

### Resultados
- Alta tasa de victoria contra agentes débiles.
- Buen desempeño contra Aha.
- Rendimiento muy bajo contra agentes basados en MCTS.
- Limitado por el tamaño del espacio de estados y la cantidad de entrenamiento posible antes de superar el tamaño razonable del archivo `.pkl`.

### Limitaciones
1. La Q-table crece muy rápido por la enorme cantidad de estados posibles.
2. El aprendizaje depende completamente de los oponentes de entrenamiento.
3. No aprende tácticas profundas ni estrategias a largo plazo.
4. No realiza búsqueda ni simulación.
5. Su rendimiento solo es competitivo frente a agentes predecibles o simples.
6. El archivo `.pkl` generado contiene un entrenamiento limitado debido a la necesidad de mantener un tamaño razonable para ser usado en Gradescope.

---

# Agentes C y D – Monte Carlo Tree Search (MCTS)

Los agentes C y D implementan búsqueda por simulación basada en MCTS.

### Agente C
Primera versión funcional del MCTS, utilizada como base para pruebas de eficiencia.

### Agente D (final)
Incluye:
- Optimización de tiempo por jugada.
- Playouts rápidos.
- Selección UCB1 eficiente.
- Reducción del árbol de búsqueda.
- Control de tiempo compatible con Gradescope.

Este agente superó ampliamente a los demás y fue seleccionado como agente final del torneo.

---

# Notebook de Análisis – entrega.ipynb

El notebook incluye:
- Evaluación sistemática de winrate.
- Comparación entre todos los agentes.
- Análisis del comportamiento del Q-learning.
- Pruebas contras random, heurístico y agentes MCTS.
- Conclusiones basadas en resultados y experimentación.

---

# Conclusiones Generales

- La correcta exploración durante el entrenamiento es fundamental para que un agente aprenda políticas útiles.
- El Q-learning tabular funciona bien frente a agentes simples, pero no escala adecuadamente al nivel requerido para Connect-4.
- Los agentes MCTS fueron significativamente superiores porque evalúan múltiples posibilidades antes de actuar.
- El tiempo de ejecución y la optimización fueron aspectos tan importantes como la técnica de IA utilizada.
- El análisis experimental permitió seleccionar el agente más competitivo de manera objetiva.

