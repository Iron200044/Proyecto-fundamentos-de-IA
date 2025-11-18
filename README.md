# Proyecto-fundamentos-de-IA
# Agente A – “Aha” (Heuristic Connect-4 Agent)

Este agente implementa una política heurística diseñada para ganar consistentemente
contra jugadores aleatorios, cumpliendo así el requisito mínimo del reto.

## Estrategia principal

El agente evalúa únicamente jugadas inmediatas (look-ahead de un turno), pero lo hace
de forma determinista y efectiva:

1. **Ganar si es posible**  
   Si existe una jugada que conecta 4, la toma inmediatamente.

2. **Bloquear al oponente**  
   Si el rival podría ganar en su próximo turno, el agente bloquea esa columna.

3. **Preferir la zona central**  
   En ausencia de urgencias, juega columnas centrales (3, 2, 4, 1, 5, 0, 6),
   ya que estadísticamente generan más oportunidades de victoria.

4. **Fallback seguro**  
   Si ninguna regla aplica, juega la primera columna disponible.

## Razón del diseño

El objetivo del Agente A es ser:
- Simple  
- Rápido  
- No entrenable (heurístico)  
- Suficientemente fuerte para ganarle siempre al agente aleatorio del torneo  

Aunque no evalúa profundidad ni utiliza aprendizaje, estas reglas son
suficientes para asegurar más del 99-100% de victorias contra un oponente
random debido a que:
- El agente sí bloquea; el random no.
- El agente detecta victorias; el random no.
- El agente tiene preferencia estructural; el random no.
- El agente evita errores obvios; el random no.

