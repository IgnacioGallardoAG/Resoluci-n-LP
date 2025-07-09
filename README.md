# Optimización de Campañas Publicitarias con Restricciones

Este proyecto resuelve un problema de optimización multiobjetivo donde se busca **asignar recursos publicitarios** en distintos medios (TV, diario, revista y radio), **maximizando la calidad** y **minimizando el costo**, cumpliendo con restricciones de presupuesto para cada combinación de medios.

El enunciado establece que existen combinaciones posibles de cantidad de anuncios por canal, cada uno con un alcance y costo distinto, pero sujeto a **restricciones presupuestarias combinadas** entre ellos.

---

##  Tecnologías utilizadas

- **Python 3.12**
- **Algoritmo de Consistencia AC-3:** Reduce los valores posibles de cada variable aplicando restricciones entre pares.
- **Algoritmo Artic Puffin Optimization (APO):** Método metaheurístico inspirado en aves árticas, utilizado para explorar soluciones válidas en el espacio multiobjetivo.
- **Matplotlib:** Se usa para graficar el frente de Pareto final (calidad vs costo).

---

## ¿Cómo ejecutar el programa?

1. Asegúrate de tener Python 3.10 o superior.
2. Clona o descarga este repositorio.
3. Instala la única dependencia:
   ```bash
   pip install matplotlib
