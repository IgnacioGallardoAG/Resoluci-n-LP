import math
import random
import statistics

import matplotlib.pyplot as plt

# ========================
# DOMINIOS Y RESTRICCIONES AC-3
# ========================

domains = {
    'A': list(range(0, 16)),   # TV tarde
    'B': list(range(0, 11)),   # TV noche
    'C': list(range(0, 26)),   # Diario
    'D': list(range(0, 5)),    # Revista
    'E': list(range(0, 31))    # Radio
}

constraints = {
    ('A', 'B'): lambda a, b: 180*a + 325*b <= 3800,
    ('B', 'A'): lambda b, a: 180*a + 325*b <= 3800,
    ('C', 'D'): lambda c, d: 60*c + 110*d <= 2800,
    ('D', 'C'): lambda d, c: 60*c + 110*d <= 2800,
    ('C', 'E'): lambda c, e: 60*c + 15*e <= 3500,
    ('E', 'C'): lambda e, c: 60*c + 15*e <= 3500,
}

def revise(x, y):
    revised = False
    x_domain = domains[x][:]
    y_domain = domains[y]
    all_constraints = [c for c in constraints if c[0] == x and c[1] == y]
    for x_val in x_domain:
        satisfies = any(
            constraints[c](x_val, y_val)
            for y_val in y_domain
            for c in all_constraints
        )
        if not satisfies:
            domains[x].remove(x_val)
            revised = True
    return revised

def ac3(arcs):
    queue = arcs[:]
    while queue:
        (x, y) = queue.pop(0)
        if revise(x, y):
            neighbors = [n for n in arcs if n[1] == x and n[0] != y]
            queue += neighbors

ac3(list(constraints.keys()))

print("Dominios luego de aplicar AC-3:")
for var, dom in domains.items():
    print(f"{var}: {dom}")
print("="*40)


# ========================
# FUNCIONES DE PARETO
# ========================

def is_dominated(f1, f2):
    return all(a <= b for a, b in zip(f1, f2)) and any(a < b for a, b in zip(f1, f2))

def update_pareto_front(front, candidate):
    new_fitness = candidate.fitness()
    non_dominated = []
    dominated = False
    for ind in front:
        if is_dominated(new_fitness, ind.fitness()):
            dominated = True
            break
        elif not is_dominated(ind.fitness(), new_fitness):
            non_dominated.append(ind)
    if not dominated:
        non_dominated.append(candidate)
    return non_dominated

# ========================
# CLASES PARA APO
# ========================

class Problem:
    def __init__(self):
        self.vars = ['A', 'B', 'C', 'D', 'E']
        self.dim = len(self.vars)
        self.domains = {v: domains[v] for v in self.vars}

    def check(self, x):
        A, B, C, D, E = x
        return (
            180*A + 325*B <= 3800 and
            60*C + 110*D <= 2800 and
            60*C + 15*E <= 3500
        )

    def fit(self, x):
        alcance = 1000*x[0] + 2000*x[1] + 1500*x[2] + 2500*x[3] + 300*x[4]
        costo = 180*x[0] + 325*x[1] + 60*x[2] + 110*x[3] + 15*x[4]
        return (alcance, -costo)

    def random_value(self, var):
        return random.choice(self.domains[var])

class Individual:
    def __init__(self):
        self.p = Problem()
        self.dimension = self.p.dim
        self.x = self.generate_valid_solution()

    def generate_valid_solution(self):
        while True:
            sol = [self.p.random_value(var) for var in self.p.vars]
            if self.p.check(sol):
                return sol

    def is_feasible(self):
        return self.p.check(self.x)

    def fitness(self):
        return self.p.fit(self.x)

    def is_better_than(self, g):
        f1, c1 = self.fitness()
        f2, c2 = g.fitness()
        return f1 > f2 or (f1 == f2 and c1 > c2)

    def move(self, g, t, max_iter):
        for j in range(self.dimension):
            alpha = random.uniform(-1, 1) * math.exp(-2 * t / max_iter)
            new_val = round(self.x[j] + alpha * (g.x[j] - self.x[j]))
            var_name = self.p.vars[j]
            new_val = min(max(new_val, min(self.p.domains[var_name])), max(self.p.domains[var_name]))
            self.x[j] = new_val
        while not self.is_feasible():
            self.x = self.generate_valid_solution()

    def copy(self, other):
        self.x = other.x.copy()

    def __str__(self):
        f1, c1 = self.fitness()
        return f"x: {self.x}, alcance: {f1}, costo: {-c1}"

class PuffinSwarm:
    def __init__(self):
        self.max_iter = 25
        self.n_individual = 10
        self.swarm = []
        self.g = None
        self.front = []
        self.convergence_data = []  # ← agregado

    def initialize(self):
        for _ in range(self.n_individual):
            ind = Individual()
            self.swarm.append(ind)
            self.front = update_pareto_front(self.front, ind)

        self.g = self.swarm[0]
        for ind in self.swarm[1:]:
            if ind.is_better_than(self.g):
                self.g.copy(ind)
        self.show_results(0)

    def evolve(self):
        for t in range(1, self.max_iter + 1):
            for i in range(1, self.n_individual):
                self.swarm[i].move(self.g, t, self.max_iter)
                if self.swarm[i].is_better_than(self.g):
                    self.g.copy(self.swarm[i])
                self.front = update_pareto_front(self.front, self.swarm[i])
            self.show_results(t)

    def show_results(self, t):
        print(f"Iteración {t}: Mejor individuo → {self.g}")
        alcance, costo = self.g.fitness()
        self.convergence_data.append((t, alcance, -costo))  # ← guardamos valores

    def summary_table(self):
        pareto_fitness = [ind.fitness() for ind in self.front]
        alcances = [a for a, _ in pareto_fitness]
        costos = [-c for _, c in pareto_fitness]  # invertir porque están negativos

        print("\nResumen descriptivo del Frente de Pareto:")
        print(f"{'Métrica':<10} | {'Alcance':<10} | {'Costo':<10}")
        print("-" * 35)
        print(f"{'Mínimo':<10} | {min(alcances):<10} | {min(costos):<10}")
        print(f"{'Máximo':<10} | {max(alcances):<10} | {max(costos):<10}")
        print(f"{'Promedio':<10} | {statistics.mean(alcances):<10.2f} | {statistics.mean(costos):<10.2f}")
        print(f"{'Mediana':<10} | {statistics.median(alcances):<10.2f} | {statistics.median(costos):<10.2f}")

    def plot_convergence(self):
        iteraciones = [i for i, _, _ in self.convergence_data]
        alcances = [a for _, a, _ in self.convergence_data]
        costos = [c for _, _, c in self.convergence_data]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(iteraciones, alcances, marker='o', color='green')
        plt.title('Convergencia del Alcance')
        plt.xlabel('Iteración')
        plt.ylabel('Alcance')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(iteraciones, costos, marker='o', color='red')
        plt.title('Convergencia del Costo')
        plt.xlabel('Iteración')
        plt.ylabel('Costo')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def optimizer(self):
        self.initialize()
        self.evolve()

        print("\nFrente de Pareto final:")
        for ind in self.front:
            print(ind)

        self.summary_table()

        pareto_front = [ind.fitness() for ind in self.front]
        costos = [-c for _, c in pareto_front]
        alcances = [a for a, _ in pareto_front]

        plt.figure(figsize=(8, 6))
        plt.scatter(costos, alcances, color='blue', label='Soluciones Pareto')
        plt.title('Frente de Pareto (Dispersión)')
        plt.xlabel('Costo Total')
        plt.ylabel('Alcance Total')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

        self.plot_convergence()

# ========================
# EJECUCIÓN
# ========================

PuffinSwarm().optimizer()
