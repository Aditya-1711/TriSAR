import os
import time
import math
import numpy as np

class HeterogeneousGA:
    """Genetic Algorithm & Greedy Task Allocator for TriSAR Multi-UAV Swarm."""

    def __init__(self, blackboard, population_size=40, max_generations=100,
                 crossover_rate=0.8, mutation_rate=0.15, seed=None):
        self.bb = blackboard
        self.pop_size = population_size
        self.max_gens = max_generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.rng = np.random.default_rng(seed)
        self.last_run_stats = {}

    def allocate(self) -> dict:
        start_time = time.perf_counter()
        variant = os.environ.get("TRISAR_VARIANT", "full")

        if variant in ["no_ga", "floor"]:
            assignments = {}
            unassigned_victim_ids = list(self.bb.victims.keys())
            assigned_agent_ids = set()

            for victim_id in unassigned_victim_ids:
                v = self.bb.victims[victim_id]
                v_pos = np.array(v['pos'], dtype=float)
                best_agent = None
                best_dist = float('inf')

                for agent_id, agent in self.bb.agent_states.items():
                    if agent_id in assigned_agent_ids:
                        continue
                    a_pos = np.array(agent.pos, dtype=float)
                    dist = float(np.linalg.norm(a_pos - v_pos))
                    if dist < best_dist:
                        best_dist = dist
                        best_agent = agent

                if best_agent is not None:
                    assignments[victim_id] = best_agent.id
                    assigned_agent_ids.add(best_agent.id)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.last_run_stats = {
                'allocator_type': 'greedy',
                'ga_allocation_time_ms': round(elapsed_ms, 3),
                'generations_run': 0,
            }
            return assignments

        agents = list(self.bb.agent_states.values())
        tasks = []
        for tid, t in self.bb.threats.items():
            task_type = 'rooftop_rescue' if t['pos'][2] > 2 else 'ground_rescue'
            tasks.append((tid, t['pos'], task_type, t.get('urgency', 5)))
        for vid, v in self.bb.victims.items():
            task_type = 'rooftop_rescue' if v['pos'][2] > 2 else 'ground_rescue'
            tasks.append((vid, v['pos'], task_type, v.get('urgency', 8)))

        if not tasks or not agents:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.last_run_stats = {
                'allocator_type': 'ga',
                'ga_allocation_time_ms': round(elapsed_ms, 3),
                'generations_run': 0, 'best_fitness': None, 'converged_at': None
            }
            return {}

        n_tasks = len(tasks)

        if n_tasks == 1:
            assignments, best_fitness = self._decode([0], tasks, agents)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.last_run_stats = {
                'allocator_type': 'ga',
                'ga_allocation_time_ms': round(elapsed_ms, 3),
                'generations_run': 0, 'best_fitness': best_fitness, 'converged_at': 0
            }
            return assignments

        n_agents = len(agents)
        population = [self.rng.integers(0, n_agents, size=n_tasks).tolist() for _ in range(self.pop_size)]

        best_chrom = None
        best_fitness = float('-inf')
        gens_without_improvement = 0
        converged_at = self.max_gens
        actual_generations = 0

        for gen in range(self.max_gens):
            actual_generations = gen + 1
            fitnesses = []
            for chrom in population:
                _, fit = self._decode(chrom, tasks, agents)
                fitnesses.append(fit)
                if fit > best_fitness:
                    best_fitness = fit
                    best_chrom = chrom[:]
                    gens_without_improvement = 0
                    converged_at = gen
                else:
                    gens_without_improvement += 1

            if gens_without_improvement >= 15:
                break

            selected = self._tournament_selection(population, fitnesses)
            next_gen = []

            for i in range(0, self.pop_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % self.pop_size]

                if self.rng.random() < self.cx_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]

                child1 = self._mutate(child1, n_agents)
                child2 = self._mutate(child2, n_agents)

                next_gen.extend([child1, child2])

            population = next_gen[:self.pop_size]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_run_stats = {
            'allocator_type': 'ga',
            'ga_allocation_time_ms': round(elapsed_ms, 3),
            'generations_run': actual_generations,
            'best_fitness': round(best_fitness, 4) if best_fitness != float('-inf') else None,
            'converged_at': converged_at
        }

        if best_chrom is None:
            return {}

        final_assignments, _ = self._decode(best_chrom, tasks, agents)
        return final_assignments

    def _decode(self, chrom, tasks, agents):
        assignments = {}
        total_dist = 0.0

        for task_idx, agent_idx in enumerate(chrom):
            if task_idx >= len(tasks):
                break
            task_id, task_pos, _, _ = tasks[task_idx]
            agent = agents[agent_idx % len(agents)]

            dist = np.linalg.norm(np.array(agent.pos) - np.array(task_pos))
            total_dist += dist
            assignments[task_id] = agent.id

        fitness = -total_dist
        return assignments, fitness

    def _tournament_selection(self, population, fitnesses, k=3):
        selected = []
        n = len(population)
        for _ in range(n):
            idxs = self.rng.choice(n, size=k, replace=False)
            best_i = max(idxs, key=lambda i: fitnesses[i])
            selected.append(population[best_i][:])
        return selected

    def _crossover(self, parent1, parent2):
        if len(parent1) <= 1:
            return parent1[:], parent2[:]
        point = self.rng.integers(1, len(parent1))
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate(self, chrom, n_agents):
        mutated = chrom[:]
        for i in range(len(mutated)):
            if self.rng.random() < self.mut_rate:
                mutated[i] = int(self.rng.integers(0, n_agents))
        return mutated
