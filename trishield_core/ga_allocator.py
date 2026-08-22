import random
import numpy as np


class HeterogeneousGA:
    def __init__(
        self,
        blackboard,
        population_size: int = 40,
        generations: int = 100,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        elite_count: int = 2,
        patience: int = 15,
        seed: int = None,
    ):
        self.bb = blackboard
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.patience = patience
        self.rng = random.Random(seed)

        # Diagnostics populated after each allocate() call, useful for the
        # dissertation's convergence-behaviour discussion and for logging.
        self.last_run_stats = {}

    # ------------------------------------------------------------------ #
    # Per-(agent, task) fitness — UNCHANGED from the original allocator so
    # that reported cost characteristics remain comparable.
    # ------------------------------------------------------------------ #
    def calculate_fitness(self, agent, task_pos, task_type, urgency=1.0):
        if agent.mission_status in ['DEAD', 'RTB']:
            return float('inf')

        distance = np.linalg.norm(np.array(agent.pos) - np.array(task_pos))
        eta = distance / agent.max_speed

        fitness = eta
        battery_penalty = 1.0 + (100.0 - agent.battery) / 50.0
        urgency_factor = max(0.1, 11 - urgency) / 10.0  # 10 = high urgency -> 0.1 factor

        # Task-type dependent multiplier: Rooftop rescues require vertical altitude ascension
        # (33m elevation), incurring a 1.25x energy/climb complexity weighting over ground rescues.
        type_factor = 1.25 if task_type == 'rooftop_rescue' else 1.0

        return fitness * battery_penalty * urgency_factor * type_factor

    # ------------------------------------------------------------------ #
    # Decoding: chromosome (task priority order) -> {agent_id: [task_ids]}
    # ------------------------------------------------------------------ #
    UNASSIGNED_PENALTY = 500.0  # large fixed cost for a task that gets no agent

    def _decode(self, chromosome, tasks, agents):
        """
        chromosome: list of task indices (a permutation of range(len(tasks)))
        tasks: list of (task_id, task_pos, task_type, urgency)
        agents: list of Agent objects
        Returns (agent_queues dict {agent_id: [task_ids]}, total_fitness float)
        """
        agent_queues = {agent.id: [] for agent in agents}
        assigned_agent_ids_in_round = set()
        total_fitness = 0.0

        for task_idx in chromosome:
            if task_idx >= len(tasks):
                continue
            task_id, task_pos, task_type, urgency = tasks[task_idx]

            # When all agents have been assigned in current round, reset for next round
            if len(assigned_agent_ids_in_round) >= len(agents):
                assigned_agent_ids_in_round.clear()

            best_agent = None
            best_fitness = float('inf')
            for agent in agents:
                if agent.id in assigned_agent_ids_in_round:
                    continue
                fitness = self.calculate_fitness(agent, task_pos, task_type, urgency)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_agent = agent

            if best_agent is not None and best_fitness != float('inf'):
                agent_queues[best_agent.id].append(task_id)
                assigned_agent_ids_in_round.add(best_agent.id)
                total_fitness += best_fitness
            else:
                total_fitness += self.UNASSIGNED_PENALTY

        return agent_queues, total_fitness

    # ------------------------------------------------------------------ #
    # GA operators
    # ------------------------------------------------------------------ #
    def _random_chromosome(self, n_tasks):
        chromosome = list(range(n_tasks))
        self.rng.shuffle(chromosome)
        return chromosome

    def _tournament_select(self, population, fitnesses):
        contenders = self.rng.sample(range(len(population)), self.tournament_size)
        best_idx = min(contenders, key=lambda i: fitnesses[i])
        return population[best_idx][:]

    def _pmx_crossover(self, parent_a, parent_b):
        """Partially Matched Crossover — produces one permutation-valid child."""
        size = len(parent_a)
        if size < 2:
            return parent_a[:]

        p1, p2 = sorted(self.rng.sample(range(size), 2))
        child = [None] * size
        child[p1:p2] = parent_a[p1:p2]

        mapping = {parent_a[i]: parent_b[i] for i in range(p1, p2)}

        for i in range(size):
            if p1 <= i < p2:
                continue
            candidate = parent_b[i]
            while candidate in child[p1:p2]:
                candidate = mapping.get(candidate, candidate)
            child[i] = candidate

        return child

    def _swap_mutate(self, chromosome):
        chromosome = chromosome[:]
        for i in range(len(chromosome)):
            if self.rng.random() < self.mutation_rate:
                j = self.rng.randrange(len(chromosome))
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

    # ------------------------------------------------------------------ #
    # Main entry point — allocate() -> dict {agent_id: [task_ids]}
    # ------------------------------------------------------------------ #
    def allocate(self):
        import os
        import time
        start_time = time.perf_counter()
        variant = os.environ.get("TRISAR_VARIANT", "full")

        # Baseline check: If variant disables GA (no_ga or floor), perform Greedy Nearest/Fitness Allocation
        if variant in ["no_ga", "floor"]:
            relay_ids = {'UAV_4', 'UAV_5'}
            agents = [a for a in self.bb.agent_states.values() if getattr(a, 'id', '') not in relay_ids]
            if not agents:
                agents = list(self.bb.agent_states.values())
            agent_queues = {agent.id: [] for agent in agents}
            if not agents:

                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                self.last_run_stats = {
                    'allocator_type': 'greedy',
                    'ga_allocation_time_ms': round(elapsed_ms, 3),
                    'generations_run': 0,
                }
                return agent_queues

            tasks = []
            for tid, t in self.bb.threats.items():
                task_type = 'rooftop_rescue' if t['pos'][2] > 2 else 'ground_rescue'
                tasks.append((tid, t['pos'], task_type, t.get('urgency', 5)))
            for vid, v in self.bb.victims.items():
                task_type = 'rooftop_rescue' if v['pos'][2] > 2 else 'ground_rescue'
                tasks.append((vid, v['pos'], task_type, v.get('urgency', 8)))

            assigned_agent_ids_in_round = set()
            tasks_sorted = sorted(tasks, key=lambda x: x[3], reverse=True)

            for tid, tpos, ttype, urgency in tasks_sorted:
                if len(assigned_agent_ids_in_round) >= len(agents):
                    assigned_agent_ids_in_round.clear()

                best_agent = None
                best_fitness = float('inf')
                for agent in agents:
                    if agent.id in assigned_agent_ids_in_round:
                        continue
                    fit = self.calculate_fitness(agent, tpos, ttype, urgency)
                    if fit < best_fitness:
                        best_fitness = fit
                        best_agent = agent
                if best_agent is not None:
                    agent_queues[best_agent.id].append(tid)
                    assigned_agent_ids_in_round.add(best_agent.id)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.last_run_stats = {
                'allocator_type': 'greedy',
                'ga_allocation_time_ms': round(elapsed_ms, 3),
                'generations_run': 0,
            }
            return agent_queues

        relay_ids = {'UAV_4', 'UAV_5'}
        agents = [a for a in self.bb.agent_states.values() if getattr(a, 'id', '') not in relay_ids]
        if not agents:
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
            return {agent.id: [] for agent in agents}

        n_tasks = len(tasks)

        if n_tasks == 1:
            agent_queues, best_fitness = self._decode([0], tasks, agents)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.last_run_stats = {
                'allocator_type': 'ga',
                'ga_allocation_time_ms': round(elapsed_ms, 3),
                'generations_run': 0, 'best_fitness': best_fitness, 'converged_at': 0
            }
            return agent_queues

        # Initialise population
        population = [self._random_chromosome(n_tasks) for _ in range(self.population_size)]

        best_chromosome = None
        best_fitness = float('inf')
        best_queues = {}
        generations_since_improvement = 0
        converged_at = self.generations

        for gen in range(self.generations):
            decoded = [self._decode(c, tasks, agents) for c in population]
            fitnesses = [f for (_, f) in decoded]

            gen_best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_chromosome = population[gen_best_idx][:]
                best_queues = decoded[gen_best_idx][0]
                generations_since_improvement = 0
            else:
                generations_since_improvement += 1

            if generations_since_improvement >= self.patience:
                converged_at = gen
                break

            # Elitism
            elite_idx = sorted(range(len(population)), key=lambda i: fitnesses[i])[:self.elite_count]
            new_population = [population[i][:] for i in elite_idx]

            # Reproduce
            while len(new_population) < self.population_size:
                parent_a = self._tournament_select(population, fitnesses)
                parent_b = self._tournament_select(population, fitnesses)

                if self.rng.random() < self.crossover_rate:
                    child = self._pmx_crossover(parent_a, parent_b)
                else:
                    child = parent_a[:]

                child = self._swap_mutate(child)
                new_population.append(child)

            population = new_population

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_run_stats = {
            'allocator_type': 'ga',
            'ga_allocation_time_ms': round(elapsed_ms, 3),
            'generations_run': converged_at + 1,
            'best_fitness': best_fitness,
            'converged_at': converged_at,
            'population_size': self.population_size,
        }

        return best_queues

