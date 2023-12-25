import numpy as np
import random
import MachineLearning as ml

# default parameters
chromosome_length = 23  # total number of features
tournament_size = 4  # for selection


class Chromosome:
    def __init__(self, length=chromosome_length):
        self.chromosome = np.random.randint(2, size=length, dtype=np.uint8)  # create a random chromosome with 1 and 0s.
        self.fitness_value = ml.get_fitness_value(self.chromosome)
        self.id = tuple(np.packbits(self.chromosome))  # generate a unique id based on the chromosome info


def create_initial_population(pop_size):
    initial_population = [Chromosome() for _ in range(pop_size)]
    # duplicate check. if there is same chromosome, change it.
    ids = set()
    return [sol if (key := sol.id) not in ids and not ids.add(key) else Chromosome() for sol in initial_population]


init_pop = create_initial_population(20)


def tournament_select(population, t_size=tournament_size):
    # randomly select chromosomes as much as tournament size
    tournament_participants = random.sample(population, t_size)

    # compare the fitness values of the participants of the tournament and choose the best(maximum)
    winner = max(tournament_participants, key=lambda x: x.fitness_value)
    return winner


for i in range(0, len(init_pop)):
    print(init_pop[i].chromosome)
    print(init_pop[i].fitness_value)
    print(100 * '_')

parent_1 = tournament_select(init_pop)
print(parent_1.chromosome)
print(parent_1.fitness_value)

