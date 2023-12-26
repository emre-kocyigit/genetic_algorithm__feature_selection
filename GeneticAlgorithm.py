import numpy as np
import random
import matplotlib.pyplot as plt
import MachineLearning as ml

# default parameters
latest_generation_value = 50  # termination criterion, evolutionary progress will be completed with this generation
population_size = 20  # total number of solutions in one generation
chromosome_length = 23  # total number of features
tournament_size = 4  # for selection
mutation_probability = 0.1  # the probability of selected to be mutated while creating a generation
mutation_rate = 0.1  # when mutation of individual, 10% of the genes in a given chromosome will be mutated
elitism_value = 1


# solutions will be structured as Chromosome object which has attributes like fitness value
class Chromosome:
    def __init__(self, sequence=None, length=chromosome_length):
        if sequence is not None:
            self.chromosome = sequence
        else:
            self.chromosome = np.random.randint(2, size=length, dtype=np.uint8)  # create a random chromosome with 1 and 0s.
        self.fitness_value = ml.get_fitness_value(self.chromosome)
        self.id = tuple(np.packbits(self.chromosome))  # generate a unique id based on the chromosome info


# THE FIRST POPULATION
def create_initial_population(pop_size=population_size):
    initial_population = [Chromosome() for _ in range(pop_size)]
    # duplicate check. if there is same chromosome, change it.
    ids = set()
    return [sol if (key := sol.id) not in ids and not ids.add(key) else Chromosome() for sol in initial_population]


# SELECTION
def tournament_select(population, t_size=tournament_size):
    # randomly select chromosomes as much as tournament size
    tournament_participants = random.sample(population, t_size)

    # compare the fitness values of the participants of the tournament and choose the best(maximum)
    winner = max(tournament_participants, key=lambda x: x.fitness_value)
    return winner


# CROSSOVER
# NOTE: I am passing Chromosome objects as parameters to increase the readability. However, it's better to use
# 'chromosome' attributes of these objects instead of passing entire objects in terms of flexibility and reducing coupling.
# one point crossover
def crossover_one_point(parent_1, parent_2):
    # generate a random index between 1 and chromosome_length-1
    break_point = np.random.randint(1, len(parent_1.chromosome) - 1)
    child = np.concatenate((parent_1.chromosome[:break_point], parent_2.chromosome[break_point:]))
    return Chromosome(sequence=child)


# two points crossover
def crossover_two_points(parent_1, parent_2):
    break_points = sorted(random.sample(range(1, len(parent_1.chromosome) - 1), 2))
    child = np.concatenate((parent_1.chromosome[:break_points[0]], parent_2.chromosome[break_points[0]:break_points[1]], parent_1.chromosome[break_points[1]:]))
    return Chromosome(sequence=child)


# MUTATION
# bit flip mutation
def mutate_bit_flip(child, rate=mutation_rate):
    mutation_number = np.random.randint(1, int(len(child.chromosome)*rate))
    mutation_points = np.random.randint(0, len(child.chromosome), mutation_number)
    mutated_chromosome = np.array([gene ^ 1 if i in mutation_points else gene for i, gene in enumerate(child.chromosome)])
    return Chromosome(sequence=mutated_chromosome)


# ELITISM
# find the best chromosome in a given generation based on the fitness value
def find_best_chromosome(generation):
    return max(generation, key=lambda x: x.fitness_value)


# if you want to have more flexible elitism and change the number of directly added chromosomes
def find_best_k_chromosomes(generation, k=elitism_value):
    sorted_population = sorted(generation, key=lambda x: x.fitness_value, reverse=True)
    return sorted_population[:k]


# NEW GENERATION CREATION
def create_generation(previous_generation, m_prob=mutation_probability):
    # elitism: directly add the best chromosome(s) of the previous generation to the next generation
    next_generation = find_best_k_chromosomes(previous_generation)  # next generation is a list

    # apply crossover and create new chromosomes for the next generation
    while len(next_generation) < len(previous_generation):  # keep the same size with previous generations, i.e. pop_size
        parent1 = tournament_select(previous_generation)  # select parents
        parent2 = tournament_select(previous_generation)

        while parent1.id == parent2.id:  # parent 1 and parent 2 should not be same
            parent2 = tournament_select(previous_generation)
        child = crossover_one_point(parent1, parent2)
        if m_prob > random.random():
            child = mutate_bit_flip(child)
        next_generation.append(child)

    return next_generation


# COMPLETE GENETIC ALGORITHM FOR FEATURE SELECTION
def find_optimal_feature_set(termination_criterion=latest_generation_value):
    generation = create_initial_population()  # initial population
    evolutionary_progress = [find_best_chromosome(generation)]  # add best of each gen to this list to plot the progress
    for i in range(0, termination_criterion):
        print(f"\rGeneration {i + 1}: {'*' * (i + 1)}", end='', flush=True)  # just to display the progress
        generation = create_generation(generation)
        evolutionary_progress.append(find_best_chromosome(generation))
    return evolutionary_progress  # return the best chromosome of the last generation - best of all


# best feature selection results of each generation
best_chromosomes = find_optimal_feature_set()
best_chromosome = best_chromosomes[-1]  # best of best

# a list to plot the evolutionary progress of the GA
best_chromosomes_fitness_values = [chromosome.fitness_value for chromosome in best_chromosomes]

# plot
plt.plot(best_chromosomes_fitness_values)
# labels and title
plt.xlabel('Generations')
plt.ylabel('Fitness Values')
plt.title('Evolutionary Progress of the Chromosomes')

# show the plot
plt.show()
