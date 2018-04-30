import random
import numpy as np
import copy

# Program parameters
random.seed()
MUTATION_PROB = 0.1
N_GENERATIONS = 15
N_TOURS = 10
TOURNAMENT_SIZE = 3


# Creates an empty array and fills it with shuffled copies of the indices of the cities
def initialize_tours():
    new_tours = np.empty(shape=(N_TOURS, dist_matrix.shape[1]), dtype=int)
    index_list = np.arange(dist_matrix.shape[1])
    for i in range(N_TOURS):
        random.shuffle(index_list)
        new_tours[i] = copy.copy(index_list)
    return new_tours.astype(int)


# Calculates the total distance of a given tour by summing its L2 norms
def get_distance(tour):
    dist = 0
    for i in range(tour.size):
        if i+1 == tour.size:
            dist += dist_matrix[tour[i], tour[0]]
        else:
            dist += dist_matrix[tour[i], tour[i+1]]
    return dist


# Finds the tour with the smallest distance by comparing them
def get_fittest(pop, return_fittest=False):
    fittest_id = 0
    min_dist = get_distance(pop[0])
    for i in range(1, pop.shape[0]):
        temp = get_distance(pop[i])
        if min_dist > temp:
            min_dist = temp
            fittest_id = i
    if return_fittest:
        return fittest_id
    else:
        return min_dist


# Picks a predetermined amount of tours and chooses the fittest
def tournament_selection():
    tournament = np.empty(shape=(TOURNAMENT_SIZE, dist_matrix.shape[1]), dtype=int)
    for i in range(TOURNAMENT_SIZE):
        random_id = int(random.random() * N_TOURS)
        tournament[i] = tours[random_id]
    return tournament[get_fittest(tournament, True)]


def roulette_wheel_selection():
    s = 0
    partial_s = 0
    ide = 0
    for i in range(N_TOURS):
        s += get_distance(tours[i])
    rand = random.uniform(0, s)
    for i in range(N_TOURS):
        if partial_s < rand:
            partial_s += get_distance(tours[i])
            ide = i
    return tours[ide]


# Two point crossover method
# Chooses two points and transfers the chromosomes that are between those points to the child in the same slots
# then fills the remaining slots by iterating through parent2's chromosomes and choosing the ones that the
# child doesn't already have
def crossover(parent1, parent2):
    child = np.ones(shape=(dist_matrix.shape[1]), dtype=int) * -1
    pos1 = int(random.random() * dist_matrix.shape[1])
    pos2 = int(random.random() * dist_matrix.shape[1])
    if pos1 < pos2:
        for i in range(pos1, pos2 + 1):
            child[i] = parent1[i]
    else:
        for i in range(pos2, pos1 + 1):
            child[i] = parent1[i]
    for i in range(parent2.size):
        if -1 not in child:
            break
        found = False
        if pos1 < pos2:
            for j in range(pos1, pos2 + 1):
                if parent2[i] == child[j]:
                    found = True
                    break
        else:
            for j in range(pos2, pos1 + 1):
                if parent2[i] == child[j]:
                    found = True
                    break
        if not found:
            for z in range(child.size):
                if child[z] == -1:
                    child[z] = parent2[i]
                    break
    return child


# Chooses two positions at random and swaps its chromosomes
def mutation(tour):
    if random.random() < MUTATION_PROB:
        pos1 = int(random.random() * dist_matrix.shape[1])
        pos2 = int(random.random() * dist_matrix.shape[1])
        while pos1 == pos2:
            pos2 = int(random.random() * dist_matrix.shape[1])
        temp = tour[pos1]
        tour[pos1] = tour[pos2]
        tour[pos2] = temp
    return tour


# Evolves the population either by using the crossover method or by cloning a single "parent"
# and returns the new population
def evolve_population():
    new_tours = np.ones(shape=(N_TOURS, dist_matrix.shape[1]), dtype=int) * -1
    for i in range(N_TOURS):
        if random.random() < 0.75:
            parent1 = tournament_selection()
            parent2 = tournament_selection()
            while np.array_equal(parent1, parent2):
                parent2 = roulette_wheel_selection()
            new_tours[i] = mutation(crossover(parent1, parent2))
        else:
            new_tours[i] = mutation(tournament_selection())
    return new_tours


# The distance matrix as seen in the image of the exercise in matrix form
dist_matrix = np.array([[0, 4, 4, 7, 3],
                        [4, 0, 2, 3, 5],
                        [4, 2, 0, 2, 3],
                        [7, 3, 2, 0, 6],
                        [3, 5, 3, 6, 0]])
tours = initialize_tours()
for ii in range(N_GENERATIONS):
    tour_id = get_fittest(tours, True)
    dist = get_fittest(tours)
    fitness = 1./dist
    print('~~~~~~~~~~~~~~~~~~~')
    print('Generation ' + str(ii+1) + ': ' + str(dist))
    print(tours[tour_id])
    print('Fitness: ' + "%.8f" % fitness)
    tours = evolve_population()
