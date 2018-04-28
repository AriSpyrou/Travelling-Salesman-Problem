import random
import numpy as np
import time
import copy

# Seed the random number generator.
random.seed(time.time())
# Set the program parameters.
MUTATION_PROB = 0.6
N_GENERATIONS = 20
N_TOURS = 20


# Creates an empty ndarray and fills it with shuffled copies of the indices of the cities
def initialize_tours():
    new_tours = np.empty(shape=(N_TOURS, dist_matrix.shape[0]))
    index_list = np.arange(dist_matrix.shape[0])
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
def get_fittest():
    # fittest = tours[0]
    min_dist = get_distance(tours[0])
    for i in range(1, N_TOURS):
        temp = get_distance(tours[i])
        if min_dist > temp:
            min_dist = temp
            # fittest = tours[i]
    return min_dist


# The distance matrix as seen in the image of the exercise in matrix form
dist_matrix = np.array([[0, 4, 4, 7, 3],
                        [4, 0, 2, 3, 5],
                        [4, 2, 0, 2, 3],
                        [7, 3, 2, 0, 6],
                        [3, 5, 3, 6, 0]])
tours = initialize_tours()
print('Initial distance: '+str(get_fittest()))
print('done')
