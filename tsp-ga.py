import random
import math
import numpy as np
from matplotlib import pyplot as plt
import time
import copy

# Seed the random number generator.
random.seed(time.time())
# Set the program parameters.
MUTATION_PROB = 0.6
N_CITIES = 5
N_GENERATIONS = 20
N_TOURS = 40
GRID_SIZE = 20


# Creates an empty ndarray and fills it with random coordinates
def initialize_cities():
    new_cities = np.empty(shape=(N_CITIES, 2), dtype=int)
    for i in range(N_CITIES):
        new_cities[i] = [random.randint(0, GRID_SIZE), random.randint(0, GRID_SIZE)]
    return new_cities.astype(int)


# Creates an empty ndarray and fills it with shuffled copies of the indices of the cities
def initialize_tours():
    new_tours = np.empty(shape=(N_TOURS, N_CITIES))
    index_list = np.arange(N_CITIES)
    for i in range(N_TOURS):
        random.shuffle(index_list)
        new_tours[i] = copy.copy(index_list)
    return new_tours.astype(int)


# Calculates the total distance of a given tour by summing its L2 norms
def get_distance(tour):
    dist = 0
    for i in range(tour.size):
        if i+1 == tour.size:
            dist += np.linalg.norm(cities[tour[i]] - cities[tour[0]])
        else:
            dist += np.linalg.norm(cities[tour[i]] - cities[tour[i + 1]])
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


cities = initialize_cities()
tours = initialize_tours()
print('Initial distance: '+str(get_fittest()))
print('done')
