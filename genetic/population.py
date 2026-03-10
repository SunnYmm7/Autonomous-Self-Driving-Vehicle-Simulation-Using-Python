import numpy as np
import random
import pickle
from config import POPULATION_SIZE, GENOME_SIZE, MUTATION_RATE

def load_best_genome(filename="saved_genome.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def create_population_from_saved(saved_genome):
    # Create a new population where everyone is a slight mutation of the champion
    new_pop = []
    new_pop.append(saved_genome) # Keep the original
    
    while len(new_pop) < POPULATION_SIZE:
        # Clone and mutate slightly
        mutated_weights = saved_genome.weights + np.random.normal(0, 0.05, GENOME_SIZE)
        new_pop.append(Genome(mutated_weights))
    return new_pop
class Genome:
    def __init__(self, weights=None):
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.uniform(-1, 1, GENOME_SIZE)
        self.fitness = 0

def evolve(population):
    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)
    best_genomes = population[:10]
    
    # Save the best of this generation
    with open("best_car.pkl", "wb") as f:
        pickle.dump(best_genomes[0], f)

    new_pop = []
    # Keep the top 2 (Elitism)
    new_pop.extend(best_genomes[:2])

    while len(new_pop) < POPULATION_SIZE:
        p1, p2 = random.sample(best_genomes, 2)
        # Crossover
        split = random.randint(0, GENOME_SIZE)
        child_weights = np.concatenate((p1.weights[:split], p2.weights[split:]))
        
        # Mutation
        for i in range(len(child_weights)):
            if random.random() < MUTATION_RATE:
                child_weights[i] += np.random.normal(0, 0.1)
        
        new_pop.append(Genome(child_weights))
    return new_pop