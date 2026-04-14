"""
genetic/population.py – Genome data structure and genetic algorithm operators.

Key concepts:
- Genome: Weight vector for a neural network (candidate solution)
- Evolution: Select best performers, breed new generation via crossover & mutation
- Elite preservation: Keep top genomes unchanged for stability
"""
import numpy as np
import random
import pickle
from typing import List, Optional
from config import (POPULATION_SIZE, GENOME_SIZE, MUTATION_RATE, MUTATION_STD,
                    ELITE_COUNT, SELECTION_TOP)


class Genome:
    """
    Neural network weight vector with associated fitness score.
    
    Used to represent a candidate solution in the genetic algorithm.
    """

    def __init__(self, weights: Optional[np.ndarray] = None):
        """
        Initialize a genome with weights.

        Args:
            weights: Pre-existing weight vector, or None to generate randomly
        """
        if weights is not None:
            self.weights = np.asarray(weights, dtype=np.float32)
        else:
            # Xavier/Glorot-style initialization (mean=0, std~0.5)
            self.weights = (np.random.randn(GENOME_SIZE) * 0.5).astype(np.float32)
        
        self.fitness: float = 0.0

    def clone(self) -> "Genome":
        """Create an independent copy of this genome."""
        return Genome(self.weights.copy())


# ──────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────
def save_genome(genome: Genome, filename: str = "saved_genome.pkl") -> None:
    """
    Save a genome to disk for later loading.

    Args:
        genome: Genome to save
        filename: Output file path
    """
    with open(filename, "wb") as f:
        pickle.dump(genome, f)


def load_best_genome(filename: str = "saved_genome.pkl") -> Optional[Genome]:
    """
    Load a previously saved genome.

    Args:
        filename: Path to pickle file

    Returns:
        Loaded Genome, or None if file not found
    """
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def create_population_from_saved(saved: Genome) -> List[Genome]:
    """
    Bootstrap a population starting from a saved good genome.
    
    The saved genome is cloned and perturbed slightly to create diversity.

    Args:
        saved: The elite genome to build from

    Returns:
        List of POPULATION_SIZE genomes
    """
    population = [saved.clone()]
    
    while len(population) < POPULATION_SIZE:
        # Clone and add small random noise for variation
        variant = saved.clone()
        variant.weights += np.random.normal(
            0, 0.04, GENOME_SIZE
        ).astype(np.float32)
        population.append(variant)
    
    return population


# ──────────────────────────────────────────────────────────────────────
# Genetic Operators
# ──────────────────────────────────────────────────────────────────────
def _crossover(parent1: Genome, parent2: Genome) -> Genome:
    """
    Single-point crossover via random mask.
    
    Each weight is randomly inherited from one parent.

    Args:
        parent1: First parent genome
        parent2: Second parent genome

    Returns:
        Child genome containing mix of both parents' weights
    """
    # Randomly choose which parent contributes each weight
    mask = np.random.rand(GENOME_SIZE) < 0.5
    child_weights = np.where(mask, parent1.weights, parent2.weights)
    return Genome(child_weights)


def _mutate(genome: Genome) -> Genome:
    """
    Gaussian mutation: add random noise to some weights.
    
    Each of the GENOME_SIZE weights has a MUTATION_RATE probability
    of being perturbed by Gaussian noise.

    Args:
        genome: Genome to mutate

    Returns:
        Mutated copy (original unchanged)
    """
    weights = genome.weights.copy()
    
    # Determine which weights to mutate
    mutate_mask = np.random.rand(GENOME_SIZE) < MUTATION_RATE
    num_mutations = mutate_mask.sum()
    
    # Add Gaussian noise to selected weights
    weights[mutate_mask] += np.random.normal(
        0, MUTATION_STD, num_mutations
    ).astype(np.float32)
    
    return Genome(weights)


# ──────────────────────────────────────────────────────────────────────
# Evolution Pipeline
# ──────────────────────────────────────────────────────────────────────
def evolve(population: List[Genome]) -> List[Genome]:
    """
    Execute one generation of the genetic algorithm.

    Algorithm:
    1. Sort by fitness (best first)
    2. Preserve elite genomes
    3. Breed new generation from best genomes via crossover + mutation

    Args:
        population: Current generation (genomes must have fitness set)

    Returns:
        Next generation (same size as input)
    """
    # Sort by fitness descending
    population.sort(key=lambda g: g.fitness, reverse=True)
    
    # Best performers for breeding
    best_pool = population[:SELECTION_TOP]

    # Save best genome of current generation
    save_genome(best_pool[0], "best_car.pkl")

    # Elitism: copy top genomes unchanged
    next_generation = [g.clone() for g in best_pool[:ELITE_COUNT]]

    # Fill rest of population with offspring
    while len(next_generation) < POPULATION_SIZE:
        # Select two random parents from elite pool
        parent1, parent2 = random.sample(best_pool, 2)
        
        # Breed: crossover then mutate
        child = _mutate(_crossover(parent1, parent2))
        next_generation.append(child)

    return next_generation