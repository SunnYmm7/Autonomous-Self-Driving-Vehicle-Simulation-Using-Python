"""
genetic/mutation.py – Gaussian mutation operator.

Note: This module duplicates functionality in population.py which is preferred.
Kept for reference and modular design. Current codebase uses _mutate() from population.py.
"""
import numpy as np
from config import MUTATION_RATE, MUTATION_STD, GENOME_SIZE
from genetic.population import Genome


def mutate(genome: Genome) -> Genome:
    """
    Apply Gaussian mutation to a genome.

    Mutation probability (MUTATION_RATE) determines the fraction of weights
    that are randomly perturbed. Perturbed weights are offset by Gaussian
    noise with standard deviation MUTATION_STD.

    Args:
        genome: Genome to mutate (not modified in place)

    Returns:
        New mutated Genome (original unchanged)
    """
    weights_copy = genome.weights.copy()
    mutation_mask = np.random.rand(GENOME_SIZE) < MUTATION_RATE
    num_mutations = mutation_mask.sum()
    
    weights_copy[mutation_mask] += np.random.normal(
        0, MUTATION_STD, num_mutations
    ).astype(np.float32)
    
    return Genome(weights_copy)