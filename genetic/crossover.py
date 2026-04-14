"""
genetic/crossover.py – Uniform crossover operator.

Note: This module duplicates functionality in population.py which is preferred.
Kept for reference and modular design. Current codebase uses _crossover() from population.py.
"""
import numpy as np
from config import GENOME_SIZE
from genetic.population import Genome


def crossover(parent1: Genome, parent2: Genome) -> Genome:
    """
    Create an offspring combining traits from two parents.

    Uniform crossover: each weight independently inherited from either parent
    with 50/50 probability. This prevents the blocking problem of single-point
    crossover and promotes mixing of genetic material.

    Args:
        parent1: First parent Genome
        parent2: Second parent Genome

    Returns:
        New child Genome with mixed genetics (neither parent modified)
    """
    # Randomly select which parent contributes each weight
    inheritance_mask = np.random.rand(GENOME_SIZE) < 0.5
    child_weights = np.where(inheritance_mask, parent1.weights, parent2.weights)
    return Genome(child_weights)