"""
genetic/selection.py – Selection of top performers for breeding.

Note: This module duplicates functionality in population.py which is preferred.
Kept for reference and modular design. Current codebase uses evolve() from population.py.
"""
from typing import List
from config import SELECTION_TOP
from genetic.population import Genome


def select(population: List[Genome]) -> List[Genome]:
    """
    Select the best genomes from a population by fitness.

    This is a simple elitist selection that returns the top SELECTION_TOP
    performers, commonly used for breeding the next generation.

    Args:
        population: List of Genomes (fitness must be set)

    Returns:
        Top SELECTION_TOP genomes sorted by fitness (highest first)
    """
    population.sort(key=lambda g: g.fitness, reverse=True)
    return population[:SELECTION_TOP]