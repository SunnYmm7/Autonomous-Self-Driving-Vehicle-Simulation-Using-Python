"""
simulation/fitness.py – Fitness evaluation for cars in the simulation.

Fitness combines three metrics:
1. Distance traveled: Reward for forward progress (exploration)
2. Efficiency: Bonus for surviving longer (encourages stable control)
3. Damage avoidance: Penalty for hitting obstacles (encourages safe driving)
"""
from car.car import Car


def calculate_fitness(car: Car) -> float:
    """
    Calculate fitness score for a car.

    Fitness = distance + (time_alive * efficiency_weight) - (damage * penalty)

    This combination rewards:
    - Cars that travel further (exploration of track)
    - Cars that survive longer without crashing (stability)
    - Cars that avoid obstacles and potholes (safe driving)

    Args:
        car: Car object with fitness metrics

    Returns:
        Scalar fitness value (higher is better)
    """
    # Main component: distance traveled (exploration)
    distance_fitness = car.distance
    
    # Secondary component: survival time (efficiency/stability)
    # Weighted lower than distance to prioritize progress
    efficiency_bonus = car.time_alive * 0.1
    
    # Penalty component: damage taken (avoid obstacles/potholes)
    # Encourages safe driving and obstacle avoidance
    damage_penalty = car.damage * 0.5
    
    return distance_fitness + efficiency_bonus - damage_penalty