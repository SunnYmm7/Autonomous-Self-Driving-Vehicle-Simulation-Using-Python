"""
car/car.py – Car agent with physics, sensor integration, and rendering.

The Car class represents an autonomous vehicle in the simulation with:
- Position and velocity tracking
- Sensor-based environmental awareness (7-ray raycasting)
- Neural network control
- Visual trail and sensor visualization
"""
import pygame
import math
from typing import List, Tuple
import numpy as np
from config import (SENSOR_COUNT, SENSOR_LENGTH, CAR_RADIUS,
                    MAX_SPEED, MIN_SPEED, STEER_POWER,
                    COL_CAR_ALIVE, COL_CAR_DEAD, COL_BEST_CAR,
                    COL_SENSOR, COL_SENSOR_HIT)

# Trail history limit (improves rendering performance)
MAX_TRAIL_LENGTH = 60

# Sensor visualization parameters
SENSOR_FOV_DEGREES = 120
SENSOR_HIT_THRESHOLD_RATIO = 0.85  # Ratio to SENSOR_LENGTH for hit detection
SENSOR_HIT_CIRCLE_RADIUS = 3  # Pixels


class Car:
    """Autonomous vehicle with physics simulation and neural network control."""
    
    def __init__(self, x: float, y: float, genome: np.ndarray):
        """
        Initialize a car at the given position.
        
        Args:
            x: Initial x coordinate (pixels)
            y: Initial y coordinate (pixels)
            genome: Neural network weight vector for this car
        """
        self.pos: pygame.Vector2 = pygame.Vector2(x, y)
        self.angle: float = 0.0  # Degrees, 0=right, 90=up
        self.speed: float = MIN_SPEED
        self.genome: np.ndarray = genome
        self.alive: bool = True
        self.distance: float = 0.0  # Total distance traveled
        self.time_alive: int = 0  # Tick counter for efficiency fitness
        self.health: float = 100.0  # Car health (0-100), dies at 0
        self.damage: float = 0.0  # Cumulative damage taken
        self.sensors: List[float] = [float(SENSOR_LENGTH)] * SENSOR_COUNT
        self._trail: List[Tuple[int, int]] = []
        self._pothole_cooldown: int = 0  # Frames to prevent damage spam from same pothole

    # ──────────────────────────────────────────────────────────────────
    # AI Input
    # ──────────────────────────────────────────────────────────────────
    def get_inputs(self) -> np.ndarray:
        """
        Get normalized sensor readings for neural network input.
        
        Returns:
            1D array of SENSOR_COUNT values in [0, 1] range.
            1.0 = no obstacle (max distance), 0.0 = collision (zero distance)
        """
        return np.array(self.sensors, dtype=np.float32) / SENSOR_LENGTH

    # ──────────────────────────────────────────────────────────────────
    # Physics Simulation
    # ──────────────────────────────────────────────────────────────────
    def update(self, action: np.ndarray) -> None:
        """
        Apply neural network action and update physics.
        
        Args:
            action: 2-element array [steering, acceleration]
                   Values in [-1, 1] range
        """
        if not self.alive:
            return

        # Update cooldowns
        if self._pothole_cooldown > 0:
            self._pothole_cooldown -= 1

        # Steering: adjust angle (degrees)
        self.angle += float(action[0]) * STEER_POWER
        
        # Acceleration: modulate speed within bounds
        self.speed = float(np.clip(
            self.speed + float(action[1]) * 0.4,
            MIN_SPEED, MAX_SPEED
        ))

        # Update position using angle and speed
        rad = math.radians(self.angle)
        self.pos.x += math.cos(rad) * self.speed
        self.pos.y -= math.sin(rad) * self.speed  # Negative because y increases downward

        # Update fitness tracking
        self.distance += self.speed
        self.time_alive += 1

    def take_pothole_damage(self, damage: float = 5.0) -> None:
        """
        Apply damage from hitting a pothole.
        
        Args:
            damage: Damage amount (default 5 hp)
        """
        # Cooldown prevents multiple hits from same pothole
        if self._pothole_cooldown > 0:
            return
        
        self.health -= damage
        self.damage += damage
        self._pothole_cooldown = 15  # 0.25 seconds at 60 fps - prevent spam
        
        # Die if health drops to 0
        if self.health <= 0:
            self.alive = False
            self.health = 0

    def hit_speedbreaker(self, speed_reduction: float = 0.25) -> None:
        """
        Apply speed reduction when hitting a speedbreaker (bump).
        Speed temporarily goes below MIN_SPEED during impact, then recovers.
        Also reduces health slightly.
        
        Args:
            speed_reduction: Speed multiplier (e.g., 0.25 = 25% of normal speed)
        """
        # Allow speed to drop below minimum temporarily when hitting speedbreaker
        # The neural network acceleration will naturally recover it back up
        self.speed *= speed_reduction
        
        # Small damage from impact
        self.take_pothole_damage(1.0)

    def add_trail_point(self) -> None:
        """Record current position for trail visualization."""
        self._trail.append((int(self.pos.x), int(self.pos.y)))
        if len(self._trail) > MAX_TRAIL_LENGTH:
            self._trail.pop(0)

    # ──────────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────────
    def draw(self, screen: pygame.Surface, is_best: bool = False) -> None:
        """
        Render the car and its sensors.
        
        Args:
            screen: Pygame surface to draw on
            is_best: If True, draw detailed visualization (trail, bright sensors)
        """
        if is_best:
            self._draw_trail(screen)
        if self.alive:
            self._draw_sensors(screen, is_best)
        self._draw_body(screen, is_best)

    def _draw_trail(self, screen: pygame.Surface) -> None:
        """Draw the car's path history with fading effect."""
        if len(self._trail) < 2:
            return
        
        num_points = len(self._trail)
        for i in range(1, num_points):
            # Fade color from dark to bright
            alpha_ratio = i / num_points
            color = (
                int(COL_BEST_CAR[0] * alpha_ratio),
                int(COL_BEST_CAR[1] * alpha_ratio),
                int(COL_BEST_CAR[2] * alpha_ratio * 0.5)
            )
            pygame.draw.line(screen, color, self._trail[i - 1], self._trail[i], 2)

    def _draw_sensors(self, screen: pygame.Surface, is_best: bool) -> None:
        """Draw the 7-ray sensor fan."""
        half_fov = SENSOR_FOV_DEGREES / 2
        num_sensors = len(self.sensors)
        step_angle = SENSOR_FOV_DEGREES / (num_sensors - 1) if num_sensors > 1 else 0

        for i, distance in enumerate(self.sensors):
            # Ray angle relative to car orientation
            ray_angle = self.angle - half_fov + i * step_angle
            ray_rad = math.radians(ray_angle)
            
            # Ray endpoint
            end_x = self.pos.x + math.cos(ray_rad) * distance
            end_y = self.pos.y - math.sin(ray_rad) * distance

            # Detect if ray hit something (distance not at max)
            hit_obstacle = distance < (SENSOR_LENGTH * SENSOR_HIT_THRESHOLD_RATIO)
            
            if is_best:
                # Detailed visualization for best car
                color = COL_SENSOR_HIT if hit_obstacle else COL_SENSOR
                pygame.draw.line(screen, color, self.pos, (end_x, end_y), 1)
                
                if hit_obstacle:
                    pygame.draw.circle(
                        screen, COL_SENSOR_HIT,
                        (int(end_x), int(end_y)), SENSOR_HIT_CIRCLE_RADIUS
                    )
            else:
                # Subtle visualization for other cars
                pygame.draw.line(screen, (40, 70, 90), self.pos, (end_x, end_y), 1)

    def _draw_body(self, screen: pygame.Surface, is_best: bool) -> None:
        """Draw the car as a triangle pointing in its direction."""
        # Select color based on status
        color = COL_BEST_CAR if is_best else (COL_CAR_ALIVE if self.alive else COL_CAR_DEAD)

        # Geometry
        rad = math.radians(self.angle)
        perp = rad + math.pi / 2
        r = CAR_RADIUS
        cx, cy = self.pos.x, self.pos.y

        # Triangle vertices: front point + left/right flanks
        front = (cx + math.cos(rad) * r * 1.6, cy - math.sin(rad) * r * 1.6)
        left = (cx + math.cos(perp) * r, cy - math.sin(perp) * r)
        right = (cx - math.cos(perp) * r, cy + math.sin(perp) * r)

        # Draw filled triangle
        pygame.draw.polygon(screen, color, [front, left, right])

        # Draw outline for emphasis
        if is_best:
            pygame.draw.polygon(screen, (255, 255, 200), [front, left, right], 2)
        elif self.alive:
            pygame.draw.polygon(screen, (20, 80, 40), [front, left, right], 1)