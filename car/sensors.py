"""
car/sensors.py – Raycasting sensor system for obstacle detection.

The car uses 7 parallel rays (120° field of view) pointing forward.
Each ray measures distance to the nearest obstacle (wall/boundary).
Distances are [SENSOR_LENGTH, ∞) for free space, or [0, SENSOR_LENGTH] for obstacles.
"""
import math
from typing import List
import pygame
from config import SENSOR_LENGTH


class Sensors:
    """Raycasting-based obstacle detection system."""

    def __init__(self, count: int):
        """
        Initialize sensor array.

        Args:
            count: Number of rays (should be odd so one points straight ahead)
        """
        self.count = count
        self._fov = 120  # Field of view in degrees
        self._step = self._fov / (count - 1) if count > 1 else 0

    def get_readings(
        self,
        car_pos: pygame.Vector2,
        car_angle: float,
        track_mask: pygame.mask.Mask,
    ) -> List[float]:
        """
        Cast rays and measure distance to nearest obstacle.

        Args:
            car_pos: Car's (x, y) position
            car_angle: Car's heading angle in degrees (0=right, 90=up)
            track_mask: Collision mask from track (grass=obstacle, road=free)

        Returns:
            List of distance values for each ray sensor
            SENSOR_LENGTH = no obstacle detected (max range)
            < SENSOR_LENGTH = obstacle at this distance
        """
        readings: List[float] = []
        half_fov = self._fov / 2
        mask_width, mask_height = track_mask.get_size()

        for i in range(self.count):
            # Calculate ray angle for this sensor
            ray_angle_deg = car_angle - half_fov + i * self._step
            ray_angle_rad = math.radians(ray_angle_deg)

            # Pre-compute sin/cos for efficiency
            cos_angle = math.cos(ray_angle_rad)
            sin_angle = math.sin(ray_angle_rad)
            distance = float(SENSOR_LENGTH)

            # Cast ray: step from 1 to max range
            for step_distance in range(1, SENSOR_LENGTH + 1):
                # Ray point at this distance
                x = int(car_pos.x + cos_angle * step_distance)
                y = int(car_pos.y - sin_angle * step_distance)

                # Check map bounds
                if not (0 <= x < mask_width and 0 <= y < mask_height):
                    distance = float(step_distance)
                    break

                # Check obstacle (grass=1 in mask)
                if track_mask.get_at((x, y)):
                    distance = float(step_distance)
                    break

            readings.append(distance)

        return readings