import pygame
import math
import numpy as np
from config import SENSOR_COUNT, SENSOR_LENGTH

class Car:
    def __init__(self, x, y, genome):
        self.pos = pygame.Vector2(x, y)
        self.angle = 0
        self.speed = 0
        self.genome = genome
        self.alive = True
        self.distance = 0
        self.time_alive = 0
        self.sensors = [0] * SENSOR_COUNT

    def get_inputs(self):
        # Normalize sensors for the AI (0 to 1)
        return np.array(self.sensors) / SENSOR_LENGTH

    def update(self, action):
        if not self.alive: return
        
        # Action[0] = Steering, Action[1] = Acceleration
        self.angle += action[0] * 5
        self.speed = max(2, self.speed + action[1]) # Keep a minimum speed
        
        # Physics math
        rad = math.radians(self.angle)
        self.pos.x += math.cos(rad) * self.speed
        self.pos.y -= math.sin(rad) * self.speed
        
        self.distance += self.speed
        self.time_alive += 1

    def draw(self, screen):
        color = (0, 255, 0) if self.alive else (200, 0, 0)
        
        # 1. Draw Sensor Rays (Only for the living)
        if self.alive:
            start_angle = self.angle - 45
            step = 90 / (len(self.sensors) - 1)
            for i, dist in enumerate(self.sensors):
                angle = math.radians(start_angle + (i * step))
                end_x = self.pos.x + math.cos(angle) * dist
                end_y = self.pos.y - math.sin(angle) * dist
                pygame.draw.line(screen, (100, 100, 100), self.pos, (end_x, end_y), 1)

        # 2. Draw Car Body
        pygame.draw.circle(screen, color, (int(self.pos.x), int(self.pos.y)), 10)