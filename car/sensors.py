import pygame
import math
from config import SENSOR_LENGTH

class Sensors:
    def __init__(self, count, arc=90):
        self.count = count
        self.arc =arc 

    def get_readings(self, car_pos, car_angle, track_mask):
        readings = []
        # Distribute rays evenly in front of the car
        start_angle = car_angle - (self.arc / 2)
        step = self.arc / (self.count - 1)

        for i in range(self.count):
            angle = math.radians(start_angle + (i * step))
            distance = SENSOR_LENGTH
            
            # Cast the ray pixel by pixel
            for d in range(1, SENSOR_LENGTH, 2):
                x = car_pos.x + math.cos(angle) * d
                y = car_pos.y - math.sin(angle) * d
                
                # Check if this pixel hits the mask
                if 0 <= x < track_mask.get_size()[0] and 0 <= y < track_mask.get_size()[1]:
                    if track_mask.get_at((int(x), int(y))):
                        distance = d
                        break
            
            readings.append(distance)
        return readings

    def draw(self, screen, car_pops, car_angle, readings):
        """Visualize sensor rays fro debugging."""
        start_angle = car_angle - (self.arc/2)
        step = self.arc/(self.count - 1)

        for i, dist in enumerate(readings):
            angeel = math.radians(start_angel + (i * step))
            end_x = car_pos.x + math.cos(angle) * dist
            end_y = car_pos.y - math.sin(angle) * dist
            pygame.draw.line(screen, (0, 255, 0), (car_pos.x, car_pos.y), (end_x, end_y), 1)
            pygame.draw.circle(screen, (255, 0, 0), (int(end_x), int(end_y)), 3)
