import pygame
import math
from config import SENSOR_LENGTH

class Sensors:
    def __init__(self, count):
        self.count = count

    def get_readings(self, car_pos, car_angle, track_mask):
        readings = []
        # Distribute rays evenly in front of the car
        start_angle = car_angle - 45
        step = 90 / (self.count - 1)

        for i in range(self.count):
            angle = math.radians(start_angle + (i * step))
            distance = 0
            
            # Cast the ray pixel by pixel
            for d in range(1, SENSOR_LENGTH):
                x = car_pos.x + math.cos(angle) * d
                y = car_pos.y - math.sin(angle) * d
                
                # Check if this pixel hits the mask
                if 0 <= x < track_mask.get_size()[0] and 0 <= y < track_mask.get_size()[1]:
                    if track_mask.get_at((int(x), int(y))):
                        distance = d
                        break
                distance = d
            
            readings.append(distance)
        return readings