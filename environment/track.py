import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT

class Track:
    def __init__(self):
        self.surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.surface.fill((30, 30, 30)) # Background (Wall)
        
        # Draw Road (Gray Path)
        pygame.draw.rect(self.surface, (80, 80, 80), (100, 100, 800, 600), border_radius=100)
        # Draw Infield (Wall)
        pygame.draw.rect(self.surface, (30, 30, 30), (250, 250, 500, 300), border_radius=50)
        
        # Obstacles (Colored uniquely so mask picks them up)
        self.obstacles = [
            pygame.Rect(450, 100, 20, 150), # Top gate
            pygame.Rect(100, 380, 150, 20), # Side gate
        ]
        for obs in self.obstacles:
            pygame.draw.rect(self.surface, (255, 0, 0), obs)

        # The Mask: 0 for Road, 1 for everything else
        self.mask = pygame.mask.from_threshold(self.surface, (80, 80, 80), (1, 1, 1))
        self.mask.invert() # Invert so non-road pixels are collisions

    def draw(self, screen):
        screen.blit(self.surface, (0, 0))