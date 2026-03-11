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
            #masks for collision detection
        #safe zone
        self.road_mask = pygame.mask.from_threshold(self.surface, (80, 80, 80), (1, 1, 1)) 
        #red ares
        self.odstacle_mask = pygame.mask.from_threshold(self.surface, (255, 0, 0), (1, 1, 1))
        # everything else
        self.wall_mask = pygame.mask.from_threshold(self.surface, (30, 30, 30), (1, 1, 1))
        
    def draw(self, screen):
        screen.blit(self.surface, (0, 0)) 
    
    def debug_draw_mask(self, screen, mask_type="walls"):
        """Visualize masks for debugging."""
        if mask_type == "road":
            mask_surface = self.road_mask.to_surface(setcolor=(0,255,0), unsetcolor=(0,0,0))
        elif mask_type == "obstacles":
            mask_surface = self.obstacle_mask.to_suface(setcolor=(255,0,0), unsetcolor=(0,0,0))
        else:
            mask_surface = self.wall_mask.to_surface(setcolor=(255,255,255), unsetcolor=(0,0,0))
        screen.blit(mask_surface, (0,0))
            
