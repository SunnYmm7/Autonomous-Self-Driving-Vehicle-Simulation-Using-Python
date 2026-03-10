import pygame
import sys
import pickle
from config import *
from car.car import Car
from ai.neural_network import NeuralNetwork
from genetic.population import Genome, evolve
from environment.track import Track
from car.sensors import Sensors

class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI Self-Driving Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        
        self.track = Track()
        self.nn = NeuralNetwork()
        self.sensor_system = Sensors(SENSOR_COUNT)
        
        self.generation = 1
        self.paused = False
        # Button Rect for UI
        self.stop_btn = pygame.Rect(SCREEN_WIDTH - 120, 10, 100, 40)
        
        self.population = [Genome() for _ in range(POPULATION_SIZE)]
        self.reset_env()
        self.load_btn = pygame.Rect(SCREEN_WIDTH - 230, 10, 100, 40)

    def reset_env(self):
        # Coordinates must be on the gray road of your Track
        self.cars = [Car(170, 400, g) for g in self.population]

    def run(self):
        """Main Loop: Ensure this is indented exactly like __init__"""
        while True:
            self.handle_events()
            if not self.paused:
                self.update()
            self.draw()
            self.clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.save_best_and_quit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Toggle Pause
                if self.stop_btn.collidepoint(event.pos):
                    self.paused = not self.paused
                
                # Handle Loading
                if self.load_btn.collidepoint(event.pos):
                    from genetic.population import load_best_genome, create_population_from_saved
                    saved = load_best_genome()
                    if saved:
                        self.population = create_population_from_saved(saved)
                        self.generation = "Loaded"
                        self.reset_env()
                        print("Loaded Champion Genome!")

    def update(self):
        alive_count = 0
        for car in self.cars:
            if car.alive:
                # 1. Update sensors using the track mask
                car.sensors = self.sensor_system.get_readings(
                    car.pos, car.angle, self.track.mask
                )
                
                # 2. Get AI action
                inputs = car.get_inputs()
                action = self.nn.predict(inputs, car.genome.weights)
                
                # 3. Move car
                car.update(action)

                # 4. Collision check (Did we hit a wall/obstacle?)
                try:
                    if self.track.mask.get_at((int(car.pos.x), int(car.pos.y))):
                        car.alive = False
                        car.genome.fitness = car.distance
                except IndexError:
                    car.alive = False
                
                alive_count += 1

        if alive_count == 0:
            self.population = evolve(self.population)
            self.generation += 1
            self.reset_env()

    def draw(self):
        self.track.draw(self.screen)
        for car in self.cars:
            car.draw(self.screen)

        # Draw Buttons
        pygame.draw.rect(self.screen, (200, 50, 50) if not self.paused else (50, 200, 50), self.stop_btn)
        pygame.draw.rect(self.screen, (50, 50, 200), self.load_btn) # Blue Load Button
        
        self.screen.blit(self.font.render("STOP" if not self.paused else "START", True, (255,255,255)), (self.stop_btn.x + 25, self.stop_btn.y + 10))
        self.screen.blit(self.font.render("LOAD", True, (255,255,255)), (self.load_btn.x + 25, self.load_btn.y + 10))
        
        # Stats
        stats = self.font.render(f"Gen: {self.generation} | Alive: {sum(1 for c in self.cars if c.alive)}", True, (255, 255, 255))
        self.screen.blit(stats, (10, 10))
        
        pygame.display.flip()

    def save_best_and_quit(self):
        # Save the best performing genome before closing
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        with open("saved_genome.pkl", "wb") as f:
            pickle.dump(self.population[0], f)
        pygame.quit()
        sys.exit()