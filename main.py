import pygame
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1200, 800
CAR_SIZE = 20
SENSOR_LENGTH = 150
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (50, 50, 50)
ROAD_COLOR = (100, 100, 100)

class Car:
    def __init__(self):
        self.surface = pygame.Surface((CAR_SIZE, CAR_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(self.surface, RED, (0, 0, CAR_SIZE, CAR_SIZE))
        self.rect = self.surface.get_rect(center=(WIDTH//2, HEIGHT//2))
        self.angle = 0
        self.speed = 6
        self.radars = []
        self.alive = True
        self.center = [150, 150] # Start Position

    def draw(self, screen):
        # Rotate car
        rotated_surface = pygame.transform.rotate(self.surface, -self.angle)
        rect = rotated_surface.get_rect(center=self.center)
        screen.blit(rotated_surface, rect.topleft)
        self.draw_radars(screen)

    def draw_radars(self, screen):
        for radar in self.radars:
            pos, dist = radar
            pygame.draw.line(screen, GREEN, self.center, pos, 1)
            pygame.draw.circle(screen, GREEN, pos, 3)

    def check_collision(self, map_surface):
        self.alive = True
        # Check simple point collision at the center
        if 0 <= self.center[0] < WIDTH and 0 <= self.center[1] < HEIGHT:
            if map_surface.get_at((int(self.center[0]), int(self.center[1]))) != ROAD_COLOR:
                self.alive = False
        else:
            self.alive = False

    def check_radar(self, degree, map_surface):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not map_surface.get_at((x, y)) != ROAD_COLOR and len < SENSOR_LENGTH:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, map_surface):
        # Drive forward
        self.center[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.center[1] += math.sin(math.radians(360 - self.angle)) * self.speed

        # Clear and update radars
        self.radars.clear()
        # 5 angles: -90, -45, 0, 45, 90
        for d in [-90, -45, 0, 45, 90]:
            self.check_radar(d, map_surface)
        
        self.check_collision(map_surface)

    def get_data(self):
        # Normalize distances for the neural network
        return [int(r[1]) / SENSOR_LENGTH for r in self.radars]

class TrackEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self):
        super(TrackEnv, self).__init__()
        self.action_space = spaces.Discrete(3) # 0: Left, 1: Straight, 2: Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.training_active = False
        self.button_start = pygame.Rect(WIDTH - 320, 20, 140, 50)
        self.button_stop = pygame.Rect(WIDTH - 160, 20, 140, 50)

                
        # Pygame Setup
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("RL Self Driving Car - PPO")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        
        # Generate Map
        self.map_surface = pygame.Surface((WIDTH, HEIGHT))
        self.map_surface.fill(BLACK)
        self.draw_track()
        
        self.car = Car()
        self.mode = "IDLE" # IDLE, SINGLE, AUTO
        self.generation = 0
        self.waiting_for_input = True

    
    def handle_button_clicks(self, mouse_pos):
        """Handle start/stop button clicks"""
        if self.button_start.collidepoint(mouse_pos):
            self.training_active = True
            print("▶ Training STARTED")
            return "START"
        elif self.button_stop.collidepoint(mouse_pos):
            self.training_active = False
            print("⏸ Training PAUSED")
            return "STOP"
        return None

    def draw_track(self):
        # Draw a complex closed loop track
        points = [
            (150, 150), (400, 100), (800, 100), (1000, 300), 
            (1000, 600), (600, 700), (200, 600), (100, 400)
        ]
        pygame.draw.lines(self.map_surface, ROAD_COLOR, True, points, 90)

    def step(self, action):
        # Action Logic
        if action == 0: # Turn Left
            self.car.angle += 5
        elif action == 2: # Turn Right
            self.car.angle -= 5
        
        self.car.update(self.map_surface)

        # Rewards
        reward = 1 # Reward for staying alive
        done = False
        
        if not self.car.alive:
            reward = -100 # Penalty for crashing
            done = True
        
        observation = np.array(self.car.get_data(), dtype=np.float32)
        info = {}
        
        # Rendering inside step for visualization
        self.render()
        
        return observation, reward, done, False, info

    def reset(self, seed=None):
        self.car = Car()
        self.generation += 1
        
        # --- FIX: Force an initial look ---
        # The car is created blind. We must cast rays once immediately 
        # so the AI has data for the very first frame.
        self.car.radars.clear()
        for d in [-90, -45, 0, 45, 90]:
            self.car.check_radar(d, self.map_surface)
        # ----------------------------------

        return np.array(self.car.get_data(), dtype=np.float32), {}

    def render(self):
        if self.render_mode != 'human':
            return
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_button_clicks(event.pos)
        
        # Draw everything
        self.screen.fill(BLACK)
        self.screen.blit(self.map_surface, (0, 0))
        self.car.draw(self.screen)
        
        # Draw START button
        start_color = GREEN if self.training_active else (0, 150, 0)
        pygame.draw.rect(self.screen, start_color, self.button_start)
        pygame.draw.rect(self.screen, WHITE, self.button_start, 2)
        start_text = self.font.render("START", True, WHITE)
        text_rect = start_text.get_rect(center=self.button_start.center)
        self.screen.blit(start_text, text_rect)
        
        # Draw STOP button
        stop_color = RED if not self.training_active else (150, 0, 0)
        pygame.draw.rect(self.screen, stop_color, self.button_stop)
        pygame.draw.rect(self.screen, WHITE, self.button_stop, 2)
        stop_text = self.font.render("STOP", True, WHITE)
        text_rect = stop_text.get_rect(center=self.button_stop.center)
        self.screen.blit(stop_text, text_rect)
        
        # Draw stats
        stats = [
            f"Episode: {self.episode_count}",
            f"Steps: {self.total_steps}",
            f"Distance: {int(self.car.distance_traveled)}",
            f"Best: {int(self.best_distance)}",
            f"Status: {'TRAINING' if self.training_active else 'PAUSED'}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, WHITE)
            self.screen.blit(text, (10, 10 + i * 25))
        
        pygame.display.flip()
        self.clock.tick(FPS)


# Add this new callback class:
class StartStopCallback(BaseCallback):
    """Callback to handle start/stop button logic"""
    def __init__(self, env, verbose=0):
        super(StartStopCallback, self).__init__(verbose)
        self.env_ref = env
        
    def _on_step(self) -> bool:
        # Keep rendering even when paused
        self.env_ref.render()
        
        # Pause training if stop button was pressed
        while not self.env_ref.training_active:
            self.env_ref.render()
            pygame.time.wait(100)  # Small delay to prevent CPU overuse
        
        return True


# In the main execution, replace the callback with:
callback = StartStopCallback(env, verbose=1)

# The training will now pause when STOP is clicked and resume when START is clicked That's

class InteractiveCallback(BaseCallback):
    """
    Custom callback to handle the Pause/Play logic for the UI.
    """
    def __init__(self, env):
        super(InteractiveCallback, self).__init__()
        self.env_ref = env

    def _on_step(self) -> bool:
        # Check if we need to pause for user input
        if self.env_ref.mode == "SINGLE" and self.locals.get("dones")[0]:
            self.env_ref.mode = "IDLE"
            self.env_ref.waiting_for_input = True
            
        # If IDLE, we loop here consuming events until a mode is picked
        while self.env_ref.waiting_for_input:
            self.env_ref.render()
            
        return True

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Create Environment
    env = TrackEnv()
    
    # Initialize PPO Agent (State of the Art RL)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)

    print("---------------------------------------")
    print("Press 'Run 1 Iteration' to train for one life cycle.")
    print("Press 'Auto Loop' to train continuously.")
    print("---------------------------------------")

    # Start Training with our interactive callback
    callback = InteractiveCallback(env)
    
    # We train for a large number of steps, but the callback controls the flow
    model.learn(total_timesteps=100000, callback=callback)
