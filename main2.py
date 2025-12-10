import pygame
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import time

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1200, 800
CAR_SIZE = 20
SENSOR_LENGTH = 200
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ROAD_COLOR = (100, 100, 100)

class Car:
    def __init__(self):
        self.surface = pygame.Surface((CAR_SIZE, CAR_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(self.surface, RED, (0, 0, CAR_SIZE, CAR_SIZE))
        self.angle = 0
        self.speed = 7
        self.radars = []
        self.alive = True
        self.center = [150, 150]
        self.distance_traveled = 0
        self.time_alive = 0

    def draw(self, screen):
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
        # Check multiple points on the car for better collision detection
        points = [
            self.center,
            [self.center[0] + 8, self.center[1]],
            [self.center[0] - 8, self.center[1]],
            [self.center[0], self.center[1] + 8],
            [self.center[0], self.center[1] - 8]
        ]
        
        for point in points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                if map_surface.get_at((x, y)) != ROAD_COLOR:
                    self.alive = False
                    return
            else:
                self.alive = False
                return

    def check_radar(self, degree, map_surface):
        length = 0
        x = int(self.center[0])
        y = int(self.center[1])

        while length < SENSOR_LENGTH:
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
            
            if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                break
            if map_surface.get_at((x, y)) != ROAD_COLOR:
                break
            length += 1

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, map_surface):
        # Save previous position
        prev_x, prev_y = self.center[0], self.center[1]
        
        # Move forward
        self.center[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.center[1] += math.sin(math.radians(360 - self.angle)) * self.speed

        # Calculate distance traveled
        dx = self.center[0] - prev_x
        dy = self.center[1] - prev_y
        self.distance_traveled += math.sqrt(dx*dx + dy*dy)
        self.time_alive += 1

        # Update radars
        self.radars.clear()
        for d in [-90, -45, 0, 45, 90]:
            self.check_radar(d, map_surface)
        
        self.check_collision(map_surface)

    def get_data(self):
        # Normalize radar distances and add speed/angle info
        radar_data = [r[1] / SENSOR_LENGTH for r in self.radars]
        return radar_data

class TrackEnv(gym.Env):
    """Improved Custom Environment with better reward shaping"""
    metadata = {'render_modes': ['human'], 'render_fps': FPS}
    
    def __init__(self, render_mode='human'):
        super(TrackEnv, self).__init__()
        self.render_mode = render_mode
        # Action space: 0=Left, 1=Straight, 2=Right
        self.action_space = spaces.Discrete(3)
        # Observation space: 5 radar sensors
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.training_active = False
        self.button_start = pygame.Rect(WIDTH - 320, 20, 140, 50)
        self.button_stop = pygame.Rect(WIDTH - 160, 20, 140, 50)
        
        # Pygame Setup
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("RL Self Driving Car - Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)
        
        # Generate Map
        self.map_surface = pygame.Surface((WIDTH, HEIGHT))
        self.map_surface.fill(BLACK)
        self.draw_track()
        
        self.car = None
        self.episode_count = 0
        self.best_distance = 0
        self.total_steps = 0

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
        # Draw a more interesting track
        points = [
            (150, 150), (400, 120), (700, 150), (950, 250),
            (1050, 450), (950, 650), (700, 700), (400, 680),
            (200, 550), (100, 350)
        ]
        pygame.draw.lines(self.map_surface, ROAD_COLOR, True, points, 120)

    def step(self, action):
        self.total_steps += 1
        if action == 0:  # Turn Left
            self.car.angle += 7
        elif action == 2:  # Turn Right
            self.car.angle -= 7
        # action == 1 is straight (no turn)
        
        self.car.update(self.map_surface)

        # IMPROVED REWARD SHAPING
        reward = 0
        done = False
        
        if not self.car.alive:
            # Penalty for crashing, but reward distance traveled
            reward = -100 + (self.car.distance_traveled * 0.1)
            done = True
        else:
            # Reward for staying alive and moving forward
            reward = 2
            
            # Bonus for distance traveled
            reward += self.car.distance_traveled * 0.01
            
            # Small penalty for turning (encourages smooth driving)
            if action != 1:
                reward -= 1
            
            # Bonus for keeping good distance from walls
            avg_distance = np.mean([r[1] for r in self.car.radars])
            if avg_distance > 50:
                reward += 2
        
        observation = np.array(self.car.get_data(), dtype=np.float32)
        
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Track best performance
        if self.car is not None and self.car.distance_traveled > self.best_distance:
            self.best_distance = self.car.distance_traveled
        
        self.car = Car()
        self.episode_count += 1
        
        # Initialize radars immediately
        self.car.radars.clear()
        for d in [-90, -45, 0, 45, 90]:
            self.car.check_radar(d, self.map_surface)
        
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

    def close(self):
        if self.render_mode == 'human':
            pygame.quit()


class TrainingCallback(BaseCallback):
    """Callback to monitor training progress"""
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode statistics
        if self.locals.get("dones")[0]:
            ep_reward = self.locals["infos"][0].get("episode", {}).get("r", 0)
            ep_length = self.locals["infos"][0].get("episode", {}).get("l", 0)
            
            if self.verbose > 0:
                print(f"Episode finished - Reward: {ep_reward:.2f}, Length: {ep_length}")
        
        return True
    
    
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


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("=" * 60)
    print("REINFORCEMENT LEARNING SELF-DRIVING CAR")
    print("=" * 60)
    print("\nTraining Configuration:")
    print("- Algorithm: PPO (Proximal Policy Optimization)")
    print("- Sensors: 5 distance sensors")
    print("- Actions: Left, Straight, Right")
    print("- Reward: Distance traveled + staying alive - crashes")
    print("\nTraining will run for 100,000 steps...")
    print("Watch the car learn to drive!\n")
    
    # Create environment
    env = TrackEnv(render_mode='human')

    # env = DummyVecEnv([lambda: env])
    
    # Initialize PPO with optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,      # Learning rate
        n_steps=2048,             # Steps per update
        batch_size=64,            # Minibatch size
        n_epochs=10,              # Optimization epochs
        gamma=0.99,               # Discount factor
        gae_lambda=0.95,          # GAE parameter
        clip_range=0.2,           # PPO clip range
        ent_coef=0.01,            # Entropy coefficient
        verbose=1,
        tensorboard_log="./ppo_car_tensorboard/"
    )
    
    # Create callback
    callback = StartStopCallback(env, verbose=1)
    
    # Train the model
    try:
        print("\nStarting training...")
        model.learn(
            total_timesteps=100000,
            callback=callback,
            progress_bar=True
        )
        
        # Save the trained model
        model.save("ppo_self_driving_car")
        print("\n✓ Model saved as 'ppo_self_driving_car.zip'")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        model.save("ppo_self_driving_car_interrupted")
        print("✓ Model saved as 'ppo_self_driving_car_interrupted.zip'")
    
    env.close()
    print("\nTraining complete!")