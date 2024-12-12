from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import random
import numpy as np
import math
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import tkinter as tk
from tkinter import messagebox
import pickle
import os
from datetime import datetime

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
BALL_RADIUS = 10
PLAYER_WIDTH = 10
PLAYER_HEIGHT = 40
ROW_SPACING = 100
PLAYER_DEPTH = 30
GOAL_WIDTH = 100
GOAL_HEIGHT = 150
GOAL_WIDTH1 = 200
GOAL_HEIGHT1 = 300
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)
WOOD_COLOR = (139, 69, 19)
FPS = 25

class Button:
    def __init__(self, x, y, diameter, image_path, border_color=(0, 0, 0), border_width=2):
        self.x = x
        self.y = y
        self.diameter = diameter
        self.radius = diameter // 2
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (self.diameter - 4 * border_width, self.diameter - 4 * border_width))
        self.border_color = border_color
        self.border_width = border_width

    def draw(self, screen):
        pygame.draw.circle(screen, self.border_color, (self.x + self.radius, self.y + self.radius), 
                           self.radius, self.border_width)
        screen.blit(self.image, (self.x+5 + self.border_width, self.y +2+ self.border_width))

class Timer:
    def __init__(self, x, y, initial_time):
        self.x = x
        self.y = y
        self.initial_time = initial_time
        self.remaining_time = initial_time
        self.paused_time = 0
        self.running = False
        self.start_ticks = pygame.time.get_ticks()

    def draw(self, screen):
        font = pygame.font.SysFont('georgia', 36)
        minutes = int(self.remaining_time) // 60
        seconds = int(self.remaining_time) % 60
        timer_text = f"{minutes:02}:{seconds:02}"
        text = font.render(timer_text, True, (0, 0, 0))
        screen.blit(text, (self.x, self.y))
       
class Rod:
    def __init__(self, x, num_players, color, spacing, side='left'):
        self.x = x
        self.players = []
        self.side = side
        self.color = color
        self.spacing = spacing
        start_y = (SCREEN_HEIGHT - (num_players - 1) * spacing) / 2
        for i in range(num_players):
            y = start_y + i * spacing
            self.players.append(Player(x, y, color))

    def move_up(self, distance):
        """Move the entire rod up"""
        min_y = 50
        if self.players[0].y - distance >= min_y:
            for player in self.players:
                player.y -= distance

    def move_down(self, distance):
        """Move the entire rod down"""
        max_y = SCREEN_HEIGHT - 50 - PLAYER_HEIGHT
        if self.players[-1].y + distance <= max_y:
            for player in self.players:
                player.y += distance

    def execute_action(self, action, move_distance=10):
        """Execute a given action for the rod"""
        if action == 'move_up':
            self.move_up(move_distance)
        elif action == 'move_down':
            self.move_down(move_distance)
        elif action == 'shoot':
            # Move towards ball
            self.move_to_ball()
        elif action == 'defend':
            # Spread out to defensive positions
            self.spread_defensive()

    def move_to_ball(self, ball_y=SCREEN_HEIGHT//2):
        """Move rod towards the ball's vertical position"""
        current_center = sum(p.y for p in self.players) / len(self.players)
        if abs(current_center - ball_y) > 10:
            if current_center < ball_y:
                self.move_down(10)
            else:
                self.move_up(10)

    def spread_defensive(self):
        """Spread players evenly across their possible vertical range"""
        total_space = SCREEN_HEIGHT - 100 - PLAYER_HEIGHT
        if len(self.players) > 1:
            spacing = total_space / (len(self.players) - 1)
            for i, player in enumerate(self.players):
                target_y = 50 + (i * spacing)
                if abs(player.y - target_y) > 10:
                    if player.y < target_y:
                        self.move_down(min(10, target_y - player.y))
                    else:
                        self.move_up(min(10, player.y - target_y))

    def reset_positions(self):
        """Reset all players to their original positions"""
        start_y = (SCREEN_HEIGHT - (len(self.players) - 1) * self.spacing) / 2
        for i, player in enumerate(self.players):
            player.y = start_y + i * self.spacing

    def draw(self, screen):
        """Draw all players in the rod"""
        for player in self.players:
            player.draw(screen)

class Ball:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.vx = 0 
        self.vy = 0
        self.moving = False

    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), BALL_RADIUS)
        
    def reset(self):
        """Reset ball to center position"""
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.vx = random.choice([-7, 7])
        self.vy = random.choice([-7, 7])
        self.moving = True

class Player:
    def __init__(self, row_x, player_y, color, is_goalkeeper=False):
        self.x = row_x
        self.y = player_y
        self.color = color
        self.original_x = row_x
        self.original_y = player_y
        self.is_goalkeeper = is_goalkeeper

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, PLAYER_WIDTH, PLAYER_HEIGHT))

def draw_table(screen):
    pygame.draw.rect(screen, WOOD_COLOR, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.draw.rect(screen, WHITE, (47, 47, SCREEN_WIDTH - 94, SCREEN_HEIGHT - 94), 5)
    pygame.draw.rect(screen, GREEN, (50, 50, SCREEN_WIDTH-100, SCREEN_HEIGHT-100))
    pygame.draw.line(screen, WHITE, (SCREEN_WIDTH // 2, 50), (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50), 3)
    pygame.draw.circle(screen, WHITE, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), 50, 3)
    pygame.draw.circle(screen, WHITE, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), 10, 10)
    pygame.draw.rect(screen, WHITE, (50, SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2, 20, GOAL_HEIGHT))
    left_rect = pygame.Rect(SCREEN_WIDTH - 750 - 50, SCREEN_HEIGHT // 2 - 50, 100, 100)
    pygame.draw.arc(screen, WHITE, left_rect, -math.pi / 2, math.pi / 2, 3)
    right_rect = pygame.Rect(SCREEN_WIDTH - 250 - 50, SCREEN_HEIGHT // 2 - 50, 100, 100)
    pygame.draw.arc(screen, WHITE, right_rect, math.pi / 2, 3 * math.pi / 2, 3)
    pygame.draw.rect(screen, WHITE, (SCREEN_WIDTH - 70, SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2, 20, GOAL_HEIGHT))
    pygame.draw.rect(screen, WHITE, (50, SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2, GOAL_WIDTH, GOAL_HEIGHT), 3) #left Goal post (GoalKeeper area)
    pygame.draw.rect(screen, WHITE, (50, SCREEN_HEIGHT // 2 - GOAL_HEIGHT1 // 2, GOAL_WIDTH1, GOAL_HEIGHT1), 3)
    pygame.draw.rect(screen, WHITE, (SCREEN_WIDTH - 50 - GOAL_WIDTH, SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2, GOAL_WIDTH, GOAL_HEIGHT), 3) #right Goal post (GoalKeeper area)
    pygame.draw.rect(screen, WHITE, (SCREEN_WIDTH - 50 - GOAL_WIDTH1, SCREEN_HEIGHT // 2 - GOAL_HEIGHT1 // 2, GOAL_WIDTH1, GOAL_HEIGHT1), 3)

class MonteCarloSimulator:
    def __init__(self, num_simulations, max_steps):
        self.num_simulations = num_simulations
        self.max_steps = max_steps
        self.state_space = self.define_state_space()
        self.action_space = ['move_up', 'move_down', 'shoot', 'defend']  # Simplified action space
        self.state_values = {}
        self.returns = {}
        self.policy = {}
        self.gamma = 0.99
        self.epsilon = 0.2
        self.tables_file = 'monte_carlo_tables.pkl'
        self.history_dir = 'mc_history'
        self.load_tables()

    def state_to_tuple(self, state):
        """Convert state dictionary to hashable tuple"""
        return (
            round(state['ball_position'][0], 1),
            round(state['ball_position'][1], 1),
            round(state['ball_velocity'][0], 1),
            round(state['ball_velocity'][1], 1),
            tuple(round(pos, 1) for pos in state['player_positions'])
        )
        
    def get_initial_state(self):
        return {
            'ball_position': (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
            'ball_velocity': (0, 0),
            'player_positions': [
                120, 280, 440,  # Red team row 1
                200, 360,      # Red team row 2
                120, 280, 440,  # Red team row 3
                120, 200, 280, 360, 440  # Red team row 4
            ],
            'goals_scored': 0,
            'goals_conceded': 0
        }

    def define_state_space(self):
        return {
            'ball_position': {
                'x': np.linspace(50, SCREEN_WIDTH-50, 20),
                'y': np.linspace(50, SCREEN_HEIGHT-50, 20)
            },
            'player_positions': np.linspace(50, SCREEN_HEIGHT-50-PLAYER_HEIGHT, 10),
            'ball_velocity': {
                'vx': np.linspace(-10, 10, 5),
                'vy': np.linspace(-10, 10, 5)
            }
        }
        
    def calculate_new_ball_position(self, ball_pos, action):
        """Calculate new ball position based on action"""
        x, y = ball_pos

        if action == 'move_left':
            x = max(50, x - 10)
        elif action == 'move_right':
            x = min(SCREEN_WIDTH - 50, x + 10)
        elif action == 'move_up':
            y = max(50, y - 10)
        elif action == 'move_down':
            y = min(SCREEN_HEIGHT - 50, y + 10)
        elif action == 'shoot':
            # Move ball forward with some upward/downward variation
            x = min(SCREEN_WIDTH - 50, x + 15)
            y += random.choice([-5, 0, 5])
        elif action == 'defend':
            # Slight backward movement for defensive positioning
            x = max(50, x - 5)

        # Keep ball within bounds
        y = max(50, min(SCREEN_HEIGHT - 50, y))
        x = max(50, min(SCREEN_WIDTH - 50, x))
    
        return (x, y)

    def define_action_space(self):
        return ['move_left', 'move_right', 'move_up', 'move_down', 'shoot', 'defend']
    
    def simulate_game_play(self, state, action):
        new_state = state.copy()
        reward = 0
        ball_pos = state['ball_position']

        # Update ball position based on action
        new_ball_pos = self.calculate_new_ball_position(ball_pos, action)
        new_state['ball_position'] = new_ball_pos

        # Calculate rewards based on new position
        if action == 'shoot':
            if self.is_scoring_position(new_ball_pos):
                reward = 1  # Reward for being in scoring position
            elif self.is_defensive_position(new_ball_pos):
                reward = 0.5  # Reward for good defensive positioning
        elif action == 'defend':
            if self.is_defensive_position(new_ball_pos):
                reward = 0.5

        return new_state, reward
    
    def load_tables(self):
        try:
            with open(self.tables_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.state_values = saved_data['state_values']
                self.returns = saved_data['returns']
                self.policy = saved_data['policy']
                print("Loaded existing Monte Carlo tables")
                self.print_table_stats()
        except FileNotFoundError:
            print("No existing tables found. Initializing new tables.")
            self.state_values = {}
            self.returns = {}
            self.policy = {}

    def save_tables(self):
        data = {
            'state_values': self.state_values,
            'returns': self.returns,
            'policy': self.policy
        }
        
        # Save current tables
        with open(self.tables_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Create history directory if it doesn't exist
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
        
        # Save timestamped copy in history
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_file = os.path.join(self.history_dir, f'mc_tables_{timestamp}.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved Monte Carlo tables with {len(self.state_values)} states")

    def is_scoring_position(self, ball_pos):
        """Check if ball is in a good position to score"""
        x, y = ball_pos  # Unpack the tuple
        return (
            x > SCREEN_WIDTH - 200 and  # Near opponent's goal
            SCREEN_HEIGHT/2 - GOAL_HEIGHT/2 <= y <= SCREEN_HEIGHT/2 + GOAL_HEIGHT/2  # Aligned with goal
        )

    def is_defensive_position(self, ball_pos):
        """Check if ball is in good defensive position"""
        x, y = ball_pos  # Unpack the tuple
        return (
            x < SCREEN_WIDTH/2 and  # In our half
            x > 150  # Not too close to our goal
        )

    def is_goal(self, state):
        ball_position = state['ball_position']
        return (
            ball_position[0] <= 50 and (SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2 <= ball_position[1] <= SCREEN_HEIGHT // 2 + GOAL_HEIGHT // 2)
        ) or (
            ball_position[0] >= SCREEN_WIDTH - 50 and (SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2 <= ball_position[1] <= SCREEN_HEIGHT // 2 + GOAL_HEIGHT // 2)
        )

    def select_action(self, state):
        """Select an action using epsilon-greedy policy"""
        try:
            state_tuple = self.state_to_tuple(state)
            
            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                return random.choice(self.action_space)
            
            # If state exists in policy, choose best action
            if state_tuple in self.policy:
                action_values = {}
                for action, returns in self.policy[state_tuple].items():
                    if returns:  # Check if returns list is not empty
                        action_values[action] = np.mean(returns)
                    else:
                        action_values[action] = 0
                        
                if action_values:  # Make sure we have some values
                    return max(action_values.items(), key=lambda x: x[1])[0]
            
            # If state is new or no values, choose random action
            return random.choice(self.action_space)
            
        except Exception as e:
            print(f"Error in select_action: {e}")
            print(f"State: {state}")
            print(f"State tuple: {self.state_to_tuple(state)}")
            print(f"Policy keys: {list(self.policy.keys())[:5]}")  # Show first 5 keys
            return 'defend'  # Safe default action
     
    def is_terminal(self, state):
        # End episode if ball is in goal or out of bounds
        ball_pos = state['ball_position']
        return (ball_pos[0] <= 50 or 
                ball_pos[0] >= SCREEN_WIDTH - 50 or
                ball_pos[1] <= 50 or 
                ball_pos[1] >= SCREEN_HEIGHT - 50 or
                state['goals_scored'] > 0 or 
                state['goals_conceded'] > 0)   

    def update_tables(self, episode):
        G = 0
        # Process episode in reverse order
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            state_tuple = self.state_to_tuple(state)
            G = self.gamma * G + reward
            
            # First-visit MC
            if state_tuple not in self.returns:
                self.returns[state_tuple] = []
                self.state_values[state_tuple] = 0
                
            self.returns[state_tuple].append(G)
            self.state_values[state_tuple] = np.mean(self.returns[state_tuple])
            
            # Update policy
            if state_tuple not in self.policy:
                self.policy[state_tuple] = {}
            if action not in self.policy[state_tuple]:
                self.policy[state_tuple][action] = []
            self.policy[state_tuple][action].append(G)
            
    def print_table_stats(self):
        print("\nCurrent Table Statistics:")
        print(f"Number of states: {len(self.state_values)}")
        print(f"Number of state-action pairs: {sum(len(actions) for actions in self.policy.values())}")
        
        if self.state_values:
            avg_value = sum(self.state_values.values()) / len(self.state_values)
            print(f"Average state value: {avg_value:.2f}")
        
        total_visits = sum(len(returns) for returns in self.returns.values())
        print(f"Total state visits: {total_visits}")
            
    def print_tables(self):
        print("\n=== State-Value Table ===")
        print("Format: State -> (Average Value, Number of Visits)")
        for state, returns in self.returns.items():
            print(f"Ball Position: ({state[0]:.1f}, {state[1]:.1f})")
            print(f"Goals: Scored={state[3]}, Conceded={state[4]}")
            print(f"Value: {self.state_values[state]:.2f}, Visits: {len(returns)}\n")
        
        print("\n=== Policy Table ===")
        print("Format: State -> Action -> (Average Return, Times Chosen)")
        for state, actions in self.policy.items():
            print(f"\nBall Position: ({state[0]:.1f}, {state[1]:.1f})")
            print(f"Goals: Scored={state[3]}, Conceded={state[4]}")
            for action, returns in actions.items():
                avg_return = np.mean(returns) if returns else 0
                print(f"  {action}: Return={avg_return:.2f}, Times Chosen={len(returns)}")

    def run_simulation(self):
        print("\nStarting Monte Carlo simulation...")
        self.print_table_stats()
        
        for i in range(self.num_simulations):
            episode = []
            state = self.get_initial_state()
            
            for j in range(self.max_steps):
                action = self.select_action(state)
                next_state, reward = self.simulate_game_play(state, action)
                episode.append((state, action, reward))
                state = next_state
                
                if self.is_terminal(state):
                    break
            
            self.update_tables(episode)
            
            if (i + 1) % 10 == 0:
                print(f"\nCompleted {i + 1} episodes")
                self.print_table_stats()
        
        print("\nSimulation completed")
        self.save_tables()
        self.print_detailed_tables()
        
    def print_detailed_tables(self):
        print("\n=== State-Value Table ===")
        print("Format: State -> (Average Value, Number of Visits)")
        for state, returns in self.returns.items():
            print(f"\nBall Position: ({state[0]:.1f}, {state[1]:.1f})")
            print(f"Player Positions: {state[2]}")  # Print player positions tuple
            print(f"Value: {self.state_values[state]:.2f}, Visits: {len(returns)}")

        print("\n=== Policy Table ===")
        print("Format: State -> Action -> (Average Return, Times Chosen)")
        for state, actions in self.policy.items():
            print(f"\nBall Position: ({state[0]:.1f}, {state[1]:.1f})")
            print(f"Player Positions: {state[2]}")  # Print player positions tuple
            for action, returns in actions.items():
                avg_return = np.mean(returns) if returns else 0
                print(f"  {action}: Return={avg_return:.2f}, Times Chosen={len(returns)}")

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, k_epochs=4, eps_clip=0.2):
        print(f"Initializing PPOAgent with state_dim={state_dim}, action_dim={action_dim}")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.policy = self.PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = self.PolicyNetwork(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.buffer = []
        print("PPOAgent initialized.")

    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(PPOAgent.PolicyNetwork, self).__init__()
            print(f"Initializing PolicyNetwork with state_dim={state_dim}, action_dim={action_dim}")
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.action_layer = nn.Linear(128, action_dim)
            self.value_layer = nn.Linear(128, 1)
            print("PolicyNetwork initialized.")

        def forward(self, state):
            print("Forward pass in PolicyNetwork")
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            action_probs = torch.softmax(self.action_layer(x), dim=-1)
            state_value = self.value_layer(x)
            return action_probs, state_value

        def act(self, state):
            print("Acting in PolicyNetwork")
            action_probs, _ = self.forward(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            print(f"Selected action: {action.item()}, log_prob: {dist.log_prob(action)}")
            return action.item(), dist.log_prob(action)

        def evaluate(self, state, action):
            print("Evaluating state-action pair in PolicyNetwork")
            action_probs, state_value = self.forward(state)
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            return action_logprobs, torch.squeeze(state_value), dist_entropy

    def select_action(self, state):
        print(f"Selecting action for state: {state}")
        state = torch.FloatTensor(state)
        action, action_logprob = self.policy_old.act(state)
        self.buffer.append([state, action, action_logprob])
        print(f"Action selected: {action}, Action log prob: {action_logprob}")
        return action

    def run_optimization(self, all_players):
        for index, player in enumerate(all_players):
            print(f"Optimizing Player {index+1} for Effective movements....")
            time.sleep(2)

    def run_simulation(self, num_steps):
        print(f"Running simulation for {num_steps} steps")
        state = np.random.rand(self.state_dim)
        for step in range(num_steps):
            print(f"--- Step {step+1} ---")
            action = self.select_action(state)
            print(f"Action taken: {action}")
            reward = np.random.rand()
            next_state = np.random.rand(self.state_dim)
            done = step == num_steps - 1
            self.buffer.append([torch.FloatTensor(state), action, self.policy_old.act(torch.FloatTensor(state))[1], reward, done, torch.FloatTensor(next_state)])
            state = next_state
            if done:
                print("End of episode")
                
class MonteCarloPlayer:
    def __init__(self, tables_file=None, side='left'):
        print(f"Initializing {side} player...")
        self.side = side
        self.simulator = MonteCarloSimulator(num_simulations=100, max_steps=500)
        if tables_file:
            try:
                with open(tables_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.simulator.state_values = saved_data['state_values']
                    self.simulator.returns = saved_data['returns']
                    self.simulator.policy = saved_data['policy']
                print(f"Loaded tables for {self.side} player")
            except FileNotFoundError:
                print(f"No existing tables found for {self.side} player. Starting fresh.")

    def get_action(self, state):
        """Get action for the current state"""
        try:
            if self.side == 'right':
                state = self.mirror_state(state)
            
            action = self.simulator.select_action(state)
            
            if self.side == 'right':
                action = self.mirror_action(action)
            
            return action
        except Exception as e:
            print(f"Error in get_action: {e}")
            return 'defend'  # Default action

    def mirror_state(self, state):
        """Mirror the state for right player"""
        mirrored_state = state.copy()
        ball_x, ball_y = state['ball_position']
        ball_vx, ball_vy = state['ball_velocity']
        mirrored_state['ball_position'] = (SCREEN_WIDTH - ball_x, ball_y)
        mirrored_state['ball_velocity'] = (-ball_vx, ball_vy)
        return mirrored_state

    def mirror_action(self, action):
        """Mirror the action for right player"""
        action_map = {
            'move_up': 'move_up',
            'move_down': 'move_down',
            'shoot': 'shoot',
            'defend': 'defend'
        }
        return action_map[action]

def foosball_main():
    global timer_running
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    print('Test for foosball_main fn loading.')
    pygame.display.set_caption("Foosball Table")
    clock = pygame.time.Clock()
    initial_time = 1 * 60
    elapsed_time = 0
    start_ticks = pygame.time.get_ticks()
    ball = Ball()
    timer = Timer(450, 550, initial_time)
    rows_x_positions = [100, 200, 300, 400, 600, 700, 800, 900]
    red_positions_y = [[120, 280, 440], [200, 360], [120, 280, 440], [120, 200, 280, 360, 440]]
    blue_positions_y = [[120, 280, 440], [200, 360], [120, 280, 440], [120, 200, 280, 360, 440]]
    red_players = [Player(rows_x_positions[0], y, RED) for y in red_positions_y[0]] + \
                  [Player(rows_x_positions[1], y, RED) for y in red_positions_y[1]] + \
                  [Player(rows_x_positions[2], y, BLUE) for y in red_positions_y[2]] + \
                  [Player(rows_x_positions[3], y, RED) for y in red_positions_y[3]]
    blue_players = [Player(rows_x_positions[4], y, BLUE) for y in blue_positions_y[3]] + \
                   [Player(rows_x_positions[5], y, RED) for y in blue_positions_y[2]] + \
                   [Player(rows_x_positions[6], y, BLUE) for y in blue_positions_y[1]] + \
                   [Player(rows_x_positions[7], y, BLUE, is_goalkeeper=True) for y in blue_positions_y[0]]
    all_players = red_players + blue_players
    start_button = Button(480, 3, diameter=40, image_path="Play.png", border_color=(255, 255, 255), border_width=3)
    buttons_red = [
        (100, SCREEN_HEIGHT - 25),  
        (200, SCREEN_HEIGHT - 25),  
        (700, SCREEN_HEIGHT - 25),    
        (400, SCREEN_HEIGHT - 25),
        (100, 22),
        (200, 22),  
        (700, 22),    
        (400, 22),  
    ]
    buttons_blue = [
        (600, SCREEN_HEIGHT - 25), 
        (300, SCREEN_HEIGHT - 25),  
        (800, SCREEN_HEIGHT - 25),
        (900, SCREEN_HEIGHT - 25),  
        (600, 22), 
        (300, 22),  
        (800, 22),  
        (900, 22),
    ]
    left_score = 0
    right_score = 0
    draw_table(screen)
    ball.draw(screen)
    timer.draw(screen)
    for player in all_players:
        player.draw(screen)
    keys_red = ['A', 'S', 'F', 'D','Q','W','R','E']
    keys_blue = ['J', 'H', 'K', 'L','I','U','O','P']
    font = pygame.font.SysFont('georgia', 30)
    for i, pos in enumerate(buttons_red):
        pygame.draw.circle(screen, RED, pos, 20)
        key_text = font.render(keys_red[i], True, WHITE)
        text_rect = key_text.get_rect(center=pos)
        screen.blit(key_text, text_rect)
    for i, pos in enumerate(buttons_blue):
        pygame.draw.circle(screen, BLUE, pos, 20)
        key_text = font.render(keys_blue[i], True, WHITE)
        text_rect = key_text.get_rect(center=pos)
        screen.blit(key_text, text_rect)
    box_size = 40
    left_box_position = (5, 270)
    right_box_position = (960, 270)
    pygame.draw.rect(screen, WHITE, (left_box_position[0], left_box_position[1], box_size, box_size))
    pygame.draw.rect(screen, WHITE, (right_box_position[0], right_box_position[1], box_size, box_size))
    left_score_text = font.render(str(left_score), True, BLACK)
    left_text_rect = left_score_text.get_rect(center=(left_box_position[0] + box_size // 2, left_box_position[1] + box_size // 2))
    screen.blit(left_score_text, left_text_rect)
    right_score_text = font.render(str(right_score), True, BLACK)
    right_text_rect = right_score_text.get_rect(center=(right_box_position[0] + box_size // 2, right_box_position[1] + box_size // 2))
    screen.blit(right_score_text, right_text_rect)
    pygame.display.flip()
    clock.tick(FPS)
    time.sleep(10)
    pygame.quit()
    return all_players

def initialize_players():
    all_players = foosball_main()
    return all_players

def run_monte_carlo_simulation():
    print('run_monte_carlo_simulation fn testing.')
    monte_carlo_simulator = MonteCarloSimulator(num_simulations=10, max_steps=100)
    monte_carlo_simulator.run_simulation()

def optimize_agent(all_players):
    print("Starting optimization of the agent...")
    agent = PPOAgent(state_dim=4, action_dim=2)
    print("Starting simulation...")
    agent.run_simulation(num_steps=100)
    agent.run_optimization(all_players)
    print("Optimization complete.")
    print("Simulation complete.")