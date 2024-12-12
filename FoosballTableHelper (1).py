from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import math
import numpy as np
import random
import tkinter as tk
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tkinter import messagebox
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
FPS = 30
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

    def is_clicked(self, mouse_pos):
        dist_x = mouse_pos[0] - (self.x + self.radius)
        dist_y = mouse_pos[1] - (self.y + self.radius)
        return dist_x ** 2 + dist_y ** 2 <= self.radius ** 2

class Timer:
    def __init__(self, x, y, initial_time):
        self.x = x
        self.y = y
        self.initial_time = initial_time
        self.remaining_time = initial_time
        self.paused_time = 0
        self.running = False
        self.start_ticks = pygame.time.get_ticks()

    def start(self):
        if not self.running:
            self.start_ticks = pygame.time.get_ticks() # - self.paused_time
            self.running = True

    def pause(self):
        if self.running:
            self.paused_time = pygame.time.get_ticks() - self.start_ticks
            self.running = False

    def reset(self):
        self.start_ticks = pygame.time.get_ticks()
        self.paused_time = 0
        self.remaining_time = self.initial_time
        self.running = False

    def update(self):
        if self.running:
            elapsed_time = (pygame.time.get_ticks() - self.start_ticks) / 1000
            self.remaining_time = max(self.initial_time - elapsed_time, 0)

    def time_up(self):
        return self.remaining_time <= 0

    def draw(self, screen):
        font = pygame.font.SysFont('georgia', 36)
        minutes = int(self.remaining_time) // 60
        seconds = int(self.remaining_time) % 60
        timer_text = f"{minutes:02}:{seconds:02}"
        text = font.render(timer_text, True, (0, 0, 0))
        screen.blit(text, (self.x, self.y))

class Ball:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.vx = 0 
        self.vy = 0
        self.moving = False

    def start(self, timer):
        global timer_running
        timer_running = True
        timer.start()
        self.vx = 5
        self.vy = random.choice([-5, 5])
        self.moving = True

    def move(self):
        if not self.moving:
            return
        # self.x  += self.vx
        # self.y += self.vy
        # if self.y - BALL_RADIUS <= 50 or self.y + BALL_RADIUS >= SCREEN_HEIGHT - 50:
        #     self.vy = -self.vy
        # if self.x - BALL_RADIUS <= 50 or self.x + BALL_RADIUS >= SCREEN_WIDTH - 50:
        #     self.vx = -self.vx
        if self.moving:
            self.x += self.vx
            self.y += self.vy

    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), BALL_RADIUS)

class Player:
    def __init__(self, row_x, player_y, color, is_goalkeeper=False):
        self.x = row_x
        self.y = player_y
        self.color = color
        self.original_x = row_x
        self.original_y = player_y
        self.is_goalkeeper = is_goalkeeper

    def reset_position(self):
        self.x = self.original_x
        self.y = self.original_y

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, PLAYER_WIDTH, PLAYER_HEIGHT))

def check_collision(ball, player):
    if player.x <= ball.x <= player.x + PLAYER_WIDTH:
        if player.y <= ball.y <= player.y + PLAYER_HEIGHT:
            ball.vx = -ball.vx
            if ball.vx > 0:
                ball.x = player.x + PLAYER_WIDTH + BALL_RADIUS
            else:
                ball.x = player.x - BALL_RADIUS

def check_goal(ball, timer):
    if ball.x - BALL_RADIUS <= 50 and (SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2 <= ball.y <= SCREEN_HEIGHT // 2 + GOAL_HEIGHT // 2):
        timer.pause()
        return "Right"
    elif ball.x + BALL_RADIUS >= SCREEN_WIDTH - 70 and (SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2 <= ball.y <= SCREEN_HEIGHT // 2 + GOAL_HEIGHT // 2):
        timer.pause()
        return "Left"
    return None

def print_instructions():
    print("To Start the Game: Press \"Enter\" (or) Click \"Play\" icon on the window")
    print("Keys for Red Teams:")
    print("To Move Up: \" Q,W,E,R\"")
    print("To Move Down: \" A,S,D,F\"")
    print("Keys for Blue Teams:")
    print("To Move Up: \" U,I,O,P\"")
    print("To Move Down: \" H,J,K,L\"")
    print("Have a great time playing!")

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
        self.action_space = self.define_action_space()
        self.returns = {}  # Store state-value estimates
        self.policy = {}   # Store policy

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

    def define_action_space(self):
        return {
            'move_up': -50,
            'move_down': 50,
            'stay': 0,
            'shoot': {'power': np.linspace(5, 15, 3)}
        }

    def simulate_episode(self):
        state = self.get_initial_state()
        episode = []
        stats = {'goals_scored': 0, 'goals_conceded': 0, 'successful_actions': 0}
        
        for step in range(self.max_steps):
            action = self.select_action(state)
            next_state, reward = self.simulate_step(state, action)
            
            episode.append((state, action, reward))
            stats = self.update_stats(stats, state, action, reward)
            
            if self.is_terminal(next_state):
                break
                
            state = next_state
            
        return episode, stats

    def simulate_step(self, state, action):
        next_state = state.copy()
        reward = 0
        
        # Ball physics simulation
        if action['type'] == 'shoot':
            ball_pos = self.update_ball_position(state['ball_position'], action['power'])
            next_state['ball_position'] = ball_pos
            
            if self.is_goal(ball_pos):
                reward = 1
                next_state['goals_scored'] += 1
            
        # Player movement
        elif action['type'] in ['move_up', 'move_down']:
            new_pos = state['player_positions'][action['player']] + self.action_space[action['type']]
            if 50 <= new_pos <= SCREEN_HEIGHT-50-PLAYER_HEIGHT:
                next_state['player_positions'][action['player']] = new_pos
                if self.is_defensive_position_improved(next_state):
                    reward = 0.1

        return next_state, reward

    def update_policy(self, episodes):
        for episode in episodes:
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                if state not in self.returns:
                    self.returns[state] = []
                self.returns[state].append(G)
                
                # Update policy to choose action with highest average return
                if state not in self.policy:
                    self.policy[state] = {}
                if action not in self.policy[state]:
                    self.policy[state][action] = []
                self.policy[state][action].append(G)

    def run_simulation(self):
        episodes = []
        total_stats = {
            'wins': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'successful_actions': 0
        }

        print(f"Starting Monte Carlo simulation with {self.num_simulations} episodes...")
        
        for i in range(self.num_simulations):
            episode, stats = self.simulate_episode()
            episodes.append(episode)
            
            # Update statistics
            total_stats['goals_scored'] += stats['goals_scored']
            total_stats['goals_conceded'] += stats['goals_conceded']
            total_stats['successful_actions'] += stats['successful_actions']
            total_stats['wins'] += 1 if stats['goals_scored'] > stats['goals_conceded'] else 0
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1} episodes")

        # Calculate metrics
        metrics = self.calculate_metrics(total_stats)
        self.update_policy(episodes)
        
        return metrics, self.policy

    def calculate_metrics(self, stats):
        total_games = self.num_simulations
        total_actions = self.num_simulations * self.max_steps
        
        return {
            'win_rate': (stats['wins'] / total_games) * 100,
            'goal_differential': ((stats['goals_scored'] - stats['goals_conceded']) / 
                                max(stats['goals_scored'], 1)) * 100,
            'strategic_adaptability': (stats['successful_actions'] / total_actions) * 100
        }

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, k_epochs=4, eps_clip=0.2):
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
        self.metrics = {
            'win_rate': [],
            'goal_differential': [],
            'strategic_adaptability': []
        }

    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(PPOAgent.PolicyNetwork, self).__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.action_layer = nn.Linear(128, action_dim)
            self.value_layer = nn.Linear(128, 1)

        def forward(self, state):
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            action_probs = torch.softmax(self.action_layer(x), dim=-1)
            state_value = self.value_layer(x)
            return action_probs, state_value

        def act(self, state):
            action_probs, _ = self.forward(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action)

        def evaluate(self, state, action):
            action_probs, state_value = self.forward(state)
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            return action_logprobs, torch.squeeze(state_value), dist_entropy

        def train_step(self, state, action):
            self.train()
            action_probs, state_value = self.forward(state)
            loss = self.compute_loss(action_probs, action)
            loss.backward()
            return loss.item()

    def initialize_from_mc(self, mc_policy):
        state_actions = []
        for state, actions in mc_policy.items():
            best_action = max(actions.items(), key=lambda x: np.mean(x[1]))[0]
            state_actions.append((self.state_to_tensor(state), best_action))
        
        for state_tensor, action in state_actions:
            self.policy.train_step(state_tensor, action)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action, action_logprob = self.policy_old.act(state)
        self.buffer.append([state, action, action_logprob])
        return action

    def select_action_with_mc(self, state, mc_policy):
        if np.random.random() < 0.2:  # Exploration rate
            state_tensor = torch.FloatTensor(state)
            return self.select_action(state_tensor)
        else:
            return self.get_mc_action(state, mc_policy)

    def update_policy(self, states, actions, rewards):
        # Monte Carlo estimate of rewards
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_states = torch.stack(states).detach()
        old_actions = torch.stack(actions).detach()
        old_logprobs = torch.stack([b[2] for b in self.buffer]).detach()

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def run_optimization(self, all_players):
        mc_sim = MonteCarloSimulator(num_simulations=100, max_steps=500)
        mc_metrics, mc_policy = mc_sim.run_simulation()
        
        self.initialize_from_mc(mc_policy)
        
        print("Starting PPO optimization...")
        for episode in range(self.num_episodes):
            states, actions, rewards = self.collect_trajectory(mc_policy)
            self.update_policy(states, actions, rewards)
            episode_metrics = self.evaluate_episode()
            self.update_metrics(episode_metrics)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1} completed")
                print(f"Current metrics: {episode_metrics}")

def handle_key_presses(red_players):
    keys = pygame.key.get_pressed()

    if keys[pygame.K_q]:
        for player in red_players[:3]:
            player.y -= 50
            if player.y < 50:
                player.y = 50

    if keys[pygame.K_a]:
        for player in red_players[:3]:
            player.y += 50
            if player.y + PLAYER_HEIGHT > SCREEN_HEIGHT - 50:
                player.y = SCREEN_HEIGHT - 50 - PLAYER_HEIGHT

    if not keys[pygame.K_a] and not keys[pygame.K_q]:
        for player in red_players[:3]:
            player.reset_position()

    if keys[pygame.K_w]:
        for player in red_players[3:5]:
            player.y -= 130
            if player.y < 50:
                player.y = 50

    if keys[pygame.K_s]:
        for player in red_players[3:5]:
            player.y += 130
            if player.y + PLAYER_HEIGHT > SCREEN_HEIGHT - 50:
                player.y = SCREEN_HEIGHT - 50 - PLAYER_HEIGHT

    if not keys[pygame.K_s] and not keys[pygame.K_w]:
        for player in red_players[3:5]:
            player.reset_position()

    if keys[pygame.K_e]:
        for player in red_players[8:13]:
            player.y -= 50
            if player.y < 50:
                player.y = 50

    if keys[pygame.K_d]:
        for player in red_players[8:13]:
            player.y += 50
            if player.y + PLAYER_HEIGHT > SCREEN_HEIGHT - 50:
                player.y = SCREEN_HEIGHT - 50 - PLAYER_HEIGHT

    if not keys[pygame.K_d] and not keys[pygame.K_e]:
        for player in red_players[8:13]:
            player.reset_position()

    if keys[pygame.K_u]:
        for player in red_players[5:8]:
            player.y -= 50
            if player.y < 50:
                player.y = 50

    if keys[pygame.K_h]:
        for player in red_players[5:8]:
            player.y += 50
            if player.y + PLAYER_HEIGHT > SCREEN_HEIGHT - 50:
                player.y = SCREEN_HEIGHT - 50 - PLAYER_HEIGHT

    if not keys[pygame.K_h] and not keys[pygame.K_u]:
        for player in red_players[5:8]:
            player.reset_position()

def handle_key_presses1(blue_players):
    keys = pygame.key.get_pressed()

    if keys[pygame.K_i]:
        for player in blue_players[:5]:
            player.y -= 50
            if player.y < 50:
                player.y = 50

    if keys[pygame.K_j]:
        for player in blue_players[:5]:
            player.y += 50
            if player.y + PLAYER_HEIGHT > SCREEN_HEIGHT - 50:
                player.y = SCREEN_HEIGHT - 50 - PLAYER_HEIGHT

    if not keys[pygame.K_j] and not keys[pygame.K_i]:
        for player in blue_players[:5]:
            player.reset_position()

    if keys[pygame.K_r]:
        for player in blue_players[5:8]:
            player.y -= 50
            if player.y < 50:
                player.y = 50

    if keys[pygame.K_f]:
        for player in blue_players[5:8]:
            player.y += 50
            if player.y + PLAYER_HEIGHT > SCREEN_HEIGHT - 50:
                player.y = SCREEN_HEIGHT - 50 - PLAYER_HEIGHT

    if not keys[pygame.K_f] and not keys[pygame.K_r]:
        for player in blue_players[5:8]:
            player.reset_position()

    if keys[pygame.K_o]:
        for player in blue_players[8:10]:
            player.y -= 130
            if player.y < 50:
                player.y = 50

    if keys[pygame.K_k]:
        for player in blue_players[8:10]:
            player.y += 130
            if player.y + PLAYER_HEIGHT > SCREEN_HEIGHT - 50:
                player.y = SCREEN_HEIGHT - 50 - PLAYER_HEIGHT

    if not keys[pygame.K_k] and not keys[pygame.K_o]:
        for player in blue_players[8:10]:
            player.reset_position()

    if keys[pygame.K_p]:
        for player in blue_players[10:13]:
            player.y -= 50
            if player.y < 50:
                player.y = 50

    if keys[pygame.K_l]:
        for player in blue_players[10:13]:
            player.y += 50
            if player.y + PLAYER_HEIGHT > SCREEN_HEIGHT - 50:
                player.y = SCREEN_HEIGHT - 50 - PLAYER_HEIGHT

    if not keys[pygame.K_l] and not keys[pygame.K_p]:
        for player in blue_players[10:13]:
            player.reset_position()

def simulate_foosball():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Foosball Simulation")
    clock = pygame.time.Clock()
    
    # Initialize game components
    ball = Ball()
    timer = Timer(450, 550, 60)  # 1 minute game
    left_score = 0
    right_score = 0
    
    # Initialize Monte Carlo and PPO
    mc_simulator = MonteCarloSimulator(num_simulations=100, max_steps=1000)
    agent = PPOAgent(state_dim=4, action_dim=2)
    
    # Run initial Monte Carlo simulation with visualization
    mc_metrics, mc_policy = mc_simulator.run_simulation()
    
    # Initialize players
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
    
    # UI Components
    start_button = Button(480, 3, diameter=40, image_path="Play.png", border_color=(255, 255, 255), border_width=3)
    
    # Control buttons
    buttons_red = [
        (100, SCREEN_HEIGHT - 25), (200, SCREEN_HEIGHT - 25), 
        (700, SCREEN_HEIGHT - 25), (400, SCREEN_HEIGHT - 25),
        (100, 22), (200, 22), (700, 22), (400, 22)
    ]
    
    buttons_blue = [
        (600, SCREEN_HEIGHT - 25), (300, SCREEN_HEIGHT - 25),
        (800, SCREEN_HEIGHT - 25), (900, SCREEN_HEIGHT - 25),
        (600, 22), (300, 22), (800, 22), (900, 22)
    ]
    
    # Game metrics
    game_stats = {
        'episodes': 0,
        'wins': 0,
        'goals_scored': 0,
        'goals_conceded': 0,
        'successful_actions': 0
    }
    
    running = True
    game_started = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and not game_started:
                    ball.start(timer)
                    game_started = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.is_clicked(pygame.mouse.get_pos()) and not game_started:
                    ball.start(timer)
                    game_started = True

        if game_started and not timer.time_up():
            # Get current game state
            state = {
                'ball_position': (ball.x, ball.y),
                'ball_velocity': (ball.vx, ball.vy),
                'player_positions': [player.y for player in all_players]
            }
            
            # Use PPO agent for AI players
            if random.random() < 0.2:  # 20% chance to use AI
                action = agent.select_action(state_to_tensor(state))
                execute_ai_action(action, blue_players)
            
            # Handle human controls
            handle_key_presses(red_players)
            handle_key_presses1(blue_players)
            
            # Update ball position
            ball.move()
            
            # Check collisions
            for player in all_players:
                check_collision(ball, player)
            
            # Check goals
            goal = check_goal(ball, timer)
            if goal == "Left":
                left_score += 1
                game_stats['goals_scored'] += 1
                ball = Ball()
                game_started = False
            elif goal == "Right":
                right_score += 1
                game_stats['goals_conceded'] += 1
                ball = Ball()
                game_started = False
            
            # Update metrics
            game_stats['successful_actions'] += 1 if check_successful_action(state) else 0
        
        # Draw game state
        draw_table(screen)
        ball.draw(screen)
        timer.update()
        timer.draw(screen)
        
        for player in all_players:
            player.draw(screen)
        
        if not game_started and not timer.time_up():
            start_button.draw(screen)
        
        # Draw control buttons
        draw_control_buttons(screen, buttons_red, buttons_blue)
        
        # Draw score
        draw_score(screen, left_score, right_score)
        
        pygame.display.flip()
        clock.tick(FPS)
        
        # Check for game end
        if timer.time_up():
            game_stats['episodes'] += 1
            if left_score > right_score:
                game_stats['wins'] += 1
            
            # Update metrics files
            update_metric_files(game_stats)
            
            # Ask for restart
            if show_game_end_dialog():
                reset_game_state(ball, timer, game_stats)
                left_score = right_score = 0
                game_started = False
            else:
                running = False
    
    pygame.quit()
    return game_stats

def state_to_tensor(state):
    return torch.FloatTensor([
        state['ball_position'][0] / SCREEN_WIDTH,
        state['ball_position'][1] / SCREEN_HEIGHT,
        state['ball_velocity'][0] / 10,
        state['ball_velocity'][1] / 10
    ])

def execute_ai_action(action, players):
    if action == 0:  # Move up
        for player in players:
            player.y = max(50, player.y - 50)
    else:  # Move down
        for player in players:
            player.y = min(SCREEN_HEIGHT - 50 - PLAYER_HEIGHT, player.y + 50)



def check_successful_action(state):
    """
    Evaluates if current game state represents a successful action based on:
    - Ball movement toward opponent's goal
    - Successful defensive blocks
    - Maintaining possession
    - Strategic positioning
    """
    ball_pos = state['ball_position']
    ball_vel = state['ball_velocity']
    player_positions = state['player_positions']
    
    # Check offensive success
    offensive_success = (
        # Ball moving towards opponent's goal (right side)
        (ball_vel[0] > 0 and ball_pos[0] > SCREEN_WIDTH/2) or
        # Strong shot (high velocity)
        (abs(ball_vel[0]) > 7)
    )
    
    # Check defensive success
    defensive_success = (
        # Blocked opponent's shot
        (ball_vel[0] < 0 and ball_pos[0] < SCREEN_WIDTH/2) or
        # Good defensive positioning (players covering different heights)
        len(set(pos for pos in player_positions if abs(pos - ball_pos[1]) < 50)) >= 2
    )
    
    # Check possession success
    possession_success = (
        # Ball near player
        any(abs(pos - ball_pos[1]) < PLAYER_HEIGHT/2 for pos in player_positions) and
        # Ball under control (low velocity)
        abs(ball_vel[0]) < 3 and abs(ball_vel[1]) < 3
    )
    
    # Check strategic positioning
    strategic_success = (
        # Players well-distributed vertically
        max(player_positions) - min(player_positions) > SCREEN_HEIGHT/3 and
        # At least one player near ball height
        any(abs(pos - ball_pos[1]) < PLAYER_HEIGHT for pos in player_positions)
    )
    
    # Return true if any type of success is achieved
    return offensive_success or defensive_success or possession_success or strategic_success


def update_metric_files(stats):
    total_games = max(stats['episodes'], 1)
    metrics = {
        'win_rate': (stats['wins'] / total_games) * 100,
        'goal_differential': ((stats['goals_scored'] - stats['goals_conceded']) / 
                            max(stats['goals_scored'], 1)) * 100,
        'strategic_adaptability': (stats['successful_actions'] / 
                                 max(stats['episodes'] * 1000, 1)) * 100
    }
    
    for metric_name, value in metrics.items():
        filename = f'{metric_name.upper()}.txt'
        with open(filename, 'a') as f:
            f.write(f'{value:.2f}\n')

def show_game_end_dialog():
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno("Game Over", "Do you want to play again?")
    root.destroy()
    return result

def reset_game_state(ball, timer, stats):
    ball.x = SCREEN_WIDTH // 2
    ball.y = SCREEN_HEIGHT // 2
    ball.vx = ball.vy = 0
    timer.reset()
    stats['goals_scored'] = stats['goals_conceded'] = 0