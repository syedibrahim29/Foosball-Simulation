from FoosballTable import (MonteCarloSimulator, Ball, Player, Timer, MonteCarloPlayer, Rod,
                          SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_HEIGHT, 
                          RED, BLUE, BLACK, WHITE, FPS)  # This has your MonteCarloSimulator class
from FoosballTableHelper import * # This has helper functions
import pygame
import random
import pickle  # Add this import
from os.path import exists  # For file checking
from train_agents import train_agents

train_agents()

def get_player_rows(players):
    """Group players by their x-position (row)"""
    rows = {}
    for player in players:
        if player.x not in rows:
            rows[player.x] = []
        rows[player.x].append(player)
    return rows

def draw_control_buttons(screen):
    """Draw control buttons for both teams"""
    font = pygame.font.SysFont('georgia', 30)
    
    # Define button positions and labels
    red_buttons = [
        # Top buttons for Red team
        ('Q', RED, (100, 22)), ('W', RED, (200, 22)), 
        ('E', RED, (400, 22)), ('R', RED, (700, 22)),
        # Bottom buttons for Red team
        ('A', RED, (100, SCREEN_HEIGHT - 25)), ('S', RED, (200, SCREEN_HEIGHT - 25)),
        ('D', RED, (400, SCREEN_HEIGHT - 25)), ('F', RED, (700, SCREEN_HEIGHT - 25))
    ]
    
    blue_buttons = [
        # Top buttons for Blue team
        ('U', BLUE, (300, 22)), ('I', BLUE, (600, 22)), 
        ('O', BLUE, (800, 22)), ('P', BLUE, (900, 22)),
        # Bottom buttons for Blue team
        ('H', BLUE, (300, SCREEN_HEIGHT - 25)), ('J', BLUE, (600, SCREEN_HEIGHT - 25)),
        ('K', BLUE, (800, SCREEN_HEIGHT - 25)), ('L', BLUE, (900, SCREEN_HEIGHT - 25))
    ]
    
    # Draw all buttons
    for letter, color, pos in red_buttons + blue_buttons:
        pygame.draw.circle(screen, color, pos, 20)
        text = font.render(letter, True, WHITE)
        text_rect = text.get_rect(center=pos)
        screen.blit(text, text_rect)

def check_wall_collision(ball):
    """Check and handle ball collision with walls"""
    # Top and bottom walls
    if ball.y - BALL_RADIUS <= 50:
        ball.y = 50 + BALL_RADIUS
        ball.vy = abs(ball.vy)  # Bounce downward
    elif ball.y + BALL_RADIUS >= SCREEN_HEIGHT - 50:
        ball.y = SCREEN_HEIGHT - 50 - BALL_RADIUS
        ball.vy = -abs(ball.vy)  # Bounce upward

    # Left and right walls (except for goal areas)
    if ball.x - BALL_RADIUS <= 50:
        # Check if it's not in the goal area
        if not (SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2 <= ball.y <= SCREEN_HEIGHT // 2 + GOAL_HEIGHT // 2):
            ball.x = 50 + BALL_RADIUS
            ball.vx = abs(ball.vx)  # Bounce right
    elif ball.x + BALL_RADIUS >= SCREEN_WIDTH - 50:
        # Check if it's not in the goal area
        if not (SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2 <= ball.y <= SCREEN_HEIGHT // 2 + GOAL_HEIGHT // 2):
            ball.x = SCREEN_WIDTH - 50 - BALL_RADIUS
            ball.vx = -abs(ball.vx)  # Bounce left

def check_player_collision(ball, player):
    """Enhanced collision detection between ball and player"""
    if (player.x - BALL_RADIUS <= ball.x <= player.x + PLAYER_WIDTH + BALL_RADIUS and 
        player.y - BALL_RADIUS <= ball.y <= player.y + PLAYER_HEIGHT + BALL_RADIUS):
        
        # Determine which side of the player the ball hit
        if abs(ball.x - player.x) < abs(ball.x - (player.x + PLAYER_WIDTH)):
            # Hit left side
            ball.x = player.x - BALL_RADIUS
            ball.vx = -abs(ball.vx) * 1.1  # Bounce left with increased speed
        else:
            # Hit right side
            ball.x = player.x + PLAYER_WIDTH + BALL_RADIUS
            ball.vx = abs(ball.vx) * 1.1  # Bounce right with increased speed
        
        # Add some vertical deflection based on where the ball hits the player
        hit_position = (ball.y - player.y) / PLAYER_HEIGHT
        ball.vy = ball.vy + (hit_position - 0.5) * 10  # Increased vertical effect
        
        # Ensure minimum speed after collision
        maintain_ball_speed(ball, min_speed=7.0)  # Slightly higher minimum speed after collision
        return True
    return False

# Second: helper functions like check_goal
def check_goal(ball, timer=None):
    """Check if a goal has been scored"""
    if ball.x - BALL_RADIUS <= 50 and (SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2 <= ball.y <= SCREEN_HEIGHT // 2 + GOAL_HEIGHT // 2):
        if timer:
            timer.pause()
        return "Right"
    elif ball.x + BALL_RADIUS >= SCREEN_WIDTH - 70 and (SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2 <= ball.y <= SCREEN_HEIGHT // 2 + GOAL_HEIGHT // 2):
        if timer:
            timer.pause()
        return "Left"
    return None

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
    
    def load_tables(self, tables_file):
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
        try:
            print("Available methods:", dir(self.simulator))  # Debug print
            if self.side == 'right':
                state = self.mirror_state(state)

            action = self.simulator.select_action(state)

            if self.side == 'right':
                action = self.mirror_action(action)

            return action
        except Exception as e:
            print(f"Error in get_action: {e}")
            return 'defend'

    def mirror_state(self, state):
        """Mirror the state for right player"""
        mirrored_state = state.copy()
        ball_x, ball_y = state['ball_position']
        mirrored_state['ball_position'] = (SCREEN_WIDTH - ball_x, ball_y)
        mirrored_state['player_positions'] = [SCREEN_HEIGHT - pos for pos in state['player_positions']]
        return mirrored_state

    def mirror_action(self, action):
        """Mirror the action for right player"""
        action_map = {
            'move_left': 'move_right',
            'move_right': 'move_left',
            'move_up': 'move_down',
            'move_down': 'move_up',
            'shoot': 'shoot',
            'defend': 'defend'
        }
        return action_map[action]

def simulate_live_game():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Monte Carlo vs Monte Carlo")
    clock = pygame.time.Clock()

    # Initialize players
    print("Initializing players...")
    left_player = MonteCarloPlayer(tables_file='monte_carlo_tables_left.pkl', side='left')
    right_player = MonteCarloPlayer(tables_file='monte_carlo_tables_right.pkl', side='right')

    # Initialize game objects
    ball = Ball()
    timer = Timer(450, 550, 60)  # 1 minute game
    left_score = 0
    right_score = 0

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
                   [Player(rows_x_positions[7], y, BLUE) for y in blue_positions_y[0]]

    all_players = red_players + blue_players
    
    # Start game
    ball.vx = 5
    ball.vy = random.choice([-5, 5])
    ball.moving = True
    timer.start()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if ball.moving and not timer.time_up():
            # Get current game state
            current_state = {
                'ball_position': (ball.x, ball.y),
                'ball_velocity': (ball.vx, ball.vy),
                'player_positions': [player.y for player in all_players],
                'goals_scored': 0,
                'goals_conceded': 0
            }

            # Get actions from both players
            left_action = left_player.get_action(current_state)
            right_action = right_player.get_action(current_state)

            # Execute actions
            execute_player_action(left_action, red_players)
            execute_player_action(right_action, blue_players)

            # Update ball position
            if ball.moving:
                ball.x += ball.vx
                ball.y += ball.vy
                
                # Apply friction/drag
                ball.vx *= 0.995
                ball.vy *= 0.995

                # Check wall collisions
                check_wall_collision(ball)

                # Check collisions with players
                for player in all_players:
                    if check_player_collision(ball, player):
                        break  # Exit after first collision

            # Check goals
            goal_result = check_goal(ball, timer)
            if goal_result:
                if goal_result == "Left":
                    left_score += 1
                else:
                    right_score += 1
                
                # Reset ball
                ball.x = SCREEN_WIDTH / 2
                ball.y = SCREEN_HEIGHT / 2
                ball.vx = random.choice([-5, 5])
                ball.vy = random.choice([-5, 5])

        # Draw everything
        draw_table(screen)
        ball.draw(screen)
        timer.draw(screen)
        
        for player in all_players:
            player.draw(screen)

        # Draw score
        font = pygame.font.SysFont('georgia', 36)
        left_text = font.render(str(left_score), True, BLACK)
        right_text = font.render(str(right_score), True, BLACK)
        screen.blit(left_text, (20, 20))
        screen.blit(right_text, (SCREEN_WIDTH - 40, 20))

        pygame.display.flip()
        clock.tick(FPS)

        if timer.time_up():
            running = False

    pygame.quit()
    return left_score, right_score

def execute_player_action(action, players, move_distance=10):
    """Execute the given action for a team's players while maintaining formation"""
    # Group players by rows
    rows = get_player_rows(players)
    
    # Process each row separately
    for x_pos, row_players in rows.items():
        if len(row_players) <= 1:
            continue
            
        # Sort players in row by vertical position
        row_players.sort(key=lambda p: p.y)
        
        # Calculate current spacing
        spacing = (row_players[-1].y - row_players[0].y) / (len(row_players) - 1)
        
        if action == 'move_up':
            # Move the entire row up
            new_top = max(50, row_players[0].y - move_distance)
            for i, player in enumerate(row_players):
                player.y = new_top + (i * spacing)
                
        elif action == 'move_down':
            # Move the entire row down
            max_y = SCREEN_HEIGHT - 50 - PLAYER_HEIGHT
            new_bottom = min(max_y, row_players[-1].y + move_distance)
            new_top = new_bottom - (spacing * (len(row_players) - 1))
            for i, player in enumerate(row_players):
                player.y = new_top + (i * spacing)
                
        elif action == 'shoot':
            # Move row towards ball while maintaining spacing
            ball_y = SCREEN_HEIGHT // 2  # Default target
            middle_idx = len(row_players) // 2
            target_y = ball_y - (spacing * middle_idx)
            
            # Ensure the formation stays within bounds
            target_y = max(50, min(SCREEN_HEIGHT - 50 - (spacing * (len(row_players) - 1)), target_y))
            
            for i, player in enumerate(row_players):
                player.y = target_y + (i * spacing)
                
        elif action == 'defend':
            # Spread the row evenly across the available space
            available_height = SCREEN_HEIGHT - 100 - PLAYER_HEIGHT
            new_spacing = available_height / (len(row_players) - 1)
            for i, player in enumerate(row_players):
                player.y = 50 + (i * new_spacing)
                
def maintain_ball_speed(ball, min_speed=5.0, max_speed=15.0):
    """Maintain ball speed within desired range"""
    current_speed = (ball.vx ** 2 + ball.vy ** 2) ** 0.5
    
    if current_speed < min_speed:
        # Increase speed
        speed_multiplier = min_speed / current_speed
        ball.vx *= speed_multiplier
        ball.vy *= speed_multiplier
    elif current_speed > max_speed:
        # Reduce speed
        speed_multiplier = max_speed / current_speed
        ball.vx *= speed_multiplier
        ball.vy *= speed_multiplier
        

def check_collision(ball, player):
    """Check if ball collides with player"""
    return (player.x <= ball.x <= player.x + PLAYER_WIDTH and 
            player.y <= ball.y <= player.y + PLAYER_HEIGHT)
    
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Monte Carlo Foosball")
    clock = pygame.time.Clock()

    # Initialize Monte Carlo players
    left_player = MonteCarloPlayer(tables_file='monte_carlo_tables_left.pkl', side='left')
    right_player = MonteCarloPlayer(tables_file='monte_carlo_tables_right.pkl', side='right')

    # Initialize ball and timer
    ball = Ball()
    timer = Timer(SCREEN_WIDTH//2 - 50, SCREEN_HEIGHT - 35, 60)  # 1-minute game

    # Rod positions (x-coordinates) for both teams
    left_x_positions = [100, 250, 400, 550]   # Left team rod positions
    right_x_positions = [450, 600, 750, 900]  # Right team rod positions

    def get_spacing(num_players):
        playable_height = SCREEN_HEIGHT - 150  # Leave some margin at top and bottom
        return playable_height / (num_players - 1) if num_players > 1 else 0

    # Initialize rods with the official 1-2-5-3 formation
    rods = {
        # Left team (Red)
        'red_gk': Rod(left_x_positions[0], 1, RED, get_spacing(1), 'left'),    # Goalkeeper
        'red_def': Rod(left_x_positions[1], 2, RED, get_spacing(2), 'left'),   # Defense
        'red_mid': Rod(left_x_positions[2], 5, RED, get_spacing(5), 'left'),   # Midfield
        'red_fwd': Rod(left_x_positions[3], 3, RED, get_spacing(3), 'left'),   # Forward

        # Right team (Blue)
        'blue_gk': Rod(right_x_positions[3], 1, BLUE, get_spacing(1), 'right'),   # Goalkeeper
        'blue_def': Rod(right_x_positions[2], 2, BLUE, get_spacing(2), 'right'),  # Defense
        'blue_mid': Rod(right_x_positions[1], 5, BLUE, get_spacing(5), 'right'),  # Midfield
        'blue_fwd': Rod(right_x_positions[0], 3, BLUE, get_spacing(3), 'right')   # Forward
    }

    def execute_team_action(action, team_rods):
        """Execute action for all rods in a team"""
        for rod in team_rods:
            rod.execute_action(action)

    # Initialize game state
    ball.x = SCREEN_WIDTH / 2
    ball.y = SCREEN_HEIGHT / 2
    ball.vx = random.choice([-7, 7])
    ball.vy = random.choice([-7, 7])
    ball.moving = True
    timer.start()
    score_red = 0
    score_blue = 0

    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not timer.time_up() and ball.moving:
            # Get current game state
            current_state = {
                'ball_position': (ball.x, ball.y),
                'ball_velocity': (ball.vx, ball.vy),
                'player_positions': [player.y for rod in rods.values() for player in rod.players],
                'goals_scored': score_red if ball.x < SCREEN_WIDTH//2 else score_blue,
                'goals_conceded': score_blue if ball.x < SCREEN_WIDTH//2 else score_red
            }

            # Get Monte Carlo actions
            left_action = left_player.get_action(current_state)
            right_action = right_player.get_action(current_state)

            # Execute actions for each team
            left_team_rods = [rods['red_gk'], rods['red_def'], rods['red_mid'], rods['red_fwd']]
            right_team_rods = [rods['blue_gk'], rods['blue_def'], rods['blue_mid'], rods['blue_fwd']]
            
            execute_team_action(left_action, left_team_rods)
            execute_team_action(right_action, right_team_rods)

            # Update ball position and check collisions
            ball.x += ball.vx
            ball.y += ball.vy

            # Check wall collisions
            check_wall_collision(ball)
            
            # Check for collisions with players
            for rod in rods.values():
                for player in rod.players:
                    if check_player_collision(ball, player):
                        ball.vx = -ball.vx * 1.1
                        maintain_ball_speed(ball)
                        break

            # Check for goals
            goal_result = check_goal(ball, timer)
            if goal_result:
                if goal_result == "Left":
                    score_red += 1
                else:
                    score_blue += 1
                # Reset ball
                ball.x = SCREEN_WIDTH / 2
                ball.y = SCREEN_HEIGHT / 2
                ball.vx = random.choice([-7, 7])
                ball.vy = random.choice([-7, 7])

            # Draw everything
            draw_table(screen)
            ball.draw(screen)
            for rod in rods.values():
                rod.draw(screen)

            # Draw score
            font = pygame.font.SysFont('georgia', 36)
            score_text_left = font.render(str(score_red), True, BLACK)
            score_text_right = font.render(str(score_blue), True, BLACK)
            screen.blit(score_text_left, (20, 20))
            screen.blit(score_text_right, (SCREEN_WIDTH - 40, 20))

            # Draw control buttons
            draw_control_buttons(screen)

            # Update timer
            timer.update()
            timer.draw(screen)

            pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
#    left_score, right_score = simulate_live_game()
#    print(f"Game Over! Final Score - Left: {left_score}, Right: {right_score}")