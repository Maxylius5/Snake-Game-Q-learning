import pygame, sys, time, random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import numpy as np
import random
import torch
from collections import deque
from Model_SnakeAI import SnakeNetAI, SnakeLearningAI
from Agent import Agent
nr_inputs=11
model_path=r'D:\Snake Q-learning\Vision\SnakeAi_Parameters'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_model = SnakeNetAI(nr_inputs).to(device)
target_model = SnakeNetAI(nr_inputs).to(device)
target_model.load_state_dict(policy_model.state_dict()) #copies the weights and biases from a nn to the other

gamma=0.9
learning_rate=0.0001
policy_model.load_state_dict(torch.load(model_path))
policy_model.eval()  # Don't forget this for inference

trainer = SnakeLearningAI(policy_model, learning_rate, gamma)
#Starting variables
GRID_width = 275  # 20 columns (x direction)
GRID_height = 55  # 19 rows (y direction)
CELL_size = 10
difficulty=200
frame_size_x = GRID_width*CELL_size
frame_size_y = GRID_height*CELL_size
snake_head = [30, 40]
snake_body = [[30, 40], [30, 30], [30, 20]]
food_pos = [
    random.randint(0, GRID_width - 1) * CELL_size,
    random.randint(0, GRID_height - 1) * CELL_size
]
food_spawn = True

nr_inputs=11
in_layer = np.zeros((nr_inputs,), dtype=np.float32)
score=0

# Checks for errors encountered
check_errors = pygame.init()
# pygame.init() example output -> (6, 0)
# second number in tuple gives number of errors
if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')


# Initialise game window
pygame.display.set_caption('Snake')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))



# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)


# FPS (frames per second) controller
fps_controller = pygame.time.Clock()

def game_over():
    my_font = pygame.font.SysFont('times new roman', 20)
    game_over_surface = my_font.render('YOU DIED', True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (frame_size_x/2, frame_size_y/4)
    game_window.fill(black)
    game_window.blit(game_over_surface, game_over_rect)
    show_score(0, red, 'times', 20)
    pygame.display.flip()
    time.sleep(3)
    pygame.quit()
    sys.exit()

def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    if choice == 1:
        score_rect.midtop = (frame_size_x/10, 15)
    else:
        score_rect.midtop = (frame_size_x/2, frame_size_y/1.25)
    game_window.blit(score_surface, score_rect)

pos = random.randint(2, GRID_height-3)
snake_head = [pos*CELL_size, 3*CELL_size]
snake_body = [[pos*CELL_size,3*CELL_size], [(pos-1)* CELL_size, 3*CELL_size]] #, [(pos-2)*CELL_size, 3*CELL_size]
food_pos = [
    random.randint(0, GRID_width - 1) * CELL_size,
    random.randint(0, GRID_height - 1) * CELL_size
]
food_spawn = True
direction = 'RIGHT'
directions = ["RIGHT", "DOWN", "LEFT", "UP"]
score=0
n=10
while True:
    for i in range(n):    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        action=[0,0,0]
        with torch.no_grad():
            state_current = Agent.compute_state(snake_body, food_pos, snake_head, direction, GRID_width, GRID_height, CELL_size)
            state_current = torch.tensor(state_current, dtype=torch.float32).unsqueeze(0).to(device)
            prezi = policy_model(state_current).cuda()        #Exploit
            move= torch.argmax(prezi).item()
        action[move]=1

        idx = directions.index(direction)
        if np.array_equal(action,[1,0,0]):
            new_dir = directions[idx]
        elif np.array_equal(action,[0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = directions[next_idx] # right Turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = directions[next_idx]
        direction=new_dir
        # Update head position
        x, y = snake_head
        if new_dir == "RIGHT":
            x += CELL_size
        elif new_dir == "LEFT":
            x -= CELL_size
        elif new_dir == "UP":
            y -= CELL_size
        elif new_dir == "DOWN":
            y += CELL_size
        snake_head = [x, y]

        # Snake body growing mechanism
        snake_body.insert(0, list(snake_head))
        if snake_head == food_pos:
            score += 1
            food_spawn = False
        else:
            snake_body.pop()

        # Spawning food on the screen
        if not food_spawn:
            food_pos = [
                random.randint(0, GRID_width - 1) * CELL_size,
                random.randint(0, GRID_height - 1) * CELL_size
            ]

        food_spawn = True

        game_window.fill(black)
        for posi in snake_body:
        # Snake body
        # .draw.rect(play_surface, color, xy-coordinate)
        # xy-coordinate -> .Rect(x, y, size_x, size_y)
            pygame.draw.rect(game_window, green, pygame.Rect(posi[0], posi[1], CELL_size, CELL_size))

    # Snake food
        pygame.draw.rect(game_window, white, pygame.Rect(food_pos[0], food_pos[1], CELL_size+1, CELL_size+1))

    # Game Over conditions
    # Getting out of bounds
        if snake_head[0] < 0 or snake_head[0] > frame_size_x-CELL_size:
            game_over()
        if snake_head[1] < 0 or snake_head[1] > frame_size_y-CELL_size:
            game_over()
    # Touching the snake body
        for block in snake_body[1:]:
            if snake_head[0] == block[0] and snake_head[1] == block[1]:
                game_over()
    
    # Refresh game screen
        pygame.display.update()
    # Refresh rate
        fps_controller.tick(difficulty)