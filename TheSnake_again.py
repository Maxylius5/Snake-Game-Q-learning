import numpy as np
import random


class Snake_again:

    def movee( snake_head, snake_body, direction, food_pos, food_spawn, score, action, GRID_width, GRID_height, CELL_size):

        # Determine new direction
        directions = ["RIGHT", "DOWN", "LEFT", "UP"]
        idx = directions.index(direction)
        if np.array_equal(action,[1,0,0]):
            new_dir = directions[idx]
        elif np.array_equal(action,[0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = directions[next_idx] # right Turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = directions[next_idx]

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

        snake_head= [x, y]

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

        return snake_head, snake_body, new_dir, food_pos, food_spawn, score
    

    def compute_state(snake_body,food_pos, snake_head, direction, GRID_width, GRID_height, CELL_size):
        in_variables = np.zeros(8, dtype=np.float32)
        if food_pos[0] < snake_head[0]:
            in_variables[0]=(GRID_width-(snake_head[0]-food_pos[0])/CELL_size)/GRID_width # food is in left
        if food_pos[0] > snake_head[0]:
            in_variables[1]=(GRID_width-(food_pos[0]-snake_head[0])/CELL_size)/GRID_width  # food is in right
        if food_pos[1] < snake_head[1]:
            in_variables[2]=(GRID_height-(snake_head[1]-food_pos[1])/CELL_size)/GRID_height  # food is up
        if food_pos[1] > snake_head[1]:
            in_variables[3]=(GRID_height-(food_pos[1]-snake_head[1])/CELL_size)/GRID_height
        
        if direction == 'RIGHT':
            in_variables[4]=1
        if direction == 'LEFT':
            in_variables[5]=1
        if direction == 'UP':
            in_variables[6]=1   
        if direction == 'DOWN':
            in_variables[7]=1
        obs = np.zeros((11, 11), dtype=np.float32)  
        #nr_inputs=11 *11 +8=129
        dx, dy = {
        'UP': (0, -1),
        'DOWN': (0, 1),
        'LEFT': (-1, 0),
        'RIGHT': (1, 0)
            }[direction]
        for i in range(11):
            for j in range(11):
            # Calculate world coordinates relative to head
                wx = snake_head[0] + (j - 5) * CELL_size * dx - (i - 5) * CELL_size * dy
                wy = snake_head[1] + (j - 5) * CELL_size * dy + (i - 5) * CELL_size * dx
            
                if 0 <= wx < GRID_width*CELL_size and 0 <= wy < GRID_height*CELL_size:
                    
                    if [wx, wy] in snake_body:
                        obs[i, j] = 1
                else:
                    obs[i,j]=1 #for walls
        
        flat_vision=obs.flatten()
        state= np.concatenate((in_variables, flat_vision))
        
        
        return state
    
    def danger(snake_head, snake_body, GRID_width, GRID_height, CELL_size):
        done=False
        if snake_head[0] < 0 or snake_head[0] > CELL_size*(GRID_width-1):
            done=True
        if snake_head[1] < 0 or snake_head[1] > CELL_size*(GRID_height-1):
            done=True
        # Touching the snake body
        for block in snake_body[1:]:
            if snake_head[0] == block[0] and snake_head[1] == block[1]:
                done=True
        return done
    
    def reward_function(snake_body,food_pos, snake_head, GRID_width, GRID_height, CELL_size):
        reward=0.0
        
        # Reward being closer to food

        if snake_head == food_pos:
            reward = +20

        if snake_head[0] < 0 or snake_head[0] > GRID_width - CELL_size:
            reward=-20
        if snake_head[1] < 0 or snake_head[1] > GRID_height- CELL_size:
            reward=-20
        for block in snake_body[1:]:
            if snake_head[0] == block[0] and snake_head[1] == block[1]:
                reward=-20
        
        return reward/2