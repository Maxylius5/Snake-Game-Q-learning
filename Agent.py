import numpy as np
import random

class Agent:
    def compute_state(snake_body, food_pos, snake_head, direction, GRID_width, GRID_height, CELL_size):
        
        point_l=[snake_head[0] -CELL_size , snake_head[1]]
        point_r=[snake_head[0] +CELL_size , snake_head[1]]
        point_u=[snake_head[0] , snake_head[1] - CELL_size]
        point_d=[snake_head[0] , snake_head[1] + CELL_size]

        dir_l = direction == "LEFT"
        dir_r = direction == "RIGHT"
        dir_u = direction == "UP"
        dir_d = direction == "DOWN"

         
        state = [
            # Danger Straight
            (dir_u and Agent.danger(point_u, snake_body, GRID_width, GRID_height, CELL_size)) or
            (dir_d and Agent.danger(point_d, snake_body, GRID_width, GRID_height, CELL_size)) or
            (dir_l and Agent.danger(point_l, snake_body, GRID_width, GRID_height, CELL_size)) or
            (dir_r and Agent.danger(point_r, snake_body, GRID_width, GRID_height, CELL_size)),

            # Danger right
            (dir_u and Agent.danger(point_r, snake_body, GRID_width, GRID_height, CELL_size))or
            (dir_d and  Agent.danger(point_l, snake_body, GRID_width, GRID_height, CELL_size))or
            (dir_l and Agent.danger(point_u, snake_body, GRID_width, GRID_height, CELL_size))or
            (dir_r and Agent.danger(point_d, snake_body, GRID_width, GRID_height, CELL_size)),

            #Danger Left
            (dir_u and Agent.danger(point_l, snake_body, GRID_width, GRID_height, CELL_size))or
            (dir_d and Agent.danger(point_r, snake_body, GRID_width, GRID_height, CELL_size))or
            (dir_r and Agent.danger(point_u, snake_body, GRID_width, GRID_height, CELL_size))or
            (dir_l and Agent.danger(point_d, snake_body, GRID_width, GRID_height, CELL_size)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            food_pos[0] < snake_head[0], # food is in left
            food_pos[0] > snake_head[0],  # food is in right
            food_pos[1] < snake_head[1],  # food is up
            food_pos[1] > snake_head[1],   # food is down
        ]
        return np.array(state,dtype=int)
    
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


        '''
        
        
        reward=0
        if snake_head == food_pos:
            reward = +10
        if snake_head[0] < 0 or snake_head[0] > GRID_width-CELL_size:
            reward=-10
        if snake_head[1] < 0 or snake_head[1] > GRID_height-CELL_size:
            reward=-10
        for block in snake_body[1:]:
            if snake_head[0] == block[0] and snake_head[1] == block[1]:
                reward=-10
        return reward/10'''
    
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
    
    


        


        
        
    


