import numpy as np
import random
import torch
from collections import deque
from Model_again import SnakeNet, SnakeLearning
from TheSnake_again import Snake_again
import matplotlib.pyplot as plt

model_path=r'D:\Snake Q-learning\Whole_grid\Snake_Parameters'

#Starting variables
GRID_width = 6  # 20 columns (x direction)
GRID_height = 8  # 19 rows (y direction)
CELL_size = 1
score_history=[]
loss_history=[]
epsilon_hostory=[]
gamma=0.92
learning_rate=0.0005
batch_size = 128
nr_inputs=129
experience_replay = deque(maxlen=100_000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_model = SnakeNet(nr_inputs).to(device)
target_model = SnakeNet(nr_inputs).to(device)
target_model.load_state_dict(policy_model.state_dict()) #copies the weights and biases from a nn to the other
trainer = SnakeLearning(policy_model, learning_rate, gamma)

policy_model.cr_ld(model_path)

n=600
for episode in range(n):
    pos = random.randint(4, GRID_height-4)
    snake_head = [pos*CELL_size, 3*CELL_size]
    snake_body = [[pos*CELL_size,3*CELL_size], [(pos-1)* CELL_size, 3*CELL_size], [(pos-2)*CELL_size, 3*CELL_size]]
    food_pos = [
    random.randint(0, GRID_width - 1) * CELL_size,
    random.randint(0, GRID_height - 1) * CELL_size
    ]
    food_spawn = True
    epsilon = max(0.01, 0.8 * (0.996 ** episode))
    epsilon_hostory.append(epsilon)
    direction = 'RIGHT'
    done=False
    score=0
    record=0
    step_count=0
    reward_step=0
    while not done:
        step_count+=1
        action=[0,0,0]
        #Compute current state
        state_current = Snake_again.compute_state(snake_body,food_pos, snake_head, direction, GRID_width, GRID_height, CELL_size)
        #Choose the next action
        if random.random() < epsilon:
            action[random.randint(0, 2)]=1 #Explore
        else:
            state=state_current
            with torch.no_grad():
                state=torch.tensor(state, dtype=torch.float).cuda()
                prezi = policy_model(state).cuda()         #Exploit
                move= torch.argmax(prezi).item()
            action[move]=1
        
        #Move 
        snake_head, snake_body, direction, food_pos, food_spawn, score = Snake_again.movee( snake_head, snake_body, direction, food_pos, food_spawn, score, action, GRID_width, GRID_height, CELL_size)
        #Check if game over
        done = Snake_again.danger(snake_head, snake_body, GRID_width, GRID_height, CELL_size)
        #Calculate reward
        reward= Snake_again.reward_function(snake_body,food_pos, snake_head, GRID_width, GRID_height, CELL_size)
        if reward > 1:
            reward_step = 0
        if score > record:
            reward += 3  # bonus for breaking personal best
            record = score
        elif reward < 1:
            reward_step += 1
        if reward_step > GRID_height + GRID_width +5 :
            reward = -10
            done = True
            reward_step = 0

        #Compute the next state
        state_next = Snake_again.compute_state(snake_body,food_pos, snake_head, direction, GRID_width, GRID_height, CELL_size)
        
        #Save experience to buffer
        experience=state_current, action, reward, state_next, done
        experience_replay.append(experience)

        #Perform training
        
        loss = trainer.compute_Q_live(experience)
        
        #Train on saved memories
        if len(experience_replay) > 2048:
            sample_experience= random.sample(experience_replay, batch_size)
            loss = trainer.compute_Q_live(sample_experience)
        else:
            continue
        if step_count % 150 == 0:
            trainer.update_target_network()
            
            

    trainer.update_target_network()
    score_history.append(score)
    loss_history.append(loss)

    if episode % 50 == 0:
        policy_model.save_model(model_path)
    if episode % 10 == 0:
        
        avg_loss = sum(loss_history[-10:]) / 10
        avg_score = sum(score_history[-10:]) / 10
        print(f"Game: {episode}, AvgScore: {avg_score:.1f}, AvgLoss: {avg_loss*10:.4f}")
    
        
policy_model.save_model(model_path)

plt.figure(figsize=(12, 5))
plt.subplot(131)
plt.plot(score_history)
plt.title("Scores")
plt.subplot(132)
plt.plot(epsilon_hostory)
plt.title("Epsilon")
plt.subplot(133)
plt.plot(loss_history[100:])
plt.title("Loss")
plt.tight_layout()
plt.ioff()  # Turn interactive mode off
plt.show()  # Keep the window open

            
