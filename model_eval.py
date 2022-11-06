import torch
from snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
import numpy as np
from Helper import plot

n_game = 0

model = torch.jit.load('/home/fr000gs/SnakeGameAI/model.pt')
model.eval()

def get_state(game):
    head = game.snake[0]
    point_l=Point(head.x - BLOCK_SIZE, head.y)
    point_r=Point(head.x + BLOCK_SIZE, head.y)
    point_u=Point(head.x, head.y - BLOCK_SIZE)
    point_d=Point(head.x, head.y + BLOCK_SIZE)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger Straight
        (dir_u and game.is_collision(point_u))or
        (dir_d and game.is_collision(point_d))or
        (dir_l and game.is_collision(point_l))or
        (dir_r and game.is_collision(point_r)),

        # Danger right
        (dir_u and game.is_collision(point_r))or
        (dir_d and game.is_collision(point_l))or
        (dir_u and game.is_collision(point_u))or
        (dir_d and game.is_collision(point_d)),

        #Danger Left
        (dir_u and game.is_collision(point_r))or
        (dir_d and game.is_collision(point_l))or
        (dir_r and game.is_collision(point_u))or
        (dir_l and game.is_collision(point_d)),

        # Move Direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,

        #Food Location
        game.food.x < game.head.x, # food is in left
        game.food.x > game.head.x, # food is in right
        game.food.y < game.head.y, # food is up
        game.food.y > game.head.y  # food is down
    ]
    return np.array(state,dtype=int)

def get_action(state):
    final_move = [0,0,0]
    state0 = torch.tensor(state,dtype=torch.float)
    prediction = model(state0) # prediction by model 
    move = torch.argmax(prediction).item()
    final_move[move]=1 
    return final_move

def main():
    global n_game
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    game = SnakeGameAI()
    while True:
        state = get_state(game)
        final_move = get_action(state)
        reward, done, score = game.play_step(final_move)
        if  done:
            game.reset()
            n_game += 1
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / n_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    main()
