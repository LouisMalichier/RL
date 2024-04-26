import gymnasium as gym
from dqnV0 import Agent
import  matplotlib as plt
#from utils import plotLearning
import numpy as np
#from configuration import env

if __name__ == '__main__':
    env = gym.make("highway-fast-v0", render_mode="rgb_array")

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=2, n_actions=4, eps_end=0.01,
                  input_dims=[5], alpha=0.001)
    #initialement input_dims = [8], batch size 64
    scores, eps_history = [], []
    n_games = 500
    
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        #print("obs post reset", observation)
        #print("observation", observation)
        while not done:
            action = agent.chooseAction(observation[0])
            #print("action", action )
  
            observation_, reward, done, trash,  info = env.step(action)
            #print("obs_ V0_N", observation_)
            #Trash est une info en plus dont je ne connais pas l'utilité, prends la valeur True/False
            score += reward
            agent.storeTransition(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            print("_____________")
            observation = observation_
        scores.append(score)
        eps_history.append(agent.EPSILON)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.EPSILON)
    x = [i+1 for i in range(n_games)]
    filename = 'highway.png'

    print(x, scores, eps_history, filename)
