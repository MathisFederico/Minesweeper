from learnrl import Agent
from minesweeper import MinesweeperEnv

class RandomAgent(Agent):

    def __init__(self, env:MinesweeperEnv):
        pass

    def act(self, observation:np.ndarray):
        grid_shape = observation.shape
        return (np.random.randint(grid_shape[0]), np.random.randint(grid_shape[1]), np.random.randint(2))
    
    def learn(self):
        pass

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        pass

