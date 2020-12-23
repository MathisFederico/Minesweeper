import numpy as np
from learnrl import Agent
from minesweeper import MinesweeperEnv
import pygame

class HumanAgent(Agent):

    def __init__(self, env:MinesweeperEnv):
        self.env = env
 
    def act(self, observation:np.ndarray, greedy=True):
        waiting = True
        while waiting:
            if self.env.pygame_is_init:
                self.env.render()
                for event in pygame.event.get(): # pylint: disable=E1101
                    if event.type == pygame.QUIT: # pylint: disable=E1101
                        waiting = False
                    if event.type == pygame.MOUSEBUTTONDOWN: # pylint: disable=E1101
                        waiting = False
                        block_position = (np.array(event.pos) - self.env.origin) / (self.env.scale_factor * self.env.BLOCK_SIZE)
                        block_position = (int(block_position[1]), int(block_position[0]))
                        if event.button == 1: action = 0
                        elif event.button == 3: action = 1
                        else: waiting = True
            else:
                waiting = False
        return block_position + (action,)
    
    def learn(self):
        pass

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        pass
