try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
from . import images

from gym import Env, spaces

from time import time
import numpy as np
from copy import copy
import colorsys

import pygame
from pygame.transform import scale

class MinesweeperEnv(Env):

    def __init__(self, grid_shape=(10, 15), bombs_density=0.1, n_bombs=None, impact_size=3, max_time=999, chicken=False):
        self.grid_shape = grid_shape
        self.grid_size = np.prod(grid_shape)
        self.n_bombs = max(1, int(bombs_density * self.grid_size)) if n_bombs is None else n_bombs
        self.n_bombs = min(self.grid_size - 1, self.n_bombs)
        self.flaged_bombs = 0
        self.flaged_empty = 0
        self.max_time = max_time

        if impact_size % 2 == 0:
            raise ValueError('Impact_size must be an odd number !')
        self.impact_size = impact_size

        # Define constants
        self.HIDDEN = 0
        self.REVEAL = 1
        self.FLAG = 2
        self.BOMB = self.impact_size ** 2
        
        # Setting up gym Env conventions
        nvec_observation = (self.BOMB + 2) * np.ones(self.grid_shape)
        self.observation_space = spaces.MultiDiscrete(nvec_observation)

        nvec_action = np.array(self.grid_shape + (2,))
        self.action_space = spaces.MultiDiscrete(nvec_action)

        # Initalize state
        self.state = np.zeros(self.grid_shape + (2,), dtype=np.uint8)

        ## Setup bombs places
        idx = np.indices(self.grid_shape).reshape(2, -1)
        bombs_ids = np.random.choice(range(self.grid_size), size=self.n_bombs, replace=False)
        self.bombs_positions = idx[0][bombs_ids], idx[1][bombs_ids]

        ## Place numbers
        self.semi_impact_size = (self.impact_size-1)//2
        bomb_impact = np.ones((self.impact_size, self.impact_size), dtype=np.uint8)
        for bombs_id in bombs_ids:
            bomb_x, bomb_y = idx[0][bombs_id], idx[1][bombs_id]
            x_min, x_max, dx_min, dx_max = self.clip_index(bomb_x, 0)
            y_min, y_max, dy_min, dy_max = self.clip_index(bomb_y, 1)
            bomb_region = self.state[x_min:x_max, y_min:y_max, 0]
            bomb_region += bomb_impact[dx_min:dx_max, dy_min:dy_max]

        ## Place bombs
        self.state[self.bombs_positions + (0,)] = self.BOMB
        self.start_time = time()
        self.time_left = int(time() - self.start_time)

        # Setup rendering
        self.pygame_is_init = False
        self.chicken = chicken
        self.done = False
        self.score = 0

    def get_observation(self):
        observation = copy(self.state[:, :, 1])

        revealed = observation == 1
        flaged = observation == 2

        observation += self.impact_size ** 2 + 1
        observation[revealed] = copy(self.state[:, :, 0][revealed])

        observation[flaged] -= 1
        return observation

    def reveal_around(self, coords, reward, done, without_loss=False):
        if not done:
            x_min, x_max, _, _ = self.clip_index(coords[0], 0)
            y_min, y_max, _, _ = self.clip_index(coords[1], 1)

            region = self.state[x_min:x_max, y_min:y_max, :]
            unseen_around = np.sum(region[..., 1] == 0)
            if unseen_around == 0:
                if not without_loss: 
                    reward -= 0.001
                return

            flags_around = np.sum(region[..., 1] == 2)
            if flags_around == self.state[coords + (0,)]:
                unrevealed_zeros_around = np.logical_and(region[..., 0] == 0, region[..., 1] == self.HIDDEN)
                if np.any(unrevealed_zeros_around):
                    zeros_coords = np.argwhere(unrevealed_zeros_around)
                    for zero in zeros_coords:
                        coord = (x_min + zero[0], y_min + zero[1])
                        self.state[coord + (1,)] = 1
                        self.reveal_around(coord, reward, done, without_loss=True)
                self.state[x_min:x_max, y_min:y_max, 1][self.state[x_min:x_max, y_min:y_max, 1] != self.FLAG] = 1
                unflagged_bombs_around = np.logical_and(region[..., 0] == self.BOMB, region[..., 1] != self.FLAG)
                if np.any(unflagged_bombs_around):
                    self.done = True
                    reward, done = -1, True
            else:
                if not without_loss: 
                    reward -= 0.001

    def clip_index(self, x, axis):
        max_idx = self.grid_shape[axis]
        x_min, x_max = max(0, x-self.semi_impact_size), min(max_idx, x + self.semi_impact_size + 1)
        dx_min, dx_max = x_min - (x - self.semi_impact_size), x_max - (x + self.semi_impact_size + 1) + self.impact_size
        return x_min, x_max, dx_min, dx_max

    def step(self, action):
        coords = action[:2]
        action_type = action[2] + 1 # 0 -> 1 = reveal; 1 -> 2 = toggle_flag
        case_state = self.state[coords + (1,)]
        case_content = self.state[coords + (0,)]
        NO_BOMBS_AROUND = 0

        reward, done = 0, False
        self.time_left = self.max_time - time() + self.start_time
        if self.time_left <= 0:
            score = -(self.n_bombs - self.flaged_bombs + self.flaged_empty)/self.n_bombs
            reward, done = score, True
            return self.get_observation(), reward, done, {'passed':False}
        
        if action_type == self.REVEAL:
            if case_state == self.HIDDEN:
                self.state[coords + (1,)] = action_type
                if case_content == self.BOMB:
                    if self.pygame_is_init: self.done = True
                    reward, done = -1, True
                    return self.get_observation(), reward, done, {'passed':False}
                elif case_content == NO_BOMBS_AROUND:
                    self.reveal_around(coords, reward, done)
            elif case_state == self.REVEAL:
                self.reveal_around(coords, reward, done)
                reward -= 0.01
            else:
                reward -= 0.001
                self.score += reward
                return self.get_observation(), reward, done, {'passed':True}

        elif action_type == self.FLAG:
            if case_state == self.REVEAL:
                reward -= 0.001
            else:
                flaging = 1
                if case_state == self.FLAG:
                    flaging = -1
                    self.state[coords + (1,)] = self.HIDDEN
                else:
                    self.state[coords + (1,)] = self.FLAG
                
                if case_content == self.BOMB:
                    self.flaged_bombs += flaging
                else:
                    self.flaged_empty += flaging

        if self.flaged_bombs == self.n_bombs and self.flaged_empty == 0:
            reward, done = 2 + self.time_left/self.max_time, True

        if np.any(np.logical_and(self.state[..., 0]==9, self.state[..., 1]==1)) or self.done:
            reward, done = -1 + self.time_left/self.max_time + (self.flaged_bombs - self.flaged_empty)/self.n_bombs, True
        
        self.score += reward
        return self.get_observation(), reward, done, {'passed':False}

    def reset(self):
        self.__init__(self.grid_shape, n_bombs=self.n_bombs, impact_size=self.impact_size, max_time=self.max_time, chicken=self.chicken)
        return self.get_observation()
    
    def render(self):
        if not self.pygame_is_init:
            self._init_pygame()
            self.pygame_is_init = True 
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # pylint: disable=E1101
                pygame.quit() # pylint: disable=E1101
        
        # Plot background
        pygame.draw.rect(self.window, (60, 56, 53), (0, 0, self.height, self.width))

        # Plot grid
        for index, state in np.ndenumerate(self.state[..., 1]):
            self._plot_block(index, state)
        
        # Plot infos
        score = self.score_font.render(str(round(self.score, 4)), 1, (255, 10, 10))
        self.window.blit(score, (0.1*self.header_size, 0.8*self.width))

        self.time_left = self.max_time - time() + self.start_time
        time_left = self.num_font.render(str(int(self.time_left+1)), 1, (255, 10, 10))
        self.window.blit(time_left, (0.1*self.header_size, 0.1*self.width))

        potential_bombs_left = self.n_bombs - self.flaged_bombs - self.flaged_empty
        potential_bombs_left = self.num_font.render(str(int(potential_bombs_left)), 1, (255, 255, 10))
        self.window.blit(potential_bombs_left, (0.1*self.header_size, 0.5*self.width))

        pygame.display.flip()
        pygame.time.wait(10)
        if self.done:
            pygame.time.wait(500)
    
    @staticmethod
    def _get_color(n, max_n):
        BLUE_HUE = 0.6
        RED_HUE = 0.0
        HUE = RED_HUE + (BLUE_HUE - RED_HUE) * ((max_n - n) / max_n)**3
        color = 255 * np.array(colorsys.hsv_to_rgb(HUE, 1, 0.7))
        return color

    def _plot_block(self, index, state):
        position = tuple(self.origin + self.scale_factor * self.BLOCK_SIZE * np.array((index[1], index[0])))
        label = None
        if state == self.HIDDEN and not self.done:
            img_key = 'hidden'
        elif state == self.FLAG:
            if not self.done:
                img_key = 'flag'
            else:
                content = self.state[index][0]
                if content == self.BOMB:
                    img_key = 'disabled_mine' if not self.chicken else 'disabled_chicken'
                else:
                    img_key = 'misplaced_flag'
        else:
            content = self.state[index][0]
            if content == self.BOMB:
                if state == self.HIDDEN:
                    img_key = 'mine' if not self.chicken else 'chicken'
                else:
                    img_key = 'exploded_mine' if not self.chicken else 'exploded_chicken'
            else:
                img_key = 'revealed'
                label = self.num_font.render(str(content), 1, self._get_color(content, self.BOMB))

        self.window.blit(self.images[img_key], position)
        if label: self.window.blit(label, position + self.font_offset - (content > 9) * self.decimal_font_offset)

    def _init_pygame(self):
        pygame.init() # pylint: disable=E1101

        # Open Pygame window
        self.scale_factor = 2 * min(12 / self.grid_shape[0], 25 / self.grid_shape[1])
        self.BLOCK_SIZE = 32
        self.header_size = self.scale_factor * 100
        self.origin = np.array([self.header_size, 0])
        self.width =  int(self.scale_factor * self.BLOCK_SIZE * self.grid_shape[0])
        self.height = int(self.scale_factor * self.BLOCK_SIZE * self.grid_shape[1] + self.header_size)
        self.window = pygame.display.set_mode((self.height, self.width))

        # Setup font for numbers
        num_font_size = 20
        self.num_font = pygame.font.SysFont("monospace", int(self.scale_factor * num_font_size))
        self.font_offset = self.scale_factor * self.BLOCK_SIZE * np.array([0.325, 0.15])
        self.decimal_font_offset = self.scale_factor * self.BLOCK_SIZE * np.array([0.225, 0])

        self.score_font = pygame.font.SysFont("monospace", int(self.scale_factor * 12))

        # Load images
        def scale_image(img, scale_factor=self.scale_factor):
            return scale(img, (int(scale_factor*img.get_width()), int(scale_factor*img.get_height())))

        if self.chicken: images_names = ['hidden', 'revealed', 'flag', 'chicken', 'misplaced_flag', 'exploded_chicken', 'disabled_chicken']
        else: images_names = ['hidden', 'revealed', 'flag', 'mine', 'misplaced_flag', 'exploded_mine', 'disabled_mine']

        self.images = {}
        for img_name in images_names:
            with pkg_resources.path(images, img_name + '.png') as path:
                img = pygame.image.load(str(path)).convert()
                self.images[img_name] = scale_image(img)

