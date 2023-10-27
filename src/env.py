import numpy as np
import random
import pygame
import torch

import gym
from gym import spaces

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 48}

    pieces = [
        np.array([[1], # BAR
                  [1],
                  [1],
                  [1]]),
        np.array([[0,1], # J
                  [0,1],
                  [1,1]]),
        np.array([[1,0], # L
                  [1,0],
                  [1,1]]),
        np.array([[1,1], # Square
                  [1,1]]),
        np.array([[0,1,1], # S
                  [1,1,0]]),
        np.array([[1,1,0], # Z
                  [0,1,1]]),
        np.array([[0,1,0], # T
                  [1,1,1]]),
    ]

    def __init__(self, render_mode=None, shape=(20,10), drop_freq=3, max_length=1e4):
        self.shape = shape  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(0, 1, shape=self.shape, dtype=int)

        # corresponding to "moveLeft", "down", "moveRight", "rotLeft", "rotRight"
        self.action_space = spaces.Discrete(5)

        # Number of steps allowed before dropping one block automatically
        self.drop_feq = drop_freq

        self.max_ep_length = max_length

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def _get_obs(self) -> np.ndarray:
        obs = self.board.copy()
        y,x = self.piece_loc
        h,w = self.piece.shape
        obs[y:y+h, x:x+w] += self.piece
        return obs


    def _get_info(self) -> dict:
        return {}


    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Set playfield to be blank
        self.board = np.zeros(self.shape, dtype=int)

        # Timestep counter
        self.it = 0

        self.reset_piece()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info


    def reset_piece(self):

        # Select a random piece
        self.piece:np.ndarray = random.choice(self.pieces).copy()

        # Set position to be top row in the middle of the field
        m = self.shape[1]//2 - self.piece.shape[1]//2
        # Location is top-left anchor point
        self.piece_loc = np.array([0, m], dtype=int)


    @torch.no_grad()
    def step(self, action:int) -> tuple[np.ndarray, float, bool, dict]:

        reward = 1.

        self.it += 1

        y,x = self.piece_loc
        h,w = self.piece.shape

        if action == 1 or self.it % self.drop_feq == 0: # moveDown
            if y + h >= self.shape[0] or (self.board[y+1:y+1+h, x:x+w] & self.piece).any():

                highestblock = self.board.sum(axis=0).max()

                # Place piece at current location
                self.board[y:y+h, x:x+w] += self.piece

                keepRows = self.board.sum(axis=1) != self.shape[1]
                nCleared = np.count_nonzero(~keepRows)

                # penalty of 10 every block higher than the previous
                reward -= 10 * abs(self.board.sum(axis=0).max() - highestblock)

                reward += 100 * nCleared

                if nCleared != 0:
                    print("CLEARED ROW!", nCleared)

                board = np.zeros_like(self.board)
                board[-keepRows.sum():] = self.board[keepRows]
                self.board = board

                # Reset piece
                self.reset_piece()
            else:
                self.piece_loc[0] += 1
                y+=1

        y,x = self.piece_loc
        h,w = self.piece.shape

        if action == 0: # moveLeft
            if x > 0 and not (self.board[y:y+h, x-1:x-1+w] & self.piece).any():
                self.piece_loc[1] -= 1
                x-=1
        elif action == 2: # moveRight
            if x + w < self.shape[1] and not (self.board[y:y+h, x+1:x+1+w] & self.piece).any():
                self.piece_loc[1] += 1
                x+=1
        elif action == 3: # rotRight
            piece = np.rot90(self.piece, k=3)
            h,w = piece.shape
            if y + h <= self.shape[0] and x + w <= self.shape[1] \
            and not (self.board[y:y+h, x:x+w] & piece).any():
                self.piece = piece
        elif action == 4: # rotLeft
            piece = np.rot90(self.piece, k=1)
            h,w = piece.shape
            if y + h <= self.shape[0] and x + w <= self.shape[1] \
            and not (self.board[y:y+h, x:x+w] & piece).any():
                self.piece = piece

        # Clip position
        # anchor_max = np.array(self.shape) - np.array(self.piece.shape)
        # self.piece_loc = self.piece_loc.clip(0, anchor_max)

        # Top 3 rows need to be clear of any placed blocks
        terminated = self.board[:4].any() or self.it > self.max_ep_length

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        if self.render_mode == "human":
            if self.window is None and self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
            if self.clock is None and self.render_mode == "human":
                self.clock = pygame.time.Clock()

            board = self._get_obs()

            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((255, 255, 255))
            pix_square_size = min(
                self.window_size / self.shape[0],
                self.window_size / self.shape[1]
            )  # The size of a single grid square in pixels

            for y in range(board.shape[0]):
                for x in range(board.shape[1]):
                    # First we draw the target
                    if board[y,x]:
                        pygame.draw.rect(
                            canvas,
                            (255, 0, 0),
                            pygame.Rect(
                                x*pix_square_size,y*pix_square_size,
                                pix_square_size, pix_square_size
                            ),
                        )

            # Finally, add some gridlines
            for y in range(self.shape[0] + 1):
                pygame.draw.line(
                    canvas,
                    0,
                    (0, pix_square_size * y),
                    (pix_square_size*self.shape[1], pix_square_size * y),
                    width=3,
                )

            for x in range(self.shape[1] + 1):
                pygame.draw.line(
                    canvas,
                    0,
                    (pix_square_size * x, 0),
                    (pix_square_size * x, pix_square_size*self.shape[0]),
                    width=3,
                )

            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return self._get_obs()


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    env = TetrisEnv(render_mode="human", drop_freq=3)

    obs,_ = env.reset()
    done = False

    AUTODROP = pygame.USEREVENT+1
    pygame.time.set_timer(AUTODROP, 1200)

    while not done:
        a = -1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT: a = 0
                elif event.key == pygame.K_DOWN: a = 1
                elif event.key == pygame.K_RIGHT: a = 2
                elif event.key == pygame.K_x: a = 3
                elif event.key == pygame.K_z: a = 4
                elif event.key == pygame.K_q: done = True

                if 0 <= a <= 4:
                    obs, r, done, _ = env.step(a)
                    print("Tick:", env.it, "Reward:",r)

            elif event.type == AUTODROP:
                obs, r, done, _ = env.step(1)
                print("AutoDrop, Reward:",r)

        if pygame.key.get_pressed()[pygame.K_q]:
            break
    print("Done.")
