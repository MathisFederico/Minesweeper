import learnrl as rl
from minesweeper import MinesweeperEnv, RandomAgent

env = MinesweeperEnv(
    grid_shape=(10, 15),
    bombs_density=0.1,
    n_bombs=None,
    impact_size=3,
    max_time=999,
    chicken=False
)

agent = RandomAgent(env)

pg = rl.Playground(env, agent)
pg.run(5, render=True, verbose=1)
