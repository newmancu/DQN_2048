from game import Game
from test import Test
import numpy as np
np.random.seed(0)

if __name__ == '__main__':
    Test(Game)
    Game().startGame()
