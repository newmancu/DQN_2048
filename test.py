import numpy as np
from game import Game

class Test:
    def __init__(self, obj):
        if obj == Game:
            self.test_game()

    def test_game(self):
        g = Game(np.array([
                        [2,2,0,4],
                        [2,2,2,0],
                        [0,2,0,2],
                        [0,0,0,0]]))
        g.offset(1,0)
        assert np.array_equal(g.field, np.array([
                        [0,0,4,4],
                        [0,0,2,4],
                        [0,0,0,4],
                        [0,0,0,0]]))
        print('Test 1 OK')


        g = Game(np.array([
                        [2,2,2,4],
                        [2,0,2,8],
                        [2,2,2,2],
                        [0,4,4,2]]))
        g.offset(0,1)
        assert np.array_equal(g.field, np.array([
                        [0,0,0,0],
                        [0,0,2,4],
                        [2,4,4,8],
                        [4,4,4,4]]))
        print('Test 2 OK')
        

        g = Game(np.array([
                        [2,2,2,4],
                        [2,0,2,8],
                        [2,2,2,2],
                        [0,4,4,2]]))
        g.offset(-1,0)
        assert np.array_equal(g.field, np.array([
                        [4,2,4,0],
                        [4,8,0,0],
                        [4,4,0,0],
                        [8,2,0,0]]))
        print('Test 3 OK')


        g = Game(np.array([
                        [2,2,2,4],
                        [2,0,2,8],
                        [2,2,2,2],
                        [0,4,4,2]]))
        g.offset(0,-1)
        assert np.array_equal(g.field, np.array([
                        [4,4,4,4],
                        [2,4,2,8],
                        [0,0,4,4],
                        [0,0,0,0]]))
        print('Test 4 OK')
        
        g = Game(np.array([
                        [4,4,4,4],
                        [2,2,2,2],
                        [0,0,0,0],
                        [0,0,0,0]]))
        state = g.offset(0,-1)
        assert np.array_equal(g.field, np.array([
                        [4,4,4,4],
                        [2,2,2,2],
                        [0,0,0,0],
                        [0,0,0,0]]))
        assert not state
        print('Test 5 OK')

if __name__ == '__main__':
    Test(Game)