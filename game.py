import random
import numpy as np

class Game:
    def __init__(self, field = None):
        if field is None:
            self.field = np.zeros([4,4], dtype='int').copy()
        else:
            self.field = field.copy()
        self.score = 0
        self.lose = False

    def generate_number(self):
        zero_cells = [[i,j] for i in range(4) for j in range(4) if self.field[i][j] == 0]
        cell = random.choice(zero_cells)
        self.field[cell[0],[cell[1]]] = random.choice([2]*3 + [4])

    def check_loose(self):
        g = Game(self.field.copy())
        for direction in [[0,1],[0,-1],[1,0],[-1,0]]:
            if g.offset(*direction):
                return
        self.lose = True

    def offset(self, x, y) -> bool:
        """
        X in [-1, 0, 1] Right, 
        Y in [-1, 0 ,1] Down,
        True step is OK
        False step is BAD
        """
        prev_field = self.field.copy()
        x, y = -x, -y
        direction = x
        if y:
            direction = y
            self.field = self.field.T
        
        for i in range(4):
            if x == 1 or y == 1:
                ptr_place = 0
                ptr_st = 0
            else:
                ptr_place = 3
                ptr_st = 3
            fl = True
            while fl:
                while ptr_st >= 0 and ptr_st < 4:
                    if num1 := self.field[i][ptr_st]:
                        break
                    ptr_st += direction
                else:
                    num1 = 0
                    fl = False
                ptr_en = ptr_st + direction
                while ptr_en >= 0 and ptr_en < 4:
                    if num2 := self.field[i][ptr_en]:
                        break
                    ptr_en += direction
                else:
                    num2 = 0
                    fl = False
                if num1 == num2:
                    self.field[i][ptr_place] = num1 + num2
                    self.score += num1 + num2
                    ptr_st = ptr_en + direction
                else:
                    if not num2:
                        self.field[i][ptr_place] = num1 + num2
                        ptr_st = ptr_en + direction
                    else:
                        self.field[i][ptr_place] = num1
                        ptr_st = ptr_en
                ptr_place += direction
                if not fl:
                    while ptr_place >= 0 and ptr_place < 4:
                        self.field[i][ptr_place] = 0
                        ptr_place += direction
                
        if y:
            self.field = self.field.T
        if np.array_equal(prev_field, self.field):
            return False
        return True


    def show_field(self):
        print()
        for row in self.field:
            for item in row:
                print(f"{item:4} ", end='')
            print()

    def one_hot_field(self, temp_field = None, max_classes = 16):
        if temp_field is None:
            temp_field = self.field
        temp_field = temp_field.copy().flatten()
        temp_field = np.log2(temp_field + 0.00001)
        temp_field[temp_field < 0] = 0
        temp_field = temp_field.astype('int')
        
        ans_field = np.zeros((temp_field.size, max_classes), dtype='int')
        ans_field[np.arange(temp_field.size),temp_field] = 1
        return ans_field

    def step(self, arr):
        prev_score = self.score
        order = np.argsort(arr)[::-1]
        case = {0:[0,1],
                1:[0,-1],
                2:[1,0],
                3:[-1,0]}
        for i in order:
            fl = self.offset(*case[i])
            if fl:
                break
        if not fl:
            self.lose = True
        else:
            self.generate_number()
        return self.one_hot_field(), self.score - prev_score, self.lose


    def startGame(self):
        self.generate_number()
        state = True
        while True:
            if state:
                self.generate_number()
            self.check_loose()
            if self.lose:
                print('END')
                break
            print(f'Score: {self.score}')
            self.show_field()
            offset = input().lower()
            case = {'u': [0,-1],
                    'r': [1,0],
                    'd': [0,1],
                    'l': [-1,0]}
            if offset not in case.keys():
                state = False
                continue
            offset = case[offset]
            state = self.offset(*offset)


if __name__ == '__main__':
    game = Game()
    game.generate_number()
    game.generate_number()
    game.generate_number()
    game.generate_number()
    game.show_field()
    print(game.one_hot_field())