# import pygame as pg
from pprint import pformat
import random
import numpy as np
import torch
# random.seed(2)

class Game:

  def __init__(self, fs=4, pd=[2,2,2,2,2,2,2,4]):

    self.move_list = ((-1,0), (1,0), (0,-1), (0,1))
    self.pd = pd
    
    self.fs = fs
    self.clear(fs)

  def clear(self, fs=4):
    self.cur_move = 0
    self.field =  [[0]*fs for i in range(fs)]
    self.end = False
    self.score = 0
    self.reward = 0
    
    self.prev_score = self.score
    self.prev_end = self.end
    self.prev_field = [row[:] for row in self.field]

  def reset(self, fs=4):
    self.clear()
    self.gen_cell()

  def get_field(self):
    return torch.tensor(np.array(self.field).flatten(), dtype=torch.float32)

  def swipe(self, x, y):
    """
      x=1 - right
      x=-1 - left
      y=1 - down
      y=-1 - up
    """
    if x > 0 and not y:
      fs = len(self.field)
      for k in range(fs):
        i = fs - 1
        while i > 0:
          j = i - 1
          was = False
          while j >= 0 and not was:
            if not self.field[k][i]:
              if self.field[k][j]:
                self.field[k][i] = self.field[k][j]
                self.field[k][j] = 0
            else:
              if self.field[k][i] == self.field[k][j]:
                self.field[k][i] += self.field[k][j]
                self.score += self.field[k][i]
                self.field[k][j] = 0
                was = True
              if self.field[k][i] != self.field[k][j] and self.field[k][j]:
                was = True
            j -= 1
          i -= 1

    if x < 0 and not y:
      fs = len(self.field)
      for k in range(fs):
        i = 0
        while i < fs- 1:
          j = i + 1
          was = False
          while j < fs and not was:
            if not self.field[k][i]:
              if self.field[k][j]:
                self.field[k][i] = self.field[k][j]
                self.field[k][j] = 0
            else:
              if self.field[k][i] == self.field[k][j]:
                self.field[k][i] += self.field[k][j]
                self.score += self.field[k][i]
                self.field[k][j] = 0
                was = True
              if self.field[k][i] != self.field[k][j] and self.field[k][j]:
                was = True
            j += 1
          i += 1

    if not x and y > 0:
      fs = len(self.field)
      for k in range(fs):
        i = fs - 1
        while i > 0:
          j = i - 1
          was = False
          while j >= 0 and not was:
            if not self.field[i][k]:
              if self.field[j][k]:
                self.field[i][k] = self.field[j][k]
                self.field[j][k] = 0
            else:
              if self.field[i][k] == self.field[j][k]:
                self.field[i][k] += self.field[j][k]
                self.score += self.field[i][k]
                self.field[j][k] = 0
                was = True
              if self.field[i][k] != self.field[j][k] and self.field[j][k]:
                was = True
            j -= 1
          i -= 1

    if not x and y < 0:
      fs = len(self.field)
      for k in range(fs):
        i = 0
        while i < fs + 1:
          j = i + 1
          was = False
          while j < fs and not was:
            if not self.field[i][k]:
              if self.field[j][k]:
                self.field[i][k] = self.field[j][k]
                self.field[j][k] = 0
            else:
              if self.field[i][k] == self.field[j][k]:
                self.field[i][k] += self.field[j][k]
                self.score += self.field[i][k]
                self.field[j][k] = 0
                was = True
              if self.field[i][k] != self.field[j][k] and self.field[j][k]:
                was = True
            j += 1
          i += 1

  def gen_cell(self):
    fs = len(self.field)
    empts = [(index//fs, index%fs) for index in range(fs*fs) if not self.field[index//fs][index%fs]]
    if not len(empts):
      self.end = True
      return None
    cell = random.choice(empts)
    val = random.choice(self.pd)
    self.field[cell[0]][cell[1]] = val

  def display(self):
    for row in self.field:
      for x in row:
        print(f"{x:6}", end=' ')
      print()

  def loadField(self, field):
    self.field = field

  def _dif_fields(self, sfield, efield):
    for i in range(len(sfield)):
      for j in range(len(efield)):
        if sfield[i][j] != efield[i][j]:
          return True
    
    return False

  def check_end(self):
    moves = ((-1,0), (1,0), (0,-1), (0,1))
    end = True
    start_field = [row[:] for row in self.field]
    start_score = self.score
    for move in moves:
      self.field = [row[:] for row in start_field]
      self.swipe(*move)
      if self._dif_fields(start_field, self.field):
        end = False
    self.field = start_field
    self.score = start_score
    self.end = end

  def step(self, x, y):
    if self.end:
      return None
    self.swipe(x,y)
    self.gen_cell()
    self.check_end()

    return self.field, self.score, self.end

  def step2(self, moves):
    prev_field = [row[:] for row in self.field]
    prev_score = self.score
    # print('cur')
    # print(prev_field)
    if not self.end:
      for move in moves:
        self.field = [row[:] for row in prev_field]
        self.score = prev_score
        self.swipe(*(self.move_list[move]))
        # print(self.field)
        if self._dif_fields(self.field, prev_field):
          self.end = False
          self.gen_cell()
          return self.field, self.score, self.end
      self.end = True
    return self.field, self.score, self.end

  def step3(self, moves):
    if self.end:
      return self.get_field(), self.score, self.end
    pfield = [row[:] for row in self.field]
    self.swipe(*moves)
    self.gen_cell()
    self.check_end()
    if not self._dif_fields(pfield, self.field):
      self.end = True
    return self.get_field(), self.score, self.end

  def step4(self, action):
    if self.end:
      return self.get_field(), -10, self.end, self.score
    pfield = [row[:] for row in self.field]
    self.reward = self.score
    self.swipe(*self.move_list[action])
    self.gen_cell()
    self.cur_move += 1
    self.reward = (self.score - self.reward)
    if not self._dif_fields(pfield, self.field):
      # self.end = True
      self.reward = -10
    self.check_end()
    return self.get_field(), self.reward, self.end, self.score


  def save_game(self, field=None, score=None, end=None):
    if score is None:
      self.prev_score = self.score
    else:
      self.prev_score = score
    
    if end is None:
      self.prev_end = self.end
    else:
      self.prev_end = end
    
    if field is None:
      self.prev_field = [row[:] for row in self.field]
    else:
      field = self.getFieldFrom1d(field)

  def getFieldFrom1d(self, field):
    f = [[field[self.fs*j + i] for i in range(self.fs) ] for j in range(self.fs)]
    return f

  def restore_game(self):
    self.score = self.prev_score
    self.end = self.prev_end
    self.field = self.prev_field

  def start_game(self):
    self.gen_cell()
    finished = False
    while not finished:
      pass



if __name__ == '__main__':
  game = Game()
  # game.swipe(0, 1)
  field = [
    [2,2,2,2],
    [0,0,0,2],
    [0,2,0,4],
    [0,0,2,0],
  ]
  # nf = np.array(field).flatten()
  # print(nf.)
  print(game.end)
  game.loadField(field)
  game.display()
  print()
  game.step(1,0)

  # game.swipe(0, -1)
  game.display()
  print(game.end)