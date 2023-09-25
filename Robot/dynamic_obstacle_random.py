import math
import numpy as np
import random
from dynamic_obstacle import DynamicObstacle

def is_valid_pos(pos, map):
    row_count, col_count = len(map), len(map[0])
    if 0 <= pos[0] <= row_count - 1 and 0 <= pos[1] <= col_count - 1:
        if map[pos] == 1:
            return False
        return True
    else:
        return False

class DynamicObstacleRandom(DynamicObstacle):
    def __init__(self, start_pos, obs_size, direction, velocity):
        self.cur_row, self.cur_col = start_pos
        self.height, self.width = obs_size
        self.direction = direction
        self.v = 1
        self.w = 0
        self.theta = self.getMovingAngle()
        self.velocity = velocity
        
    '''
    direction: 
          1     
    2  cur_pos  3
          4        
    '''    

    def move_one_step(self, map):
        r = random.uniform(0, 1)
        if 0 <= r <= 0.5:
            if 0 <= r <= 0.1:
                self.theta = (self.theta + math.pi / 4) % (2 * math.pi)
            else:
                self.theta = (self.theta - math.pi / 4) % (2 * math.pi)

            n = round(self.theta / (1/4 * math.pi))
            self.theta = n * math.pi / 4
        else:
            rotate = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
            n = round(self.theta / (1/4 * math.pi))
            x = self.cur_row + rotate[n][0]
            y = self.cur_col + rotate[n][1]
            if not is_valid_pos((x, y), map):
                return
            self.cur_row, self.cur_col = x, y
    