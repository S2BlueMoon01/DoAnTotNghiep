import numpy as np
import math

def is_valid_pos(pos, map):
    row_count, col_count = len(map), len(map[0])
    if 0 <= pos[0] <= row_count - 1 and 0 <= pos[1] <= col_count - 1:
        return True
    else:
        return False

class DynamicObstacle():
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
        if self.direction == 1:
            if self.cur_row == 0:
                self.direction = 4
                # self.cur_row += 1
            else:
                flag = False
                for i in range(self.width):
                    if map[self.cur_row - 1][self.cur_col + i] == 1:
                        self.direction = 4
                        # self.cur_row += 1
                        flag = True
                        break
                if flag == False:
                    self.cur_row -= 1
        
        elif self.direction == 4:
            if self.cur_row + self.height == len(map):
                self.direction = 1
                # self.cur_row -= 1
            else:
                flag = False
                for i in range(self.width):
                    if map[self.cur_row + self.height][self.cur_col + i] == 1:
                        self.direction = 1
                        # self.cur_row -= 1
                        flag = True
                        break
                if flag == False:
                    self.cur_row += 1
        
        elif self.direction == 2:
            if self.cur_col == 0:
                self.direction = 3
                # self.cur_col += 1
            else:
                flag = False
                for i in range(self.height):
                    if map[self.cur_row + i][self.cur_col - 1] == 1:
                        self.direction = 3
                        # self.cur_col += 1
                        flag = True
                        break
                if flag == False:
                    self.cur_col -= 1
        
        elif self.direction == 3:
            if self.cur_col + self.width == len(map[0]):
                self.direction = 2
                # self.cur_col -= 1
            else: 
                flag = False
                for i in range(self.height):
                    if map[self.cur_row + i][self.cur_col + self.width] == 1:
                        self.direction = 2
                        # self.cur_col -= 1
                        flag = True
                        break
                if flag == False:
                    self.cur_col += 1
        
        self.getMovingAngle()
    
    def getMovingAngle(self):
        if self.direction == 1:
            self.theta = math.pi
        elif self.direction == 2:
            self.theta = - math.pi / 2
        elif self.direction == 3:
            self.theta = math.pi / 2
        elif self.direction == 4:
            self.theta = 0
        return self.theta
    
    def get_current_occupy_positions(self):
        occupy_list = []
        for dx in range(self.height):
            for dy in range(self.width):
                occupy_list.append((self.cur_row + dx, self.cur_col + dy))
        
        return occupy_list

    def get_pos(self):
        return np.mean(self.get_current_occupy_positions(), axis=0)
    
    def predict(self, delta_t, map):
        travel_dist = self.v * delta_t
        wp_list = []
        (y, x) = (self.cur_row, self.cur_col)
        for _ in range(math.ceil(travel_dist)):
            if self.direction == 1:
                y += 1
                if not is_valid_pos((y, x), map) or map[y, x] == 1:
                    (y, x) = (self.cur_row + 1, self.cur_col)
                    self.direction = 4
                    # break
                wp_list.append((y, x))
            elif self.direction == 2:
                x -= 1
                if not is_valid_pos((y, x), map) or map[y, x] == 1:
                    # break
                    (y, x) = (self.cur_row, self.cur_col + 1)
                    self.direction = 3
                wp_list.append((y, x))
            elif self.direction == 3:
                x += 1
                if not is_valid_pos((y, x), map) or map[y, x] == 1:
                    # break
                    (y, x) = (self.cur_row, self.cur_col - 1)
                    self.direction = 2
                wp_list.append((y, x))
            elif self.direction == 4:
                y += 1
                if not is_valid_pos((y, x), map) or map[y, x] == 1:
                    # break
                    (y, x) = (self.cur_row - 1, self.cur_col)
                    self.direction = 1
                wp_list.append((y, x))
        return wp_list