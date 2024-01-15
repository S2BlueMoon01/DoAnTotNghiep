import numpy as np
import math

def is_valid_pos(pos, map):
    """
    Kiểm tra xem vị trí 'pos' có hợp lệ trên bản đồ 'map' hay không.

    Parameters:
    - pos (tuple): Vị trí cần kiểm tra dưới dạng (dòng, cột).
    - map (list): Bản đồ dưới dạng danh sách 2D với 0 là vị trí trống và 1 là vị trí có chướng ngại vật.

    Returns:
    - bool: True nếu vị trí hợp lệ, False nếu không hợp lệ.
    """
    row_count, col_count = len(map), len(map[0])
    if 0 <= pos[0] <= row_count - 1 and 0 <= pos[1] <= col_count - 1:
        return True
    else:
        return False

class DynamicObstacle():
    def __init__(self, start_pos, obs_size, direction, velocity):
        """
        Khởi tạo đối tượng DynamicObstacle.

        Parameters:
        - start_pos (tuple): Vị trí ban đầu của chướng ngại vật dưới dạng (dòng, cột).
        - obs_size (tuple): Kích thước của chướng ngại vật dưới dạng (chiều cao, chiều rộng).
        - direction (int): Hướng ban đầu của chướng ngại vật (1, 2, 3 hoặc 4).  # 1: up, 2: left, 3: right, 4: down
        - velocity (float): Vận tốc của chướng ngại vật.

        Returns:
        - None
        """
        self.cur_row, self.cur_col = start_pos
        self.height, self.width = obs_size
        self.direction = direction
        self.v = 1  # vận tốc dọc theo hướng di chuyển
        self.w = 0  # vận tốc góc
        self.theta = self.getMovingAngle()      # góc di chuyển của chướng ngại vật
        self.velocity = velocity    # vận tốc của chướng ngại vật

    def move_one_step(self, map):
        """
        Di chuyển chướng ngại vật một bước dựa trên hướng hiện tại và bản đồ.

        Parameters:
        - map (list): Bản đồ dưới dạng danh sách 2D với 0 là vị trí trống và 1 là vị trí có chướng ngại vật.

        Returns:
        - None
        """
        if self.direction == 1:
            if self.cur_row == 0:
                self.direction = 4
            else:
                flag = False
                for i in range(self.width):
                    if map[self.cur_row - 1][self.cur_col + i] == 1:
                        self.direction = 4
                        flag = True
                        break
                if not flag:
                    self.cur_row -= 1
        
        elif self.direction == 4:
            if self.cur_row + self.height == len(map):
                self.direction = 1
            else:
                flag = False
                for i in range(self.width):
                    if map[self.cur_row + self.height][self.cur_col + i] == 1:
                        self.direction = 1
                        flag = True
                        break
                if not flag:
                    self.cur_row += 1
        
        elif self.direction == 2:
            if self.cur_col == 0:
                self.direction = 3
            else:
                flag = False
                for i in range(self.height):
                    if map[self.cur_row + i][self.cur_col - 1] == 1:
                        self.direction = 3
                        flag = True
                        break
                if not flag:
                    self.cur_col -= 1
        
        elif self.direction == 3:
            if self.cur_col + self.width == len(map[0]):
                self.direction = 2
            else: 
                flag = False
                for i in range(self.height):
                    if map[self.cur_row + i][self.cur_col + self.width] == 1:
                        self.direction = 2
                        flag = True
                        break
                if not flag:
                    self.cur_col += 1
        
        self.getMovingAngle()
    
    def getMovingAngle(self):
        """
        Trả về góc di chuyển hiện tại của chướng ngại vật.

        Returns:
        - float: Góc di chuyển (rad).
        """
        if self.direction == 1: # direction = 1: up
            self.theta = math.pi    
        elif self.direction == 2:   # direction = 2: left
            self.theta = - math.pi / 2
        elif self.direction == 3:   # direction = 3: right
            self.theta = math.pi / 2
        elif self.direction == 4:   # direction = 4: down
            self.theta = 0
        return self.theta
    
    def get_current_occupy_positions(self):
        """
        Trả về danh sách các vị trí hiện tại mà chướng ngại vật đang chiếm giữ.

        Returns:
        - list: Danh sách các vị trí dưới dạng (dòng, cột).
        """
        occupy_list = []
        for dx in range(self.height):
            for dy in range(self.width):
                occupy_list.append((self.cur_row + dx, self.cur_col + dy))
        return occupy_list

    def get_pos(self):
        """
        Trả về vị trí trung bình của chướng ngại vật.

        Returns:
        - tuple: Vị trí trung bình dưới dạng (dòng, cột).
        """
        return np.mean(self.get_current_occupy_positions(), axis=0)
    
    # def predict(self, delta_t, map):
    #     """
    #     Dự đoán quỹ đạo di chuyển của chướng ngại vật trong khoảng thời gian 'delta_t'.

    #     Parameters:
    #     - delta_t (float): Khoảng thời gian dự đoán (s).
    #     - map (list): Bản đồ dưới dạng danh sách 2D với 0 là vị trí trống và 1 là vị trí có chướng ngại vật.

    #     Returns:
    #     - list: Danh sách các vị trí dự đoán trong khoảng thời gian 'delta_t'.
    #     """
    #     travel_dist = self.v * delta_t
    #     wp_list = []
    #     (y, x) = (self.cur_row, self.cur_col)
    #     for _ in range(math.ceil(travel_dist)):
    #         if self.direction == 1:
    #             y += 1
    #             if not is_valid_pos((y, x), map) or map[y, x] == 1:
    #                 (y, x) = (self.cur_row + 1, self.cur_col)
    #                 self.direction = 4
    #             wp_list.append((y, x))
    #         elif self.direction == 2:
    #             x -= 1
    #             if not is_valid_pos((y, x), map) or map[y, x] == 1:
    #                 (y, x) = (self.cur_row, self.cur_col + 1)
    #                 self.direction = 3
    #             wp_list.append((y, x))
    #         elif self.direction == 3:
    #             x += 1
    #             if not is_valid_pos((y, x), map) or map[y, x] == 1:
    #                 (y, x) = (self.cur_row, self.cur_col - 1)
    #                 self.direction = 2
    #             wp_list.append((y, x))
    #         elif self.direction == 4:
    #             y += 1
    #             if not is_valid_pos((y, x), map) or map[y, x] == 1:
    #                 (y, x) = (self.cur_row - 1, self.cur_col)
    #                 self.direction = 1
    #             wp_list.append((y, x))
    #     return wp_list
