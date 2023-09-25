import math
import numpy as np
import random
from dynamic_obstacle import DynamicObstacle  # Bạn cần import DynamicObstacle từ một tệp khác

def is_valid_pos(pos, map):
    """
    Kiểm tra xem vị trí 'pos' có hợp lệ trên bản đồ 'map' hay không.

    Parameters:
    - pos (tuple): Vị trí cần kiểm tra dưới dạng (dòng, cột).
    - map (list): Bản đồ dưới dạng danh sách 2D với 0 là vị trí trống và 1 là vị trí có chướng ngại vật.

    Returns:
    - bool: True nếu vị trí hợp lệ và không có chướng ngại vật, False nếu không hợp lệ hoặc có chướng ngại vật.
    """
    row_count, col_count = len(map), len(map[0])
    if 0 <= pos[0] <= row_count - 1 and 0 <= pos[1] <= col_count - 1:
        if map[pos[0]][pos[1]] == 1:
            return False
        return True
    else:
        return False

class DynamicObstacleRandom(DynamicObstacle):
    def __init__(self, start_pos, obs_size, direction, velocity):
        """
        Khởi tạo đối tượng DynamicObstacleRandom.

        Parameters:
        - start_pos (tuple): Vị trí ban đầu của chướng ngại vật dưới dạng (dòng, cột).
        - obs_size (tuple): Kích thước của chướng ngại vật dưới dạng (chiều cao, chiều rộng).
        - direction (int): Hướng ban đầu của chướng ngại vật (1, 2, 3 hoặc 4).
        - velocity (float): Vận tốc của chướng ngại vật.

        Returns:
        - None
        """
        self.cur_row, self.cur_col = start_pos
        self.height, self.width = obs_size
        self.direction = direction
        self.v = 1
        self.w = 0
        self.theta = self.getMovingAngle()
        self.velocity = velocity

    def move_one_step(self, map):
        """
        Di chuyển chướng ngại vật một bước dựa trên hướng hiện tại và bản đồ.

        Parameters:
        - map (list): Bản đồ dưới dạng danh sách 2D với 0 là vị trí trống và 1 là vị trí có chướng ngại vật.

        Returns:
        - None
        """
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
