import math
import numpy as np
from collections import deque
from a_star import GridMapGraph, a_star_search


# Định nghĩa các trạng thái của thuật toán
class Q:
    START, NORMAL, DEADLOCK, FINISH = range(4)

# Định nghĩa các ô hàng xóm theo 8 hướng
neighbors = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]

class LogicAlgorithm:
    def __init__(self, row_count, col_count):
        self.state = Q.START    # trạng thái của thuật toán có các giá trị Q.START, Q.NORMAL, Q.DEADLOCK, Q.FINISH
        self.weight_map = np.zeros((row_count, col_count))      # bản đồ trọng số 
        self.prob_map = np.zeros((row_count, col_count))    # bản đồ xác suất
        self.direction = 4  # hướng di chuyển của robot (1: lên, 2: trái, 3: phải, 4: xuống)

    def init_weight_map(self, environment):
        """
        Khởi tạo bản đồ trọng số dựa trên môi trường đầu vào.

        Parameters:
            - environment (numpy.ndarray): Môi trường với các ô 0 (ô trống) hoặc 1 (ô chướng ngại vật).
        """
        row_count, col_count = len(environment), len(environment[0])
        for x, row in enumerate(environment):   # Duyệt qua mỗi hàng trong môi trường. x là chỉ số của hàng và row là hàng đó.
            for y, val in enumerate(row):   # Duyệt qua mỗi ô trong hàng. y là chỉ số của ô và val là giá trị của ô đó.
                self.weight_map[x, y] = environment[x, y]   # Gán giá trị của ô trong môi trường cho ô trong bản đồ trọng số.
    
    def set_map(self, map):
        """
        Thiết lập bản đồ trọng số tùy chỉnh.

        Parameters:
            - map (numpy.ndarray): Bản đồ trọng số mới.
        """
        self.weight_map = map

    def set_prob_map(self, map):
        """
        Thiết lập bản đồ xác suất tùy chỉnh.

        Parameters:
            - map (numpy.ndarray): Bản đồ xác suất mới.
        """
        self.prob_map = map

    def four_neighbours(self, cur_pos):
        """
        Tìm các ô hàng xóm trong bốn hướng (lên, xuống, trái, phải).

        Parameters:
            - cur_pos (tuple): Vị trí hiện tại (x, y).

        Returns:
            - list: Danh sách các ô hàng xóm có thể đi đến từ vị trí hiện tại.
        """
        relative_pos = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        neighbours = []
        direction_priority = 0
        for dx, dy in relative_pos:
            direction_priority += 1
            x, y = cur_pos[0] + dx, cur_pos[1] + dy
            if x < 0 or x >= len(self.weight_map):
                continue
            if y < 0 or y >= len(self.weight_map[0]):
                continue
            if self.weight_map[x, y] == 1:
                continue
            neighbours.append((x, y))
        return neighbours
    
    def calculate_zeta(self, cur_pos, nb_pos):
        """
        Tính giá trị zeta dựa trên vị trí hiện tại và vị trí hàng xóm.

        Parameters:
            - cur_pos (tuple): Vị trí hiện tại (x, y).
            - nb_pos (tuple): Vị trí hàng xóm (x, y).

        Returns:
            - float: Giá trị zeta.
        """
        W2 = 1
        W1 = 0.5
        return self.prob_map[cur_pos] * W1 + self.prob_map[nb_pos] * W2

    # def two_step_aution(self, cur_pos):
    #     """
    #     Thực hiện đánh giá và đấu giá bước hai trong thuật toán.

    #     Parameters:
    #         - cur_pos (tuple): Vị trí hiện tại (x, y).

    #     Returns:
    #         - list: Danh sách các giá trị đấu giá và vị trí hàng xóm tương ứng.
    #     """
    #     self.state = Q.NORMAL
    #     cur_neighbours = self.four_neighbours(cur_pos)
    #     bid_value = []
    #     for cur_nb, direction_priority in cur_neighbours:
    #         zeta_nb = self.calculate_zeta(cur_pos, cur_nb)
    #         nb_neighbours = self.four_neighbours(cur_nb)    # Lấy danh sách các ô lân cận của các ô lân cận
    #         max_zeta = -1000
    #         for nb, dp in nb_neighbours:
    #             if nb == cur_pos:
    #                 continue
    #             if self.calculate_zeta(cur_nb, nb) > max_zeta:
    #                 max_zeta = self.calculate_zeta(cur_nb, nb)
    #         if zeta_nb == 0 and max_zeta == 0:
    #             c = math.inf
    #         else:
    #             c = 1 / (zeta_nb + max_zeta)
    #         bid_value.append((c, direction_priority, cur_nb))

    #     return bid_value

    def get_replan_wp(self, cur_pos):
        """
        Lấy danh sách các điểm đích tiềm năng dựa trên đánh giá bước hai.

        Parameters:
            - cur_pos (tuple): Vị trí hiện tại (x, y).

        Returns:
            - tuple: Giá trị đấu giá cao nhất và vị trí hàng xóm tương ứng.
        """
        bid_value_list = self.two_step_evaluation(cur_pos)
        return self.get_score_max(bid_value_list)

    def two_step_evaluation(self, cur_pos):  
        """
        Thực hiện đánh giá hai bước trong thuật toán.

        Parameters:
            - cur_pos (tuple): Vị trí hiện tại (x, y).

        Returns:
            - dict: Danh sách các giá trị đánh giá cho các vị trí hàng xóm.
        """
        self.state = Q.NORMAL
        W1 = 2/3
        W2 = 1/3
        cur_neighbours = self.four_neighbours(cur_pos)
        bid_value = dict()
        for cur_nb in cur_neighbours:
            zeta_nb = self.prob_map[cur_nb]
            nb_neighbours = self.four_neighbours(cur_nb)
            max_zeta = -1000
            for nb in nb_neighbours:
                if nb == cur_pos:
                    continue
                if self.prob_map[nb] > max_zeta:
                    max_zeta = self.prob_map[nb]
            c = 1 - (W1 * zeta_nb + W2 * max_zeta) / 100
            bid_value[cur_nb] = c
        return bid_value
    
    def calculate_reward(self, cur_pos, goal):
        """
        Tính toán phần thưởng dựa trên vị trí hiện tại và điểm đích.

        Parameters:
            - cur_pos (tuple): Vị trí hiện tại (x, y).
            - goal (tuple): Điểm đích (x, y).

        Returns:
            - dict: Danh sách các phần thưởng cho các vị trí hàng xóm.
        """
        cur_neighbors = self.four_neighbours(cur_pos)
        delta_dist = dict()
        graph = GridMapGraph(self.weight_map)
        _, cur_dist = a_star_search(graph, cur_pos, goal)
        for nb in cur_neighbors:
            _, nb_dist = a_star_search(graph, nb, goal)
            if cur_dist - nb_dist > 0:
                delta_dist[nb] = 1
            else:
                delta_dist[nb] = 0
        return delta_dist

    def get_score_max(self, score_dict: dict):
        """
        Lấy giá trị cao nhất và vị trí tương ứng từ danh sách điểm số.

        Parameters:
            - score_dict (dict): Danh sách các điểm số cho các vị trí.

        Returns:
            - tuple: Giá trị điểm số cao nhất và vị trí tương ứng.
        """
        max_score = - math.inf
        max_score_pos = None
        for pos, score in score_dict.items():
            if score > max_score:
                max_score = score
                max_score_pos = pos
        return max_score, max_score_pos
    
    # get next waypoint
    def get_wp(self, current_pos):
        wp = []
        weight_map = self.weight_map

        # not deadlock
        self.state = Q.NORMAL
        wp = self.boustrophedon_moving(current_pos)
        
        if len(wp) > 0: return wp

        # deadlock
        self.state = Q.DEADLOCK
        # return wp
    
        wp = self.get_deadlock_wp(current_pos)

        if len(wp) == 0: 
            self.state = Q.FINISH

        return wp
    
    # def boustrophedon_moving(self, current_pos):
    #     row_count, col_count = len(self.weight_map), len(self.weight_map[0])
    #     (x, y) = current_pos

    #     if (x + 1) < row_count and self.weight_map[x + 1][y] == 0:
    #         return [(x + 1, y)]
    #     if (x - 1) >= 0 and self.weight_map[x - 1][y] == 0:
    #         return [(x - 1, y)]
    #     if y + 1 < col_count and self.weight_map[x][y+1] == 0:
    #         return [(x, y + 1)]
    #     if y - 1 > 0 and self.weight_map[x][y-1] == 0:
    #         return [(x, y - 1)]
    #     return []
    
    def boustrophedon_moving(self, current_pos):
        """
        Thực hiện di chuyển theo kiểu boustrophedon (zigzag) từ vị trí hiện tại.
        Ưu tiên di chuyển theo hướng xuống, lên, trái, phải
        
        Parameters:
            - current_pos (tuple): Vị trí hiện tại (x, y).

        Returns:
            - list: Danh sách các vị trí tiếp theo trong hướng di chuyển boustrophedon.
        """
        row_count, col_count = len(self.weight_map), len(self.weight_map[0])
        (x, y) = current_pos

        if (x + 1) < row_count and self.weight_map[x + 1][y] == 0:  # di chuyển xuống dưới
            return [(x + 1, y)]
        if (x - 1) >= 0 and self.weight_map[x - 1][y] == 0: # di chuyển lên trên
            return [(x - 1, y)]
        if y + 1 < col_count and self.weight_map[x][y+1] == 0:  # di chuyển sang phải
            if self.direction == 3:
                return [(x, y + 1)]
        self.direction = 4
        if y - 1 > 0 and self.weight_map[x][y-1] == 0:  # di chuyển sang trái
            return [(x, y - 1)]
        self.direction = 3
        if y + 1 < col_count and self.weight_map[x][y+1] == 0:  # di chuyển sang phải
            return [(x, y + 1)]
        return []

    def escape_deadlock_path(self, current_pos):
        """
        Tìm đường thoát khỏi tình trạng bế tắc (deadlock) từ vị trí hiện tại.

        Parameters:
            - current_pos (tuple): Vị trí hiện tại (x, y).

        Returns:
            - list: Danh sách các vị trí trong đường thoát khỏi bế tắc.
        """
        weight_map = self.weight_map

        queue = deque()
        visited = []
        parent = dict()
        deadlock_wp = None
        path = []

        queue.append(current_pos)
        visited.append(current_pos)
        parent[current_pos] = -1

        neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        
        flag = True
        while queue:
            if flag == False:
                break
            cur_node = queue.popleft()
            # duyệt neighbors
            for dx, dy in neighbors:
                x, y = cur_node[0] + dx, cur_node[1] + dy
                
                if x < 0 or x >= len(weight_map): continue
                if y < 0 or y >= len(weight_map[0]): continue

                if weight_map[x, y] in (1, 3): continue # obstacle
                elif weight_map[x, y] == 2 or weight_map[x, y] == 4:  # visited
                    if (x, y) not in visited:
                        visited.append((x, y))
                        queue.append((x, y))
                        parent[x, y] = cur_node
                    continue 
                else: # chưa thăm
                    deadlock_wp = (x, y)  # unvisited
                    parent[deadlock_wp] = cur_node
                    flag = False
                    break
        
        if deadlock_wp == None:
            return []
        
        while parent[deadlock_wp] != -1:
            path.append(deadlock_wp)
            deadlock_wp = parent[deadlock_wp]

        return path[::-1]

    def escape_deadlock_dynamic(self, cur_pos, goal):
        """
        Tìm đường thoát khỏi tình trạng bế tắc (deadlock) với sự đánh giá độ ưu tiên động.

        Parameters:
            - cur_pos (tuple): Vị trí hiện tại (x, y).
            - goal (tuple): Điểm đích (x, y).

        Returns:
            - tuple: Vị trí tiếp theo và giá trị đánh giá.
        """
        bid_value_list = self.two_step_evaluation(cur_pos)
        reward_list = self.calculate_reward(cur_pos, goal)
        score_dict = dict()

        for pos in bid_value_list.keys():
            score_dict[pos] = 3/4 * bid_value_list[pos] + 1/4 * reward_list[pos]

        return self.get_score_max(score_dict)

    def get_deadlock_wp(self, current_pos):
        """
        Tìm điểm tiếp theo trong trường hợp bế tắc (deadlock).

        Parameters:
            - current_pos (tuple): Vị trí hiện tại (x, y).

        Returns:
            - list: Danh sách vị trí tiếp theo trong trường hợp bế tắc.
        """
        weight_map = self.weight_map

        queue = deque()
        visited = []

        queue.append(current_pos)
        visited.append(current_pos)

        neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        
        while queue:
            cur_node = queue.popleft()
            # traverse neighbors
            for dx, dy in neighbors:
                x, y = cur_node[0] + dx, cur_node[1] + dy
                
                if x < 0 or x >= len(weight_map): continue
                if y < 0 or y >= len(weight_map[0]): continue

                if weight_map[x, y] in (1, 3): continue # obstacle
                elif weight_map[x, y] == 2 or weight_map[x, y] == 4:  # visited
                    if (x, y) not in visited:
                        visited.append((x, y))
                        queue.append((x, y))
                    continue 
                else: return [(x, y)]  # unvisited
        
        return []

    # def predict(self, current_pos, step_count):
    #     """
    #     Dự đoán các vị trí tiếp theo dựa trên thuật toán và số bước.

    #     Parameters:
    #         - current_pos (tuple): Vị trí hiện tại (x, y).
    #         - step_count (int): Số bước dự đoán.

    #     Returns:
    #         - list: Danh sách các vị trí dự đoán.
    #     """
    #     waypoint_list = [current_pos]

    #     temporary_visited_list = []
    #     for _ in range(step_count):
    #         wp = self.get_wp(current_pos)
    #         if self.state == Q.FINISH:
    #             break
    #         elif self.state == Q.NORMAL:
    #             current_pos = wp[0]
    #             if self.weight_map[current_pos] == 2:
    #                 pass
    #             temporary_visited_list.append(current_pos)
    #             self.weight_map[current_pos] = 2
    #             waypoint_list.append(current_pos)
    #         elif self.state == Q.DEADLOCK:
    #             graph = GridMapGraph(self.weight_map)
    #             path, dist = a_star_search(graph, current_pos, wp[0])
    #             current_pos = path[0]
    #             waypoint_list.append(current_pos)

    #     for pos in temporary_visited_list:
    #         self.weight_map[pos] = 0

    #     return waypoint_list

    def update_explored(self, pos):
        """
        Cập nhật trạng thái của ô đã khám phá.

        Parameters:
            - pos (tuple): Vị trí ô đã khám phá (x, y).
        """
        self.weight_map[pos] = 2

if __name__ == "__main__":
    logic = LogicAlgorithm(7, 7)
    
    environment = np.zeros((7, 7))
    environment[3, 3] = environment[3, 4] = environment[4, 3] = environment[4, 4] = 1
    logic.init_weight_map(environment)

    print(logic.weight_map)
