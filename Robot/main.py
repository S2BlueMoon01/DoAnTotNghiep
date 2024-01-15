import math
import numpy as np
from collections import deque
import pygame as pg
import time
from copy import deepcopy

# from a_star import GridMapGraph, a_star_search
from dynamic_obstacle import DynamicObstacle
from dynamic_obstacle_random import DynamicObstacleRandom
from logic import LogicAlgorithm, Q
from grid_map import Grid_Map

# Phạm vi cảm biến của robot
VISION_SENSOR_RANGE = 5

# coverage:             1 unit of energy / cell width
# advance & retreat:    0.5 unit of energy / cell width

# Năng lương của robot
ENERGY_CAPACITY = math.inf
# ENERGY_CAPACITY = 100

ui = Grid_Map(VISION_SENSOR_RANGE)  # Tạo một đối tượng lưới bản đồ

# Đọc và chỉnh sửa bản đồ từ file 'test_4.txt' và lấy vị trí ban đầu của pin
ui.read_map('map/scenario_2/test_1.txt')
ENVIRONMENT, battery_pos = ui.edit_map()

# Số hàng và số cột trong môi trường
ROW_COUNT = len(ENVIRONMENT)
COL_COUNT = len(ENVIRONMENT[0])

# Tốc độ khung hình
FPS = 40

# Số lần lấy mẫu cho tính toán xác suất
NUMS_SAMPLE = 5000

# Ngưỡng xác suất tối thiểu
MIN_PROB_THRESHOLD = 3

# Các biến để theo dõi chi tiết hành trình của robot
total_travel_length = 0 # Tổng chiều dài hành trình
coverage_length, retreat_length, advance_length = 0, 0, 0   # Chiều dài hành trình duyệt, rút lui, tiến lên
coverage_ratio = 0  # Tỷ lệ phủ sóng
nums_cell_repetition = 0   # Số lần lặp lại ô
repetition_rate = 0 # Tỷ lệ lặp lại
return_charge_count = 1 # Số lần trở về sạc pin
count_waiting = 0  # Số lần chờ


total_coverable_cells, nums_covered_cells = 0, 0    # Tổng số ô có thể duyệt, số ô đã duyệt

# Danh sách đối tượng vật thể động
dynamic_obs_list : list[DynamicObstacle] = []

# Thêm một đối tượng vật thể động vào danh sách, với các thông số cụ thể
# dynamic_obs_list.append(DynamicObstacle((3, 6), (2, 1), 4, 10))
dynamic_obs_list.append(DynamicObstacle((23, 12), (1, 3), 3, 10))    # Chướng ngại vật động 1 1.25v
# dynamic_obs_list.append(DynamicObstacle((23, 12), (1, 3), 3, 8))    # Chướng ngại vật động 1 1.25v
# dynamic_obs_list.append(DynamicObstacle((23, 12), (1, 3), 3, 12))    # Chướng ngại vật động 1 5/6v
# dynamic_obs_list.append(DynamicObstacle((12, 15), (2, 2), 3, 8))    # Chướng ngại vật động 2 1.25v
# dynamic_obs_list.append(DynamicObstacle((12, 15), (2, 2), 1, 12))    # Chướng ngại vật động 2 5/6v
# dynamic_obs_list.append(DynamicObstacle((25, 10), (1, 2), 1, 8))    # Chướng ngại vật động 3 1.25v
# dynamic_obs_list.append(DynamicObstacle((25, 10), (1, 3), 3, 12))    # Chướng ngại vật động 3 5/6v
# dynamic_obs_list.append(DynamicObstacle((3, 8), (1, 2), 1, 8))    # Chướng ngại vật động 4 1.25v
# dynamic_obs_list.append(DynamicObstacle((3, 8), (21, 2), 4, 12))    # Chướng ngại vật động 4 5/6v
# dynamic_obs_list.append(DynamicObstacle((26, 13), (1, 3), 3, 8))    # Chướng ngại vật động 5 1.25v
# dynamic_obs_list.append(DynamicObstacle((26, 13), (1, 3), 4, 12))    # Chướng ngại vật động 5 5/6v


dynamic_obs_list.append(DynamicObstacleRandom((26, 13), (1, 2), 4, 10))    # Chướng ngại vật động 5 5/6v




# Danh sách để theo dõi thời gian chờ liên tiếp của các vật thể động
consecutive_wait_obs_list = [0] * len(dynamic_obs_list)


def check_valid_pos(pos):
    """
    Kiểm tra tính hợp lệ của một vị trí trong môi trường của robot.

    Parameters:
        - pos: Vị trí cần kiểm tra, gồm hàng (row) và cột (col).

    Returns:
        - True nếu vị trí là hợp lệ, False nếu không hợp lệ.
    """
    row, col = pos
    # if row < 0 or row >= ROW_COUNT: return False
    # if col < 0 or col >= COL_COUNT: return False
    return row >= 0 and row < ROW_COUNT and col >= 0 and col < COL_COUNT

def sign(n):
    """
    Trả về dấu của một số.

    Parameters:
        - n: Số cần kiểm tra dấu.

    Returns:
        - 1 nếu số dương, -1 nếu số âm, 0 nếu số bằng không.
    """
    return int(np.sign(n))


class Robot:
    def __init__(self, battery_pos, map_row_count, map_col_count):
        """
        Hàm khởi tạo của lớp Robot.

        Parameters:
            - battery_pos: Vị trí ban đầu của pin.
            - map_row_count: Số hàng trong bản đồ.
            - map_col_count: Số cột trong bản đồ.

        Returns:
            None
        """
        self.logic = LogicAlgorithm(map_row_count, map_col_count)
        '''
            map: 
                0 : chưa thăm
                1 : chướng ngại vật
                2 : đã thăm
                3 : chướng ngại vật động
        '''
        self.static_map = None  # Bản đồ tĩnh
        self.dynamic_map = None # Bản đồ động
        self.predict_map = None # Bản đồ dự đoán
        self.prob_map = None    # Bản đồ xác suất
        self.seen_map = None    # Bản đồ đã thăm

        self.current_pos = battery_pos  # Vị trí hiện tại của robot

        # Góc giữa hướng của robot và trục từ trái sang phải tính bằng radian [0, 2pi)
        self.angle = - math.pi / 2  # Góc quay của robot 
        self.velocity = 10  # Vận tốc của robot

        self.battery_pos = battery_pos  # Vị trí của pin
        self.energy = ENERGY_CAPACITY   # Năng lượng của robot

        self.move_status = 0 # 0: di chuyển bình thường, 1: rút lui, 2: sạc pin, 3: tiến lên
        self.cache_path = [] # lưu trữ đường dẫn tạm thời (ví dụ: rút lui, tiến lên)
        self.repeated_cells = []    # Danh sách các ô đã lặp lại

        self.obs_prev_detected_dict = dict()    # Danh sách các chướng ngại vật động phát hiện được trước đó
        self.obs_detected_dict = dict()        # Danh sách các chướng ngại vật động phát hiện được hiện tại
        self.scan_freq = 2  # Tần số quét của robot
        # Hệ số alpha Tạo mẫu cho vị trí tiềm năng của chướng ngại vật dựa trên phương pháp lấy mẫu.
        self.alpha_1 = 0.3  
        self.alpha_2 = 0.7
        self.alpha_3 = 0.3
        self.alpha_4 = 0.7
    
    def init_static_map(self, environment):
        """
        Hàm khởi tạo bản đồ tĩnh.

        Parameters:
            - environment: Bản đồ môi trường ban đầu.

        Returns:
            None
        """
        row_count, col_count = len(environment), len(environment[0])    # Số hàng và số cột của bản đồ
        self.static_map = deepcopy(environment) # Bản đồ tĩnh
        self.dynamic_map = deepcopy(environment)    # Bản đồ động
        self.predict_map = deepcopy(environment)    # Bản đồ dự đoán
        self.prob_map = deepcopy(environment)   # Bản đồ xác suất
        self.seen_map = deepcopy(environment)   # Bản đồ đã thăm
        self.predict_map[self.battery_pos] = self.dynamic_map[self.battery_pos] = 2 # Đặt vị trí pin là đã thăm
        self.seen_map[self.battery_pos] = 2 # Đặt vị trí pin là đã thăm

        self.logic.init_weight_map(environment) # Khởi tạo bản đồ trọng số 

    def update_dynamic_map(self, loop_count):
        """
        Cập nhật bản đồ động.

        Parameters:
            - loop_count: Số lần lặp.

        Returns:
            None
        """
        row_count, col_count = len(self.static_map), len(self.static_map[0])    # Số hàng và số cột của bản đồ

        for x in range(row_count):
            for y in range(col_count):
                if self.dynamic_map[x, y] == 3 or self.dynamic_map[x, y] == 4:  # Nếu ô đang chứa chướng ngại vật động hoặc vật thể động tiềm năng
                    self.dynamic_map[x, y] = self.static_map[x, y]  # Cập nhật lại theo bản đồ tĩnh
                if self.predict_map[x, y] == 3 or self.predict_map[x, y] == 4:  # Nếu ô đang chứa chướng ngại vật động hoặc vật thể động tiềm năng
                    self.predict_map[x, y] = self.static_map[x, y]  # Cập nhật lại theo bản đồ tĩnh

        # Bản đồ động
        for obs in dynamic_obs_list:    # Duyệt qua danh sách các chướng ngại vật động
            if loop_count % obs.velocity == 0:  # Chỉ cập nhật mỗi khi đủ số lần lặp
                obs.move_one_step(self.static_map)  # Di chuyển chướng ngại vật động một bước
                
            for dx in range(obs.height):    # Duyệt qua chiều dài của chướng ngại vật
                for dy in range(obs.width): # Duyệt qua chiều rộng của chướng ngại vật
                    x, y = obs.cur_row + dx, obs.cur_col + dy   # Tính toán vị trí của ô chướng ngại vật động
                    if self.current_pos == (x, y):  # Nếu ô đang chứa robot 
                        print("Tổng chiều dài: ", coverage_length)
                        self.calculate_coverage_rataio()
                        print("Tỷ lệ phủ sóng: ", coverage_ratio)
                        print("Tỷ lệ lặp lại: ", repetition_rate)
                        raise Exception('Va chạm với chướng ngại vật')
                    self.dynamic_map[x, y] = 3  # Đặt ô chứa chướng ngại vật động thành 3
        ui.set_map(self.dynamic_map)    # Cập nhật bản đồ động

    def update_probability_map_and_seen_map(self):
        """
        Cập nhật bản đồ xác suất và bản đồ đã thăm.

        Parameters:
            None

        Returns:
            None
        """
        row_count, col_count = len(self.static_map), len(self.static_map[0])    # Số hàng và số cột của bản đồ

        # Đặt lại bản đồ đã thăm
        for x in range(row_count):
            for y in range(col_count):
                self.seen_map[x, y] = self.static_map[x, y] # Đặt lại bản đồ đã thăm

        # Đặt lại bản đồ xác suất
        for x in range(row_count):
            for y in range(col_count):
                if self.static_map[x, y] == 1:  
                    self.prob_map[x, y] = 0 # Nếu ô đang chứa chướng ngại vật, đặt xác suất thành 0
                else:
                    self.prob_map[x, y] = 0 

        detected_obs = self.obs_sensor(vision_range=VISION_SENSOR_RANGE)    # Xác định chướng ngại vật trong tầm nhìn của robot
        obs_potential_next_move = []    # Danh sách các vị trí tiềm năng mà chướng ngại vật có thể di chuyển đến
        obs_occupy_list = []    # Danh sách các ô mà chướng ngại vật đang chiếm giữ
        for obs in detected_obs:    # Duyệt qua danh sách các chướng ngại vật phát hiện được
            self.calculateProbabilityMap(obs)   # Tính toán bản đồ xác suất
            for row in range(row_count):    
                for col in range(col_count):
                    if self.prob_map[row, col] != 0 and self.dynamic_map[row, col] != 1:    # Nếu ô không chứa chướng ngại vật và có xác suất khác 0
                        obs_potential_next_move.append((row, col))  # Thêm vào danh sách các vị trí tiềm năng
            obs_potential_next_move += self.get_potential_positions(obs)    # Thêm nhung vị trí tiềm năng từ bản đồ xác suất
            obs_occupy_list += obs.get_current_occupy_positions()   # Thêm vào danh sách các ô mà chướng ngại vật đang chiếm giữ

        for pos in obs_potential_next_move: # Duyệt qua danh sách các vị trí tiềm năng
            self.dynamic_map[pos] = 4   # Cập nhật bản đồ động mà ô đó có thể là vật thể động tiềm năng

        for pos in obs_occupy_list: # Duyệt qua danh sách các ô mà chướng ngại vật đang chiếm giữ
            self.dynamic_map[pos] = 3   # Cập nhật bản đồ động mà ô đó có chứa chướng ngại vật động
            self.seen_map[pos] = 3  # Cập nhật bản đồ đã thăm mà ô đó có chứa chướng ngại vật động
            self.prob_map[pos] = 100    # Cập nhật bản đồ xác suất mà ô đó có chứa chướng ngại vật động
        ui.set_map(self.dynamic_map)

    def run(self):
        """
        Hàm chạy cho robot.

        Parameters:
            None

        Returns:
            None
        """
        global nums_cell_repetition  # Biến toàn cục để theo dõi số lần lặp lại ô

        global FPS  # Biến toàn cục cho tốc độ khung hình

        clock = pg.time.Clock()  # Đối tượng đồng hồ để kiểm soát tốc độ khung hình
        run = True  # Biến để kiểm tra liệu chương trình còn đang chạy hay không
        pause = False  # Biến để tạm dừng chương trình
        coverage_finish = False  # Biến để kiểm tra liệu việc duyệt đã hoàn thành hay chưa

        loop_count = 0  # Biến để đếm số lần lặp

        while run:
            loop_count += 1
            ui.draw()  # Vẽ giao diện

            clock.tick(FPS)  # Đặt tốc độ khung hình

            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:  # Ấn phím Space để tạm dừng hoặc tiếp tục
                        pause = not pause
                    elif event.key == pg.K_LEFT:  # Ấn phím mũi tên trái để chạy chậm hơn
                        FPS /= 2
                    elif event.key == pg.K_RIGHT:  # Ấn phím mũi tên phải để tăng tốc độ
                        FPS *= 2
                if event.type == pg.QUIT:
                    run = False  # Khi đóng cửa sổ, dừng chương trình

            if pause:
                continue  # Nếu đang tạm dừng, tiếp tục vòng lặp

            if coverage_finish:
                continue  # Nếu việc duyệt đã hoàn thành, không làm gì cho đến khi đóng cửa sổ

            self.task()  # Thực hiện công việc tại vị trí hiện tại
            self.update_dynamic_map(loop_count)  # Cập nhật bản đồ động

            if loop_count % self.velocity != 0:
                continue  # Chỉ cập nhật mỗi khi đủ số lần lặp

            self.update_probability_map_and_seen_map()  # Cập nhật bản đồ xác suất và bản đồ quan sát

            flag = self.detect_dynamic_obs(VISION_SENSOR_RANGE)  # Kiểm tra xem có vật thể động trong tầm nhìn không
            self.logic.set_map(self.seen_map)   # Cập nhật bản đồ đã thăm

            if flag == False:
                wp = self.logic.get_wp(self.current_pos)  # Lấy đường dẫn tối ưu không có vật thể động

            if flag == True:
                self.logic.set_prob_map(self.prob_map)  # Cập nhật bản đồ xác suất
                np.savetxt('array.txt', self.prob_map, fmt='%f', delimiter=' ', newline='\r\n') # Lưu bản đồ xác suất vào file
                max_bid_value, replan_wp = self.logic.get_replan_wp(self.current_pos)   # Lấy đường dẫn tối ưu có vật thể động - đấu giá 2 bước

                wp = [replan_wp]  # Sử dụng đường dẫn đã tính toán lại từ bản đồ xác suất - đấu giá 2 bước
                
                # QUYẾT ĐỊNH ĐI HOẠC CHỜ
                designated_wp = self.logic.get_wp(self.current_pos) # Lấy đường dẫn tối ưu không có vật thể động - boustrophedon
                
                # Nếu ô được chọn theo đấu giá 2 bước khác với ô được chọn theo boustrophedon và xác suất của ô hiện tại nhỏ hơn ngưỡng 
                if wp != designated_wp and self.prob_map[self.current_pos] < MIN_PROB_THRESHOLD and len(designated_wp) > 0:  
                    designated_wp = designated_wp[0]    # Lấy theo boustrophedon
                    if self.prob_map[designated_wp] > 0:    # Nếu xác suất của ô đó lớn hơn 0 thì đợi chướng ngại vật đi qua
                        continue
                    else:
                        wp = [designated_wp]    # Chọn ô tiếp theo theo boustrophedon

            if self.logic.state == Q.NORMAL:
                selected_cell = self.select_from_wp(wp)  # Lựa chọn ô tiếp theo để di chuyển tới
                self.move_to(selected_cell)  # Di chuyển đến ô đã chọn

            elif self.logic.state == Q.DEADLOCK:
                path = self.logic.escape_deadlock_path(self.current_pos)  # Tìm đường thoát khỏi tình huống bế tắc
                if path == []:
                    # Kết thúc việc duyệt
                    # self.move_to(self.battery_pos)
                    # coverage_finish = True
                    # return
                    pg.image.save(ui.WIN, "./out/test2_5.png")
                    continue
                else:
                    if flag == False:
                        self.move_to(path[0])
                    else:
                        _, deadlock_wp = self.logic.escape_deadlock_dynamic(self.current_pos, path[-1])
                        
                        # Quyết định đi hoặc chờ
                        if deadlock_wp != path[0] and self.prob_map[self.current_pos] < MIN_PROB_THRESHOLD:
                            if self.prob_map[path[0]] > 0:
                                continue
                        self.move_to(deadlock_wp)

        return


    def get_border_cells(self, cur_pos):
        """
        Trả về các ô xung quanh trong tầm nhìn ở vị trí hiện tại.

        Parameters:
            - cur_pos: Vị trí hiện tại của robot.

        Returns:
            - border_cells: Danh sách các ô xung quanh trong tầm nhìn.
        """
        left_border = right_border = up_border = down_border = -1   # Khởi tạo
        border_cells = []
        cur_x, cur_y = cur_pos[0], cur_pos[1]
        for x in range(cur_x - VISION_SENSOR_RANGE, cur_x + 1):
            if x >= 0:
                up_border = x
                break

        for x in range(cur_x, cur_x + VISION_SENSOR_RANGE + 1):
            if x >= ROW_COUNT:
                break
            else:
                down_border = x

        for y in range(cur_y - VISION_SENSOR_RANGE, cur_y + 1):
            if y >= 0:
                left_border = y
                break

        for y in range(cur_y, cur_y + VISION_SENSOR_RANGE + 1):
            if y >= COL_COUNT:
                break
            else:
                right_border = y

        for x in range(up_border, down_border + 1):
            for y in range(left_border, right_border + 1):
                if x == up_border or x == down_border:
                    border_cells.append((x, y))
                else:
                    if y == left_border or y == right_border:
                        border_cells.append((x, y))
        return border_cells


    def obstruct_cell_list(self, pos_from, pos_to, strict=False):
        """
        Trả về danh sách các ô trên đường từ pos_from đến pos_to bị chướng ngại vật.

        Parameters:
            - pos_from: Vị trí xuất phát.
            - pos_to: Vị trí đích.
            - strict: Xác định cách xử lý ô biên.

        Returns:
            - cell_list: Danh sách các ô bị chặn.
        """
        threshold = 0.3 # Ngưỡng giá trị: [0, 0.5]
        start = (pos_from[0] + 0.5, pos_from[1] + 0.5)  # Đưa về tâm ô
        goal = (pos_to[0] + 0.5, pos_to[1] + 0.5)   # Đưa về tâm ô

        vecto = (goal[0] - start[0], goal[1] - start[1])    # Tính toán vecto
        angle = - np.arctan2(vecto[0], vecto[1])    # Tính toán góc Hàm arctan2 trả về góc theo radian nằm trong khoảng [-pi, pi] 
            # Thêm dấu trừ (-) trước, nó đảm bảo rằng góc quay sẽ theo chiều kim đồng hồ.
        (x, y) = pos_from   # Lấy vị trí xuất phát
        cell_list = [pos_from]  # Khởi tạo danh sách các ô bị chặn

        sx, sy = sign(vecto[0]), sign(vecto[1]) # Tính toán hướng di chuyển
        dx = abs(0.5 / math.sin(angle)) if vecto[0] != 0 else math.inf  # Khoảng cách vật cản di chuyển theo trục x trong một bước
        dy = abs(0.5 / math.cos(angle)) if vecto[1] != 0 else math.inf  # Khoảng cách vật cản di chuyển theo trục y trong một bước
        sum_x, sum_y = dx, dy

        while (x, y) != pos_to: # Duyệt qua các ô cho đến khi đến ô đích sử dụng thuật toán Bresenham
            # Nếu sum_x == sum_y, tăng cả x và y
            (movx, movy) = (sum_x < sum_y or math.isclose(sum_x, sum_y), sum_y < sum_x or math.isclose(sum_x, sum_y)) # bugfix: sin, cos không đưa ra kết quả chính xác
                # hàm math.isclose(sum_x, sum_y) trả về True nếu sum_x và sum_y gần nhau
            prev_x, prev_y = x, y
            prev_sum_x, prev_sum_y = sum_x, sum_y
            if movx:
                x += sx
                sum_x += 2 * dx

            if movy:
                y += sy
                sum_y += 2 * dy

            if strict:
                if movx and movy: 
                    cell_list.extend([(prev_x, prev_y + sy), (prev_x + sx, prev_y)])
                elif movx and not movy: 
                    projection_y = (abs(prev_sum_x * math.cos(angle)) - 0.5) % 1
                    if projection_y < threshold:
                        cell_list.append((x, prev_y - sy))
                    elif projection_y > 1 - threshold:
                        cell_list.append((prev_x, prev_y + sy))
                elif movy and not movx:
                    projection_x = (abs(prev_sum_y * math.sin(angle)) - 0.5) % 1
                    if projection_x < threshold:
                        cell_list.append((prev_x - sx, y))
                    elif projection_x > 1 - threshold:
                        cell_list.append((prev_x + sx, prev_y))

            cell_list.append((x, y))

        return cell_list

    def select_from_wp(self, wp):
        """
        Chọn một điểm từ danh sách điểm đích có thể.

        Parameters:
            - wp: Danh sách các điểm đích có thể.

        Returns:
            - selected_cell: Điểm được chọn.
        """
        return min(wp, key=self.travel_cost)

    def obs_sensor(self, vision_range=VISION_SENSOR_RANGE):
        """
        Xác định chướng ngại vật trong tầm nhìn của robot.

        Parameters:
            - vision_range: Tầm nhìn của robot.

        Returns:
            - obs_detected_list: Danh sách các chướng ngại vật được phát hiện.
        """
        obs_detected_list = []  # Danh sách các chướng ngại vật được phát hiện

        in_sensor_list = [] # Danh sách các ô nằm trong tầm nhìn của robot
        border_cells = self.get_border_cells(self.current_pos)  # Lấy danh sách các ô biên
    
        for pos in border_cells:    # Duyệt qua danh sách các ô biên
            obstruct_cell_list = self.obstruct_cell_list(self.current_pos, pos)   # Lấy danh sách các ô bị chặn
            for cell in obstruct_cell_list: # Duyệt qua danh sách các ô bị chặn
                if self.dynamic_map[cell] == 1: 
                    break
                if cell not in in_sensor_list:
                    in_sensor_list.append(cell)

        for obs in dynamic_obs_list:    # Duyệt qua danh sách các chướng ngại vật động
            if set(obs.get_current_occupy_positions()) & set(in_sensor_list):   # Nếu có ô nằm trong tầm nhìn của robot
                obs_detected_list.append(obs)   # Thêm vào danh sách các chướng ngại vật được phát hiện
        
        self.obs_prev_detected_dict = self.obs_detected_dict.copy()  # Cập nhật lại danh sách các chướng ngại vật được phát hiện trước đó
        self.obs_detected_dict = {obs: obs.get_pos() for obs in obs_detected_list}  # Cập nhật lại danh sách các chướng ngại vật được phát hiện hiện tại    
        
        return obs_detected_list

    def get_neighbours(self, cur_pos, size):
        """
        Trả về các ô hàng xóm xung quanh vị trí hiện tại.

        Parameters:
            - cur_pos: Vị trí hiện tại của robot.
            - size: Khoảng cách tối đa đến hàng xóm.

        Returns:
            - neighbours: Danh sách các ô hàng xóm.
        """
        cur_x, cur_y = cur_pos
        neighbours = []
        for x in range(cur_x - size, cur_x + size + 1):
            for y in range(cur_y - size, cur_y + size + 1):
                if check_valid_pos((x, y)) == False: continue
                if self.dynamic_map[x, y] == 1 or self.dynamic_map[x, y] == 3:
                    continue
                if (x, y) == cur_pos:
                    continue
                neighbours.append((x, y))
        return neighbours

    def get_potential_positions(self, obs: DynamicObstacle):
        """
        Trả về danh sách các vị trí tiềm năng mà chướng ngại vật có thể di chuyển đến.

        Parameters:
            - obs: Chướng ngại vật động.

        Returns:
            - prob_neighbour_list: Danh sách các vị trí tiềm năng.
        """
        neighbour = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]

        obs_occupy_list = obs.get_current_occupy_positions()    # Lấy danh sách các ô mà chướng ngại vật đang chiếm giữ

        prob_neighbour_list = []    # Danh sách các vị trí tiềm năng
        visited = []    # Danh sách các ô đã được duyệt
        queue = deque() # Hàng đợi
        queue.extend([(i, 0) for i in obs_occupy_list]) # Thêm các ô mà chướng ngại vật đang chiếm giữ vào hàng đợi
        while queue:    # Duyệt qua hàng đợi
            current_pos, step = queue.popleft() # Lấy phần tử đầu tiên
            for dx, dy in neighbour:    # Duyệt qua các ô hàng xóm
                x, y = current_pos[0] + dx, current_pos[1] + dy # Tính toán vị trí hàng xóm
                if not check_valid_pos((x, y)): # Nếu vị trí không hợp lệ
                    continue
                if (x, y) in visited:   # Nếu ô đã được duyệt
                    continue
                if self.dynamic_map[x, y] == 1 or self.dynamic_map[x, y] == 3:  # Nếu ô đang chứa chướng ngại vật
                    continue
                if step > self.scan_freq * obs.velocity:    # Nếu số bước lớn hơn số bước tối đa
                    continue
                queue.append(((x, y), step + 1))    # Thêm vào hàng đợi
                visited.append((x, y))  # Thêm vào danh sách các ô đã được duyệt
                if self.prob_map[x, y] > 0 and self.dynamic_map[x, y] != 1: # Nếu ô không chứa chướng ngại vật và có xác suất khác 0
                    prob_neighbour_list.append((x, y))  # Thêm vào danh sách các vị trí tiềm năng
        return prob_neighbour_list  # Trả về danh sách các vị trí tiềm năng

    def calculate_obs_velocity(self, prev_pos, cur_pos):
        """
        Tính toán tốc độ của chướng ngại vật dựa trên vị trí trước và sau.

        Parameters:
            - prev_pos: Vị trí trước đó của chướng ngại vật.
            - cur_pos: Vị trí hiện tại của chướng ngại vật.

        Returns:
            - c: Tốc độ của chướng ngại vật.
        """
        c = (math.dist(prev_pos, cur_pos)) / (self.scan_freq)   # Tính toán tốc độ của chướng ngại vật
            # vận tốc = khoảng cách / thời gian (tần số quét)
        return c

    def predict_obs_velocity(self, obs: DynamicObstacle):
        """
        Dự đoán tốc độ của chướng ngại vật.

        Parameters:
            - obs: Chướng ngại vật động.

        Returns:
            - obs.velocity: Tốc độ dự đoán của chướng ngại vật.
        """
        # TODO: Implement
        return obs.velocity

    def calculate_obs_direction(self, prev_pos, cur_pos):
        """
        Tính toán hướng di chuyển của chướng ngại vật dựa trên vị trí trước và sau.

        Parameters:
            - prev_pos: Vị trí trước đó của chướng ngại vật.
            - cur_pos: Vị trí hiện tại của chướng ngại vật.

        Returns:
            - direction: Hướng di chuyển của chướng ngại vật.
        """
        dx, dy = cur_pos[0] - prev_pos[0], cur_pos[1] - prev_pos[1]
        if dx < 0: return 1
        if dx > 0: return 4
        if dy > 0: return 3
        return 2

    def sample(self, z):
        """
        Tạo mẫu ngẫu nhiên từ phân phối đều trong khoảng [-z, z].

        Parameters:
            - z: Khoảng giá trị để tạo mẫu.

        Returns:
            - np.sum(rand) * 1/12: Giá trị mẫu ngẫu nhiên.
        """
        rand = np.random.uniform(-z, z, 12)
        return np.sum(rand) * 1/12

    def sampling(self, obs: DynamicObstacle):
        """
        Tạo mẫu cho vị trí tiềm năng của chướng ngại vật dựa trên phương pháp lấy mẫu.

        Parameters:
            - obs: Chướng ngại vật động.

        Returns:
            - (round(x_prime), round(y_prime)): Vị trí tiềm năng mới của chướng ngại vật.
        """
        (x, y) = obs.get_pos()
        v_prime = obs.v + self.sample(self.alpha_1 * abs(obs.velocity) + self.alpha_2 * abs(obs.w))
        w_prime = obs.w + self.sample(self.alpha_3 * abs(obs.velocity) + self.alpha_4 * abs(obs.w))
        x_prime = x - v_prime/w_prime * math.sin(obs.theta) + v_prime/w_prime * math.sin(obs.theta + self.scan_freq * w_prime)
        y_prime = y + v_prime/w_prime * math.cos(obs.theta) - v_prime/w_prime * math.cos(obs.theta + self.scan_freq * w_prime)
        return (round(x_prime), round(y_prime))

    def calculateProbabilityMap(self, obs: DynamicObstacle):
        """
        Tính toán bản đồ xác suất dựa trên mẫu ngẫu nhiên của vị trí tiềm năng của chướng ngại vật.

        Parameters:
            - obs: Chướng ngại vật động.

        Returns:
            - None
        """
        new_pos_dict = dict()
        for _ in range(NUMS_SAMPLE):
            new_pos = self.sampling(obs)
            if new_pos not in new_pos_dict.keys():
                new_pos_dict[new_pos] = 1
            else:
                new_pos_dict[new_pos] += 1

        for new_pos in new_pos_dict.keys():
            prob = round(new_pos_dict[new_pos] / NUMS_SAMPLE * 100, 1)
            if not check_valid_pos(new_pos):
                continue
            if prob < self.prob_map[new_pos]:
                continue
            self.prob_map[new_pos] = prob

    def detect_dynamic_obs(self, vision_range=VISION_SENSOR_RANGE):
        """
        Xác định sự xuất hiện của chướng ngại vật động trong tầm nhìn của robot.

        Parameters:
            - vision_range: Tầm nhìn của robot.

        Returns:
            - True nếu có chướng ngại vật động, False nếu không có.
        """
        cur_x, cur_y = self.current_pos

        in_sensor_list = []
        border_cells = self.get_border_cells(self.current_pos)
        for pos in border_cells:
            obstruct_cell_list = self.obstruct_cell_list(self.current_pos, pos)
            for cell in obstruct_cell_list:
                if self.dynamic_map[cell] == 1:
                    break
                if cell not in in_sensor_list:
                    in_sensor_list.append(cell)
        
        for cell in in_sensor_list:
            if self.dynamic_map[cell] == 3:
                return True

        return False

    def task(self):
        """
        Thực hiện nhiệm vụ của robot tại vị trí hiện tại.

        Returns:
            - None
        """
        current_pos = self.current_pos
        self.static_map[current_pos] = 2
        self.logic.update_explored(current_pos)
        ui.task(current_pos)
    
    def move_to(self, pos):
        """
        Di chuyển robot đến vị trí mới.

        Parameters:
            - pos: Vị trí đích.

        Raises:
            - Exception nếu có xung đột với chướng ngại vật.

        Returns:
            - None
        """
        if self.dynamic_map[pos] == 3:
            raise Exception('Va chạm với chướng ngại vật')
        if self.static_map[pos] == 2:
            global nums_cell_repetition
            # Đếm số lần lặp lại ô
            nums_cell_repetition += 1   

        global total_travel_length, coverage_length, retreat_length, advance_length
        dist = energy = math.dist(self.current_pos, pos)

        if self.move_status in (1, 3): # retreat or advance cost half energy as coverage
            energy = 0.5 * energy

        if self.energy < energy:
            raise Exception('Robot hết pin')
        self.energy -= energy

        self.rotate_to(pos)
        self.current_pos = pos

        if self.move_status == 0:
            ui.move_to(pos)
            coverage_length += dist
            # if self.static_map[pos] == 2:
            #     global nums_cell_repetition
            #     # Đếm số lần lặp lại ô
            #     nums_cell_repetition += 1   
        
        elif self.move_status == 1:
            ui.move_retreat(pos)
            retreat_length += dist
        elif self.move_status == 3:
            ui.move_advance(pos)
            advance_length += dist
        
        total_travel_length += dist

        ui.set_energy_display(self.energy)

    def travel_cost(self, pos_to):
        """
        Tính toán chi phí di chuyển từ vị trí hiện tại đến vị trí đích.

        Parameters:
            - pos_to: Vị trí đích.

        Returns:
            - cost: Chi phí di chuyển.
        """
        pos_from = self.current_pos
        turn_angle = abs(self.angle - self.get_angle(pos_to))
        if turn_angle > math.pi: # luôn chọn góc quay nhỏ hơn
            turn_angle = 2 * math.pi - turn_angle 
        travel_dist = math.dist(pos_from, pos_to)

        # Chi phí của quãng đường di chuyển, góc quay (ước tính)
        cost = 2 * travel_dist + 1 * turn_angle
        return cost

    def get_angle(self, pos_to):
        """
        Tính toán góc quay cần thiết để quay đến vị trí đích.

        Parameters:
            - pos_to: Vị trí đích.

        Returns:
            - angle: Góc quay.
        """
        pos_from = self.current_pos
        vecto = (pos_to[0] - pos_from[0], pos_to[1] - pos_from[1])
        angle = - np.arctan2(vecto[0], vecto[1])
        return angle % (2 * math.pi)
    
    def rotate_to(self, pos_to):
        """
        Quay robot đến hướng vị trí đích.

        Parameters:
            - pos_to: Vị trí đích.

        Returns:
            - None
        """
        self.angle = self.get_angle(pos_to)

    def follow_path_plan(self, path, time_delay=0, check_energy=False, stop_on_unexpored=False):
        """
        Theo dõi kế hoạch di chuyển theo đường dẫn.

        Parameters:
            - path: Đường dẫn để di chuyển.
            - time_delay: Độ trễ giữa các bước di chuyển.
            - check_energy: Kiểm tra năng lượng trước khi di chuyển.
            - stop_on_unexpored: Dừng khi gặp ô chưa khám phá.

        Returns:
            - None
        """
        first_loop = True
        clock = pg.time.Clock()
        for pos in path:
            clock.tick(FPS)

            while True:
                if first_loop: first_loop = False
                else: self.update_dynamic_map()
                
                dynamic_planning_flag = self.decision_making()
                if dynamic_planning_flag == 1: break # dynamic planning: wait

            self.move_to(pos)
            ui.draw()
            time.sleep(time_delay)

            if stop_on_unexpored:
                if self.logic.weight_map[pos] > 0: return

    def calculate_coverage_rataio(self):
        """
        Tính toán tỷ lệ phủ sóng của môi trường đã được khám phá.
        total_coverable_cells: Tổng số ô có thể khám phá.
        nums_covered_cells: Số ô đã được khám phá.
        nums_cell_repetition: Số ô lặp lại.
        coverage_ratio: Tỉ lệ bao phủ.
        repetition_rate: Tỉ lệ lặp lại.
        Returns:
            - None
        """
        
        global total_coverable_cells, nums_covered_cells
        global coverage_ratio, repetition_rate
        for rows in self.static_map:
            for i in rows:
                if i == 0:
                    total_coverable_cells += 1
                elif i == 2:
                    total_coverable_cells += 1
                    nums_covered_cells += 1
        coverage_ratio = round(nums_covered_cells/total_coverable_cells*100, 2)
        repetition_rate = round(nums_cell_repetition/nums_covered_cells*100, 2)

def main():
    """
    Hàm chính thực hiện chạy robot và tính toán thông số kết quả.

    Returns:
        - None
    """
    robot = Robot(battery_pos, ROW_COUNT, COL_COUNT)
    robot.init_static_map(ENVIRONMENT)
    robot.run()
    robot.calculate_coverage_rataio()

    
    print('\nTổng số ô trống trên bản đồ:\t', total_coverable_cells)
    # print('Retreat:\t\t', retreat_length)
    # print('Advance:\t\t', advance_length)
    print('Số ô đã khám phá:\t', nums_covered_cells)
    print('Tỷ lệ phủ sóng:\t\t', coverage_ratio,'%')
    print('Số ô lặp lại:\t\t', nums_cell_repetition)
    print('Tỷ lệ lặp lại:\t\t', repetition_rate,'%')
    print('-' * 8)
    print('Total:', total_travel_length)

    print('\nNumber Of Return: ', return_charge_count)

if __name__ == "__main__":
    main()

