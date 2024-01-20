import numpy as np
import pygame as pg
import copy
import colorsys
import random

# Một số màu sắc được định nghĩa
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (197, 198, 208)
DARKGREY = (125, 125, 135)
RED = (255, 0, 0)
DARK_YELLOW = (255,253,175)
GREEN = (0,255,0)
BLUE = (5, 16, 148)
BROWN = (76, 1, 33)
ORANGE = (255, 165, 0)

# Kích thước ô lưới
EPSILON = 15

BORDER = 1

# INFO_BAR_HEIGHT = 30

# Pham vi cảm biến của robot
VISION_SENSOR_RANGE = 5

# Khởi tạo pygame
pg.init()
font = pg.font.SysFont(None, 30)

def hsv2rgb(h, s, v): 
    """
    Chuyển đổi từ không gian màu HSV sang RGB.

    Parameters:
        - h (float): Giá trị màu hue.
        - s (float): Giá trị độ bão hòa.
        - v (float): Giá trị độ sáng.

    Returns:
        - tuple: Giá trị màu RGB tương ứng.
    """
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
    return (int(255*r), int(255*g), int(255*b)) 

def getDistinctColors(n): 
    """
    Tạo danh sách màu sắc phân biệt.

    Parameters:
        - n (int): Số lượng màu sắc cần tạo.

    Returns:
        - list: Danh sách các màu sắc phân biệt.
    """
    huePartition = 1.0 / (n + 1) 
    return [hsv2rgb(huePartition * value, 1.0, 1.0) for value in range(0, n)]

class Grid_Map:
    def __init__(self, vision_range = VISION_SENSOR_RANGE):
        # đặt tiêu đề
        pg.display.set_caption("Coverage")
        
        #lưu trữ cửa sổ hiển thị của ứng dụng
        self.WIN = None

        # Lưu trữ thông tin về môi trường
        self.map = None
        
        # Số hàng trong map
        self.row_count = 0

        # Số cột trong map
        self.col_count = 0

        # Vị trí của pin và robot
        self.battery_pos = (0, 0)
        self.vehicle_pos = (0, 0)

        # Hình dạng của pin và robot
        self.battery_img = pg.Rect(BORDER, BORDER, EPSILON - BORDER, EPSILON - BORDER)
        self.vehicle_img = pg.Rect(BORDER, BORDER, EPSILON - BORDER, EPSILON - BORDER)

        
        self.trajectories = [[(0, 0)]] # Danh sách các đường đi (đang hiển thị đường đi của việc phủ sóng)

        self.move_status = 0 # 0: di chuyển bình thường, 1: rút lui, 2: sạc, 3: tiến lên
        self.charge_path_plan = [] # dùng cho rút lui và tiến lên

        self.energy_display = None  # thông tin hiển thị về năng lượng
        
        self.vision_range = vision_range # phạm vi cảm biến của robot

    def read_map(self, filepath):
        """
        Đọc bản đồ từ tệp và cài đặt các thông số liên quan.

        Parameters:
            - filepath (str): Đường dẫn đến tệp chứa bản đồ.

        Returns:
            - tuple: Bản sao của bản đồ và vị trí ban đầu của pin.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self.col_count, self.row_count = [int(i) for i in f.readline().strip().split()]

            display_size = [EPSILON * self.col_count, EPSILON * self.row_count]
            self.WIN = pg.display.set_mode(display_size)

            map = []
            for idx, line in enumerate(f):
                line =[int(value) for value in line.strip().split()]
                map.append(line)
            
            if len(map) == 0:
                map = np.zeros((self.row_count, self.col_count), dtype=object)
                
            self.map = np.array(map, dtype=object)
        
        return copy.deepcopy(map), self.battery_pos

    def edit_map(self):
        """
        Cho phép người dùng chỉnh sửa bản đồ bằng cách vẽ lên màn hình.

        Returns:
            - tuple: Bản sao của bản đồ và vị trí ban đầu của pin sau khi chỉnh sửa.
        """
        done = False
        draw_obstacle = False 
        prev_cell = None

        while done == False:
            pos = pg.mouse.get_pos()
            col = pos[0] // EPSILON
            row = pos[1] // EPSILON

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    done = True
                elif event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pressed = pg.mouse.get_pressed()
                    if mouse_pressed[0]: # Nhấn chuột trái: vẽ chướng ngại vật
                        draw_obstacle = True
                    if mouse_pressed[2]: # Nhấn chuột phải: thiết lập vị trí ban đầu của pin
                        if self.check_valid_pos((row, col)) == False: continue
                        self.update_battery_pos((row, col))
                        self.trajectories[0] = [(row, col)]
                        self.map[row][col] = 0
                        
                elif event.type == pg.MOUSEBUTTONUP:
                    draw_obstacle = False
                    prev_cell = None
            
            # Kiểm tra cờ boolean để cho phép giữ chuột trái để vẽ
            if draw_obstacle:
                if self.check_valid_pos((row, col)) == False: continue
                if (prev_cell != (row, col)):
                    prev_cell = (row, col)
                    if self.map[row][col] == 0 and self.battery_pos != (row, col):
                        self.map[row][col] = 1
                    else:
                        self.map[row][col] = 0

            # Vẽ bản đồ lên màn hình
            self.draw_map()
            pg.draw.rect(self.WIN, GREEN, self.battery_img)
            pg.display.flip()

        pg.image.save(self.WIN, "./out/map/map1.png")
        return copy.deepcopy(self.map), self.battery_pos

    def save_map(self, output_file):
        """
        Lưu bản đồ hiện tại vào một tệp.

        Parameters:
            - output_file (str): Đường dẫn đến tệp đầu ra.

        Returns:
            None
        """
        map = self.map
        
        with open(output_file, "w", encoding="utf-8") as f:
            col_count, row_count = len(map[0]), len(map)
            f.write(str(col_count) + ' ' + str(row_count) + '\n')
            for row in map:
                line = [str(value) for value in row]
                line = " ".join(line)
                f.write(line +'\n')
            print("Save map done!")
    
    def set_map(self, map):
        self.map = map
    
    def draw(self):
        """
        Vẽ bản đồ lên màn hình.

        Returns:
            None
        """
        self.draw_map()
    
        for coverage_path in self.trajectories:
            self.draw_path(coverage_path)
        
        if self.move_status == 1: # rút lui
            self.draw_path(self.charge_path_plan, BROWN, 4)
        elif self.move_status == 3: # tiến lên
            self.draw_path(self.charge_path_plan, BLUE, 4)

        pg.draw.rect(self.WIN, GREEN, self.battery_img)
        pg.draw.rect(self.WIN, RED, self.vehicle_img)
        
        sensor_centor = ((self.vehicle_pos[1] + 1/2) * EPSILON + BORDER, (self.vehicle_pos[0] + 1/2) * EPSILON + BORDER)
        # sensor_radius = (VISION_SENSOR_RANGE + 1/2) * EPSILON
        sensor_radius = (self.vision_range + 1/2) * EPSILON
        pg.draw.circle(self.WIN, (204, 255, 255), sensor_centor, sensor_radius, width = 4)

        energy_display_img = font.render('Energy: ' + str(self.energy_display), True, RED)
        pg.display.flip()

    def draw_map(self):
        """
        Vẽ các ô trên bản đồ.

        Returns:
            None
        """
        self.WIN.fill(BLACK)
        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                color = WHITE
                if self.map[row][col] in (1, 'o'):
                    color = BLACK

                # Màu đường đã đi
                elif self.map[row][col] in (2, 'e'):
                    color = DARK_YELLOW
 
                # Màu chướng ngại vật
                elif self.map[row][col] == 3:
                    color = DARKGREY

                # Màu những ô mà chướng ngại vật có thể xuất hiện
                elif self.map[row][col] == 4:
                    color = ORANGE
                
                
                pg.draw.rect(self.WIN,
                            color,
                            [EPSILON * col + BORDER,
                            EPSILON * row + BORDER,
                            EPSILON - BORDER,
                            EPSILON - BORDER])

    # def illustrate_regions(self, decomposed, region_count):
    #     """
    #     Hiển thị các khu vực được phân chia trên bản đồ.

    #     Parameters:
    #         - decomposed (list): Ma trận chứa thông tin về việc phân chia khu vực.
    #         - region_count (int): Số lượng khu vực.

    #     Returns:
    #         None
    #     """
    #     self.WIN.fill(BLACK)
    #     region_colors = getDistinctColors(region_count)
    #     random.shuffle(region_colors)

    #     for row in range(len(decomposed)):
    #         for col in range(len(decomposed[0])):
    #             color = BLACK
    #             #region = decomposed[row][col]
    #             if region != 0:
    #                 color = region_colors[region - 1]

    #             pg.draw.rect(self.WIN,
    #                         color,
    #                         [EPSILON * col + BORDER,
    #                         EPSILON * row + BORDER,
    #                         EPSILON - BORDER,
    #                         EPSILON - BORDER])

    #     pg.display.flip()

    def draw_path(self, path, color=RED, width=2):
        """
        Vẽ đường đi lên màn hình.

        Parameters:
            - path (list): Danh sách các vị trí trên đường đi.
            - color (tuple): Màu sắc của đường đi.
            - width (int): Độ dày của đường đi.

        Returns:
            None
        """
        point_list = [(EPSILON * pos[1] + EPSILON / 2, EPSILON * pos[0] + EPSILON / 2) for pos in path]
        if len(point_list) > 1:
            pg.draw.lines(self.WIN, color, False, point_list, width=2)

    def update_battery_pos(self, pos):
        """
        Cập nhật vị trí của pin và hình ảnh trên màn hình.

        Parameters:
            - pos (tuple): Vị trí mới của pin (x, y).

        Returns:
            None
        """
        self.battery_pos = pos
        self.battery_img.x = EPSILON * pos[1] + BORDER
        self.battery_img.y = EPSILON * pos[0] + BORDER
    
    def update_vehicle_pos(self, pos):
        """
        Cập nhật vị trí của robot và hình ảnh trên màn hình.

        Parameters:
            - pos (tuple): Vị trí mới của robot (x, y).

        Returns:
            None
        """
        self.vehicle_pos = pos
        self.vehicle_img.x = EPSILON * pos[1] + BORDER
        self.vehicle_img.y = EPSILON * pos[0] + BORDER

    def task(self, pos):
        """
        Đánh dấu một ô đã được thăm.

        Parameters:
            - pos (tuple): Vị trí ô đã được thăm (x, y).

        Returns:
            None
        """
        self.map[pos] = 2

    def move_to(self, pos):
        """
        Di chuyển robot đến vị trí mới.

        Parameters:
            - pos (tuple): Vị trí mới của robot (x, y).

        Returns:
            None
        """
        if self.move_status != 0:
            self.trajectories[-1].append(self.vehicle_pos)
        self.move_status = 0
        self.update_vehicle_pos(pos)
        self.trajectories[-1].append(pos)
    
    def move_retreat(self, pos):
        """
        Di chuyển robot vào trạng thái rút lui.

        Parameters:
            - pos (tuple): Vị trí mới của robot (x, y).

        Returns:
            None
        """
        self.move_status = 1
        self.update_vehicle_pos(pos)
        if pos == self.battery_pos:
            self.move_status = 2

    def move_advance(self, pos):
        """
        Di chuyển robot vào trạng thái tiến lên.

        Parameters:
            - pos (tuple): Vị trí mới của robot (x, y).

        Returns:
            None
        """
        if self.move_status != 3:
            self.trajectories.append([])
        self.move_status = 3
        self.update_vehicle_pos(pos)
    
    def set_charge_path(self, path):
        """
        Thiết lập đường đi cho việc sạc pin.

        Parameters:
            - path (list): Đường đi để sạc pin.

        Returns:
            None
        """
        self.charge_path_plan = path
    
    def set_energy_display(self, energy):
        """
        Cập nhật thông tin hiển thị về năng lượng.

        Parameters:
            - energy (float): Giá trị năng lượng.

        Returns:
            None
        """
        self.energy_display = round(energy, 2)

    def check_valid_pos(self, pos):
        """
        Kiểm tra xem một vị trí có hợp lệ trên bản đồ không.

        Parameters:
            - pos (tuple): Vị trí cần kiểm tra (x, y).

        Returns:
            - bool: True nếu vị trí hợp lệ, False nếu không hợp lệ.
        """
        row, col = pos
        return 0 <= row < self.row_count and 0 <= col < self.col_count

def main():
    ui = Grid_Map()
    ui.read_map('./map/scenario_1/test_1.txt')
    ui.edit_map()
    pg.quit()

if __name__ == "__main__":
    main()
