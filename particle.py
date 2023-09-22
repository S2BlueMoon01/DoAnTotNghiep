import numpy as np


class Particle:
    def __init__(self, num_cities):
        """
        Khởi tạo một particle với vị trí và vận tốc ngẫu nhiên.

        Parameters:
        num_cities (int): Số lượng thành phố trong bài toán TSP.

        Returns:
        None
        """
        self.position = np.random.permutation(num_cities)
        self.velocity = np.random.permutation(num_cities)
        self.best_position = self.position
        self.best_distance = float("inf")
