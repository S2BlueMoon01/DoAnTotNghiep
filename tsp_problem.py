import numpy as np
from euclidean_distance import euclidean_distance


class TSPProblem:
    def __init__(self, cities):
        """
        Khởi tạo bài toán TSP với danh sách các thành phố.

        Parameters:
        cities (list): Danh sách các thành phố.

        Returns:
        None
        """
        self.num_cities = len(cities)
        self.cities = np.array(cities)
        self.distance_matrix = np.zeros((self.num_cities, self.num_cities))

        # Tính và lưu trữ ma trận khoảng cách
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                distance = euclidean_distance(cities[i], cities[j])
                self.distance_matrix[i, j] = distance
                # Khoảng cách giữa thành phố i và thành phố j giống khoảng cách giữa thành phố j và thành phố i
                self.distance_matrix[j, i] = distance

    def calculate_distance(self, tour):
        """
        Tính khoảng cách của một chuỗi thành phố.

        Parameters:
        tour (list): Chuỗi các thành phố.

        Returns:
        float: Khoảng cách của chuỗi thành phố.
        """
        total_distance = 0
        for i in range(len(tour) - 1):
            total_distance += self.distance_matrix[tour[i], tour[i + 1]]
        # Quay lại thành phố xuất phát
        total_distance += self.distance_matrix[tour[-1], tour[0]]
        return total_distance
