import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(point1, point2):
    """
    Tính khoảng cách Euclidean giữa hai điểm.

    Parameters:
    point1 (list): Tọa độ của điểm thứ nhất.
    point2 (list): Tọa độ của điểm thứ hai.

    Returns:
    float: Khoảng cách Euclidean giữa hai điểm.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


# Định nghĩa bài toán TSP
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


# Định nghĩa lớp Particle cho PSO
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


# PSO cho bài toán TSP
def pso_tsp(num_particles, num_iterations, tsp_problem, w, c1, c2):
    """
    Áp dụng thuật toán PSO để giải quyết bài toán TSP.

    Parameters:
    num_particles (int): Số lượng particle trong thuật toán PSO.
    num_iterations (int): Số lần lặp của thuật toán PSO.
    tsp_problem (TSPProblem): Đối tượng bài toán TSP.
    w (float): Hệ số trọng số của vận tốc hiện tại.
    c1 (float): Hệ số trọng số của vận tốc cá nhân.
    c2 (float): Hệ số trọng số của vận tốc toàn cục.

    Returns:
    tuple: Một tuple gồm hai phần tử: chuỗi thành phố tốt nhất và độ dài quãng đường tương ứng.
    """
    num_cities = tsp_problem.num_cities
    particles = [Particle(num_cities) for _ in range(num_particles)]
    global_best_tour = None
    global_best_distance = float("inf")

    for iteration in range(num_iterations):
        for particle in particles:
            # Đánh giá độ lợi (fitness) của particle
            tour_distance = tsp_problem.calculate_distance(particle.position)

            # Cập nhật best_position và best_distance của particle
            if tour_distance < particle.best_distance:
                particle.best_position = particle.position
                particle.best_distance = tour_distance

            # Cập nhật global_best_tour và global_best_distance nếu cần
            if tour_distance < global_best_distance:
                global_best_tour = particle.position
                global_best_distance = tour_distance

        for particle in particles:
            # Cập nhật vận tốc và vị trí của particle
            r1, r2 = np.random.rand(), np.random.rand()
            particle.velocity = (
                w * particle.velocity
                + c1 * r1 * (particle.best_position - particle.position)
                + c2 * r2 * (global_best_tour - particle.position)
            )
            particle.position = np.roll(
                particle.position, int(particle.velocity[0])
            )  # Cuộn vòng lại nếu vượt quá số thành phố
            # print("------------------------------")
            # print(f"Độ dài quãng đường ngắn nhất: {global_best_distance}")
            # print(f"Chuỗi thành phố tốt nhất: {global_best_tour}")
    return global_best_tour, global_best_distance


# Sử dụng bài toán TSP đã định nghĩa
custom_cities = [(0, 0), (3, 0), (3, 4), (6, 4), (6, 0)]  # Tọa độ các thành phố
num_cities = 5

# Tạo ngẫu nhiên các tọa độ của các thành phố
# custom_cities = np.random.rand(num_cities, 2)
tsp_problem = TSPProblem(custom_cities)

# Thiết lập các tham số cho PSO
num_particles = 30  # Số lượng hạt trong PSO
num_iterations = 100  # Số lượng vòng lặp PSO
w = 0.5  # Trọng số vận tốc
c1 = 1.5  # Hệ số học tập cho best_position của particle
c2 = 1.5  # Hệ số học tập cho global_best_tour

best_tour, best_distance = pso_tsp(
    num_particles, num_iterations, tsp_problem, w, c1, c2
)

print(tsp_problem.distance_matrix)

# Vẽ đồ thị
plt.figure(figsize=(8, 6))
cities = tsp_problem.cities
plt.scatter(cities[:, 0], cities[:, 1], c="b", marker="o", label="Thành phố")

# Vẽ đường đi ngắn nhất (đường màu đỏ)
best_tour_cities = cities[best_tour]
best_tour_cities = np.append(
    best_tour_cities, [best_tour_cities[0]], axis=0
)  # Để nối lại thành phố xuất phát
plt.plot(
    best_tour_cities[:, 0],
    best_tour_cities[:, 1],
    c="r",
    linestyle="-",
    linewidth=2,
    label="Đường đi ngắn nhất",
)

# Vẽ các đoạn nối giữa các thành phố (khoảng cách) bằng màu xám
for i in range(len(cities)):
    for j in range(i + 1, len(cities)):
        city1 = cities[i]
        city2 = cities[j]
        distance = np.linalg.norm(city1 - city2)
        plt.plot(
            [city1[0], city2[0]],
            [city1[1], city2[1]],
            c="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )

print(f"Độ dài quãng đường ngắn nhất: {best_distance}")
print(f"Chuỗi thành phố tốt nhất: {best_tour}")

# Đặt tên cho các thành phố (tùy chọn)
for i, city in enumerate(cities):
    plt.annotate(f"TP {i}", (city[0] + 0.005, city[1] + 0.005))


plt.xlabel("Tọa độ X")
plt.ylabel("Tọa độ Y")
plt.legend()
plt.title("Bài toán TSP và đường đi ngắn nhất")
plt.grid(True)
plt.show()
