import numpy as np
import matplotlib.pyplot as plt
from tsp_problem import TSPProblem
from pso_tsp import pso_tsp

# Sử dụng bài toán TSP đã định nghĩa
custom_cities = [(0, 0), (1, 2), (3, 5), (7, 1), (8, 3)]  # Tọa độ các thành phố
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
