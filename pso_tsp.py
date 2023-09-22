import numpy as np
from particle import Particle
from tsp_problem import TSPProblem


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
