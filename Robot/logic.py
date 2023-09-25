import math
import numpy as np
from collections import deque
from a_star import GridMapGraph, a_star_search

class Q: # State
    START, NORMAL, DEADLOCK, FINISH = range(4)

neighbors = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]

class LogicAlgorithm:
    def __init__(self, row_count, col_count):
        self.state = Q.START
        self.weight_map = np.zeros((row_count, col_count))
        self.prob_map = np.zeros((row_count, col_count))
        self.direction = 4

    def init_weight_map(self, environment):
        row_count, col_count = len(environment), len(environment[0])
        for x, row in enumerate(environment):
            for y, val in enumerate(row):
                # if val == 0: self.weight_map[x, y] = col_count - y  # free cell
                # elif val == 1: self.weight_map[x, y] = -1           # obstacle cell
                self.weight_map[x, y] = environment[x, y]
    
    def set_map(self, map):
        self.weight_map = map

    def set_prob_map(self, map):
        self.prob_map = map

    def four_neighbours(self, cur_pos):
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
            # neighbours.append(((x, y), direction_priority))
            neighbours.append((x, y))
        return neighbours
    
    def calculate_zeta(self, cur_pos, nb_pos):
        W2 = 1
        W1 = 0.5
        return self.prob_map[cur_pos] * W1 + self.prob_map[nb_pos] * W2

    # Implement two-step aution
    def two_step_aution(self, cur_pos):
        self.state = Q.NORMAL
        cur_neighbours = self.four_neighbours(cur_pos)
        bid_value = []
        for cur_nb, direction_priority in cur_neighbours:
            zeta_nb = self.calculate_zeta(cur_pos, cur_nb)
            nb_neighbours = self.four_neighbours(cur_nb)
            max_zeta = -1000
            for nb, dp in nb_neighbours:
                if nb == cur_pos:
                    continue
                if self.calculate_zeta(cur_nb, nb) > max_zeta:
                    max_zeta = self.calculate_zeta(cur_nb, nb)
            if zeta_nb == 0 and max_zeta == 0:
                c = math.inf
            else: c = 1 / (zeta_nb + max_zeta)
            bid_value.append((c, direction_priority, cur_nb))

        return bid_value

    def get_replan_wp(self, cur_pos):
        # bid_value_list2 = self.two_step_aution(cur_pos)
        bid_value_list = self.two_step_evaluation(cur_pos)
        # bid_value_list = sorted(bid_value_list, key=lambda c: (c[0], 1 / c[1]))
        # return [(i[0], i[2]) for i in bid_value_list]
        return self.get_score_max(bid_value_list)

    def two_step_evaluation(self, cur_pos):  
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
            # bid_value.append((c, cur_nb))
            bid_value[cur_nb] = c
        return bid_value
    
    def calculate_reward(self, cur_pos, goal):
        cur_neighbors = self.four_neighbours(cur_pos)
        delta_dist = dict()
        graph = GridMapGraph(self.weight_map)
        _, cur_dist = a_star_search(graph, cur_pos, goal)
        for nb in cur_neighbors:
            _, nb_dist = a_star_search(graph, nb, goal)
            # delta_dist.append((cur_dist - nb_dist, nb))
            # delta_dist[nb] = cur_dist - nb_dist
            if cur_dist - nb_dist > 0:
                delta_dist[nb] = 1
            else: delta_dist[nb] = 0

        return delta_dist

    def get_score_max(self, score_dict: dict):
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
        return wp
    
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
        row_count, col_count = len(self.weight_map), len(self.weight_map[0])
        (x, y) = current_pos

        if (x + 1) < row_count and self.weight_map[x + 1][y] == 0:
            return [(x + 1, y)]
        if (x - 1) >= 0 and self.weight_map[x - 1][y] == 0:
            return [(x - 1, y)]
        if y + 1 < col_count and self.weight_map[x][y+1] == 0:
            if self.direction == 3:
                return [(x, y + 1)]
        self.direction = 4
        if y - 1 > 0 and self.weight_map[x][y-1] == 0:
            return [(x, y - 1)]
        self.direction = 3
        if y + 1 < col_count and self.weight_map[x][y+1] == 0:
            return [(x, y + 1)]
        return []
    
    def escape_deadlock_path(self, current_pos):
        weight_map = self.weight_map

        queue = deque()
        visited = []
        parent = dict()
        deadlock_wp = None
        path = []

        queue.append(current_pos)
        visited.append(current_pos)
        parent[current_pos] = -1

        # neighbors = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
        neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        
        flag = True
        while queue:
            if flag == False:
                break
            cur_node = queue.popleft()
            # traverse neighbors
            for dx, dy in neighbors:
                x, y = cur_node[0] + dx, cur_node[1] + dy
                
                if x < 0 or x >= len(weight_map): continue
                if y < 0 or y >= len(weight_map[0]): continue

                if weight_map[x, y] in (1, 3): continue # obstacle
                elif weight_map[x, y] == 2 or weight_map[x, y] == 4:             # visited
                    if (x, y) not in visited:
                        visited.append((x, y))
                        queue.append((x, y))
                        parent[x, y] = cur_node
                    continue 
                else: 
                    deadlock_wp = (x, y)                   # unvisited
                    parent[deadlock_wp] = cur_node
                    flag = False
                    break
        
        if deadlock_wp == None:
            return []
        
        # path.append(deadlock_wp)
        while parent[deadlock_wp] != -1:
            path.append(deadlock_wp)
            deadlock_wp = parent[deadlock_wp]

        return path[::-1]
    
    def escape_deadlock_dynamic(self, cur_pos, goal):
        # wp_list = self.two_step_aution(cur_pos)
        bid_value_list = self.two_step_evaluation(cur_pos)
        reward_list = self.calculate_reward(cur_pos, goal)
        score_dict = dict()

        for pos in bid_value_list.keys():
            score_dict[pos] = 3/4 * bid_value_list[pos] + 1/4 * reward_list[pos]

        return self.get_score_max(score_dict)

        # bid_value = 0
        # for i in wp_list:
        #     if path[0] == i[1]:
        #         bid_value = i[0]

        # if bid_value == wp_list[0][0]:
            # return path[0]
        # if bid_value > 1/6:
        #     return path[0]
        # else:
        #     return wp_list[0][1]

    # find wp using wavefront similar method (when deadlock)
    def get_deadlock_wp(self, current_pos):
        weight_map = self.weight_map

        queue = deque()
        visited = []

        queue.append(current_pos)
        visited.append(current_pos)

        # neighbors = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
        neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        
        while queue:
            cur_node = queue.popleft()
            # traverse neighbors
            for dx, dy in neighbors:
                x, y = cur_node[0] + dx, cur_node[1] + dy
                
                if x < 0 or x >= len(weight_map): continue
                if y < 0 or y >= len(weight_map[0]): continue

                if weight_map[x, y] in (1, 3): continue # obstacle
                elif weight_map[x, y] == 2 or weight_map[x, y] == 4:             # visited
                    if (x, y) not in visited:
                        visited.append((x, y))
                        queue.append((x, y))
                    continue 
                else: return [(x, y)]                   # unvisited
        
        return []
    
    def predict(self, current_pos, step_count):
        waypoint_list = [current_pos]

        temporary_visited_list = []
        for _ in range(step_count):
            wp = self.get_wp(current_pos)
            if self.state == Q.FINISH:
                break
            elif self.state == Q.NORMAL:
                current_pos = wp[0]
                if self.weight_map[current_pos] == 2:
                    print("Haha")
                    pass
                temporary_visited_list.append(current_pos)
                self.weight_map[current_pos] = 2
                waypoint_list.append(current_pos)
            elif self.state == Q.DEADLOCK:
                graph = GridMapGraph(self.weight_map)
                path, dist = a_star_search(graph, current_pos, wp[0])
                current_pos = path[0]
                waypoint_list.append(current_pos)
            # if len(wp) == 0: break

        for pos in temporary_visited_list:
            self.weight_map[pos] = 0

        return waypoint_list
    
    def update_explored(self, pos):
        self.weight_map[pos] = 2


if __name__ == "__main__":
    logic = LogicAlgorithm(7, 7)
    
    environment = np.zeros((7, 7))
    environment[3, 3] = environment[3, 4] = environment[4, 3] = environment[4, 4] = 1
    logic.init_weight_map(environment)

    print(logic.weight_map)
