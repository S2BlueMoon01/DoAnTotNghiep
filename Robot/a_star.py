from copy import deepcopy
import math
from queue import PriorityQueue

# undirected graph
class GridMapGraph:
    def __init__(self, map):
        self.map = map
        self.row_count = len(map)
        self.col_count = len(map[0])
    
    def neighbors(self, node):
        neighbor_list = []
        # rotate = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
        rotate = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        for i in range(len(rotate)):
            x, y = node[0] + rotate[i][0], node[1] + rotate[i][1]
            if x < 0 or x >= self.row_count: continue
            if y < 0 or y >= self.col_count: continue
            if self.map[(x, y)] not in (0, 2, 4): continue     # unvisited or visited or threat

            neighbor_list.append((x, y))
        
        return neighbor_list
    
    def weight(self, node1, node2):
        if node1 in self.neighbors(node2):
            return math.dist(node1, node2)

        return math.inf

def heuristic(pos_from, pos_to):
    # return abs(pos_from[0] - pos_to[0]) + abs(pos_from[1] - pos_to[1]) # manhattan distance
    return math.dist(pos_from, pos_to) # euclidean distance

def a_star_search(graph: GridMapGraph, start, goal):
    open = PriorityQueue() # tuple: (priority f, item)
    open.put((heuristic(start, goal), start))
    came_from = {}  # key: node, value: parent node
    g = {}  # key: node, value: g cost
    g[start] = 0

    while not open.empty():
        cur_node = open.get()[1]

        if cur_node == goal: break
        
        for neighbor in graph.neighbors(cur_node):
            new_g_cost = g[cur_node] + graph.weight(cur_node, neighbor)
            if neighbor not in g.keys() or new_g_cost < g[neighbor]:
                g[neighbor] = new_g_cost
                f = new_g_cost + heuristic(neighbor, goal)
                open.put((f, neighbor))
                came_from[neighbor] = cur_node

    if cur_node != goal: raise Exception('A Star path not found')

    path = []
    while cur_node != start:
        path.insert(0, cur_node)
        cur_node = came_from[cur_node]
    
    return path, g[goal]
