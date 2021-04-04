import collections
import math
import numpy as np
from numpy.random._generator import default_rng


class Vertex:
    def __init__(self, node, state=None):
        self.name = node
        self.discovered = False
        self.leaves = {}
        self.ancestor = []
        self.heuristic = 0
        self.state = state

    def __str__(self):
        return self.name + ' leaves: ' + str([x.name for x in self.leaves])

    def add_leaf(self, leaf, time):
        self.leaves[leaf] = time

    def add_ancestor(self, anc):
        self.ancestor.append(anc)

    def discover(self):
        self.discovered = True

    def lose(self):
        self.discovered = False

    def get_leaves(self):
        return self.leaves.keys()

    def get_ancestor(self):
        return self.ancestor

    def get_name(self):
        return self.name

    def get_time(self, leaf):
        return self.leaves[leaf]

    def is_discovered(self):
        return self.discovered

    def set_heuristic(self, heuristic_cost):
        self.heuristic = heuristic_cost

    def set_state(self, state):
        self.state = state


def t_list_contains(q, s):
    for t in q:
        if t[1] == s:
            return True
    return False


def check_solvable(state):
    # put puzzle into row-major order
    row_maj = state.reshape((1, 9))
    inv_count = 0
    # count number of inversions
    for i in range(0, 8):
        for j in range(i + 1, 9):
            if row_maj[0, j] < row_maj[0, i]:
                inv_count += 1
    # return True if even
    return inv_count % 2 == 0


def generate_random_8puzzle():
    # generate 8-puzzles until a solvable puzzle is created, return the solvable puzzle
    while 1:
        r = default_rng()
        # select random numbers 0-9 without reselecting numbers, create an array and reshape to 3x3 matrix
        state = np.array(r.choice(9, size=9, replace=False).reshape((3, 3)), np.int32)
        if check_solvable(state):
            return state


def display_8puzzle(state):
    for i in range(0, 3):
        for j in range(0, 3):
            n = state[i, j]
            if n == 0:
                n = '*'
            print(n, end="  ")
        print()
    print(" ")


def swap_tiles(state, t0, t1):
    # RETURNS a swapped state, will not alter the original
    new_state = np.array(state)
    new_state[t0] = state[t1]
    new_state[t1] = state[t0]
    return new_state


def misplaced_tile(state, g_state):
    misplaced_count = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if state[i, j] != g_state[i, j]:
                misplaced_count += 1
    return misplaced_count


def manhattan_distance(state, g_state):
    def man_single_tile_cost(s, g_s, si, sj):
        for i1 in range(0, 3):
            for j1 in range(0, 3):
                if s[si, sj] == g_s[i1, j1]:
                    return abs(i1 - i) + abs(j1 - j)

    manhattan = 0
    for i in range(0, 3):
        for j in range(0, 3):
            manhattan += man_single_tile_cost(state, g_state, i, j)
    return manhattan


def max_mis_man(state, g_state):
    return max(misplaced_tile(state, g_state), manhattan_distance(state, g_state))


class Graph:
    def __init__(self, mode=0, start_state=None):
        self.vertexes = {}
        self.edges = {}
        self.n_vertexes = 0
        self.visited = {}
        self.exact = collections.deque()
        self.traversal = collections.deque()
        self.nodes = 0
        self.state_library = {}  # maps states to their vertex
        if start_state is None:
            self.start_state = generate_random_8puzzle()
        else:
            self.start_state = start_state
        self.build_mode = mode
        self.goal_state = np.array([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 0]], np.int32)
        self.states = [self.start_state]

    def __iter__(self):
        return iter(self.vertexes.values())

    def add_vertex(self, node, heuristic=None, state=None):
        self.n_vertexes = self.n_vertexes + 1
        if state is not None:
            vert = Vertex(node, state)
        elif state is None:
            vert = Vertex(node)
        self.vertexes[node] = vert
        vert.heuristic = heuristic
        return vert

    def add_vertex_node(self, node):
        self.n_vertexes = self.n_vertexes + 1
        self.vertexes[node.name] = node

    def add_edge(self, node, leaf, time):
        self.edges[(node, leaf)] = time

    def get_edge_time(self, node, leaf):
        return self.edges[(node, leaf)]

    def get_vertex(self, node):
        if node in self.vertexes:
            return self.vertexes[node]
        else:
            return None

    def connect(self, node, leaf, time):
        if node.name not in self.vertexes:
            self.add_vertex_node(node)
        if leaf not in self.vertexes:
            self.add_vertex_node(leaf)
        self.add_edge(node, leaf, time)
        self.vertexes[node.name].add_leaf(self.vertexes[leaf.name], time)
        self.vertexes[leaf.name].add_ancestor(self.vertexes[node.name])

    def get_vertexes(self):
        return self.vertexes.keys()

    def lose_all(self):
        self.traversal.clear()
        self.exact.clear()
        for u in self.vertexes:
            self.vertexes[u].lose()

    def getCost(self, move):
        if self.build_mode == 0:
            return misplaced_tile(move, self.goal_state)
        elif self.build_mode == 1:
            return manhattan_distance(move, self.goal_state)
        elif self.build_mode == 2:
            return max_mis_man(move, self.goal_state)

    def process_new_state(self, t0, t1, state, ancestor):
        new_state = swap_tiles(state, t0, t1)
        immutable_new_state = new_state.tobytes()
        if not self.state_library.__contains__(immutable_new_state):
            leaf = Vertex(str(self.nodes), new_state)
            self.state_library[immutable_new_state] = leaf
            self.nodes += 1
            cost = self.getCost(leaf.state)
            self.connect(ancestor, leaf, cost)

    def generate_play_graph(self, v):
        # iterate through the matrix to find all valid moves
        # v = self.get_vertex(v)
        for i in range(0, 3):
            for j in range(0, 3):
                if v.state[i, j] == 0:
                    for n in range(-1, 2):
                        si = i + n
                        if 0 <= si <= 2:
                            for m in range(-1, 2):
                                sj = j + m
                                if 0 <= sj <= 2:
                                    if n == 0:
                                        if m != 0:
                                            # create new state for valid move and process it
                                            self.process_new_state((i, j), (si, sj), v.state, v)
                                    if m == 0:
                                        if n != 0:
                                            # create new state for valid move and process it
                                            self.process_new_state((i, j), (si, sj), v.state, v)

    def a_star(self, a, z):
        opened = {a}
        closed = {}
        costs = {a: 0}
        ancestors = {}
        while opened:
            minimum = math.inf
            # select minimum cost state
            for o in opened:
                if costs[o] < minimum:
                    minimum = costs[o]
                    i = o
            try:
                if opened.__contains__(i):
                    opened.remove(i)
            except KeyError:
                print("1 {}".format(KeyError))
                print("2 {}".format(self.get_vertex(i.get_name())))
                print("3 {}".format(i.get_name()))
                print("4 {}\n".format(type(i.get_name())))
            node = i
            self.traversal.append(node)
            # check is this is the goal state
            if (node.state == self.goal_state).all():
                while True:
                    self.exact.appendleft(node)
                    if (node.state == self.start_state).all():
                        return #costs[]
                    node = ancestors[node]

            # create leaves
            self.generate_play_graph(node)

            # expand leaves
            for leaf in node.leaves:
                # if leaf is not present in opened or closed, add it to opened and add it's costs, add it's ancestor
                # Note that get_edge_time returns the cost calculated by Manhattan distance or misplaced tile
                # that is stored in the self.edges list which represents h(n) in this version of A*
                if not (opened.__contains__(self.get_vertex(leaf.get_name())) or closed.__contains__(
                        self.get_vertex(leaf.get_name()))):
                    opened.add(self.get_vertex(leaf.get_name()))
                    # f(n) = g(n) + h(n)
                    costs[self.get_vertex(leaf.get_name())] = costs[node] + self.get_edge_time(node, leaf)
                    ancestors[leaf] = node

                # if leaf IS present in opened, update its cost and ancestor if the cost is cheaper
                elif opened.__contains__(self.get_vertex(leaf.get_name()).state.tobytes()):
                    # F(n) > f(n) --> F(n) = g(n) + h(n)
                    if costs[self.get_vertex(leaf.get_name())] > costs[node] + self.get_edge_time(node, leaf):
                        # f(n) = (g(n)) + h(n)
                        costs[self.get_vertex(leaf.get_name())] = costs[node] + self.get_edge_time(node, leaf)
                        ancestors[leaf] = node

                # if leaf IS present in closed and its cost is cheaper, move it back to opened and update its cost
                # and ancestor
                elif closed.__contains__(self.get_vertex(leaf.get_name())):
                    if costs[self.get_vertex(leaf.get_name())] > costs[node] + self.get_edge_time(node, leaf):
                        opened.add(self.get_vertex(leaf.get_name()))
                        closed.pop(self.get_vertex(leaf.get_name()))
                        costs[self.get_vertex(leaf.get_name())] = costs[node] + self.get_edge_time(node, leaf)
                        ancestors[leaf] = node
            closed[node] = 0
        return -1