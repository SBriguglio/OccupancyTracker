import collections
from _datetime import datetime
import math
import numpy as np
from random import seed
from random import choice


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


class Graph:
    def __init__(self, start_state=None, n_registers=5, avg_reg_throughput=1, queue_limit=5):
        self.vertexes = {}
        self.edges = {}
        self.n_vertexes = 0
        self.visited = {}
        self.exact = collections.deque()
        self.traversal = collections.deque()
        self.nodes = 0
        self.state_library = {}  # maps states to their vertex
        self.n_reg = n_registers
        self.reg_output = avg_reg_throughput  # average register throughput per minute
        self.q_limit = queue_limit
        self.wait_flag = False
        if start_state is None:
            self.start_state = np.zeros((n_registers,), np.int32)
        else:
            self.start_state = start_state
        self.goal_state = np.zeros((n_registers,), np.int32)
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

    # NEEDS TO BE UPDATED!
    # Cost here is traversal cost, NOT node cost which is based on heuristic
    def getCost(self, move):
        # cost is representative of risk calculated using a number of metrics
        # first, the risk associated with isle length is n^(n-1) where n in the number of people in the isle
        # isle risk has an additional risk added based on the risk from adjacent isles which is calculated using the
        # inverse square law where the risk of adjacent isles is divided by the square of it's distance+1
        # the total cost/risk of a node is the summation of these two risks
        risk = np.zeros((self.n_reg,), np.int32)
        ext_risk = np.zeros((self.n_reg,), np.int32)
        total_risk = 0
        for i in range(0, self.n_reg):
            if move[i] > 0:
                risk[i] = move[i] ** (move[i] - 1)
                e_risk = 0
                for j in range(0, self.n_reg):
                    if j != i and risk[j] != 0:
                        distance = (j-i)
                        e_risk += risk[j] / distance**2
                ext_risk[i] = e_risk
                total_risk += risk[i] + ext_risk[i]
        return total_risk

    def random_throughput(self, state):  # randomly removes customers from n registers where n is the avg throughput
        roles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for i in range(self.n_reg):
            seed(datetime.now())
            role = choice(roles)
            if state[i] != 0:
                if role > self.reg_output:
                    state[i] -= 1
        return state

    def process_new_state(self, i, state, ancestor):
        new_state = np.array(state)
        new_state[i] = new_state[i] + 1
        immutable_new_state = new_state.tobytes()
        if not self.state_library.__contains__(immutable_new_state):
            leaf = Vertex(str(self.nodes), new_state)
            self.state_library[immutable_new_state] = leaf
            self.nodes += 1
            cost = self.getCost(leaf.state)
            self.connect(ancestor, leaf, cost)

    def generate_checkout_graph(self, v):
        # iterate through the checkout array to find valid places to add a customer to the queue
        # if no moves result in queue length under the limit, then wait_flag is set
        # randomly processes output of checkout
        wait_flag = True
        new_state = self.random_throughput(np.array(v.state))
        for i in range(0, self.n_reg):
            if new_state[i] < self.q_limit:
                self.process_new_state(i, new_state, v)
                wait_flag = False
        self.wait_flag = wait_flag
        return wait_flag

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
            self.generate_checkout_graph(node)

            # expand leaves
            for leaf in node.leaves:  # NEED TO MAKE SURE TO ADD HEURISTICS HERE
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