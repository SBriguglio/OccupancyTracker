import graph
import numpy as np

if __name__ == '__main__':
    t = graph.Graph()
    t.start_state = np.array([1, 0, 3, 4, 2], np.int32)
    t.add_vertex('0', state=t.start_state)
    t.add_vertex('G', t.goal_state)
    t.nodes += 1
    if t.a_star(t.get_vertex('0'), t.get_vertex('G')) == -1:
        print("Rekt")
    print("Solution")
    i = 0
    for n in t.exact:
        i += 1
        print("Step {}:".format(i))
        print(n.state)
    print(t.goal_state)