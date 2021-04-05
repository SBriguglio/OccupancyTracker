import graph
import numpy as np

if __name__ == '__main__':
    t = graph.Graph()
    t.start_state = np.array([1, 0, 3, 4, 2], np.int32)
    t.add_vertex('0', state=t.start_state, depth=0)
    t.add_vertex('G', t.goal_state)
    t.nodes += 1
    rec = t.a_star(t.get_vertex('0'), 3)
    print("Solution")
    i = 0
    for n in t.exact:
        i += 1
        print("Step {}:".format(i))
        print(n.state)
    print("RECOMMENDED STATE: {}".format(rec.state))
    print("GO TO CHECKOUT: {}".format(rec.add_to))