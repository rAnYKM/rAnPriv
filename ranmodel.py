import networkx as nx
from ran_graph import RanGraph

soc_node = {
    0: [0, 2, 4],
    1: [0, 3, 5],
    2: [1, 2, 4],
    3: [0, 2, 4],
    4: [1, 2, 4],
    5: [0, 3, 5],
    6: [0, 2, 3],
    7: [1, 4],
    8: [0, 2, 3, 4, 5],
    9: [1, 2, 3, 5]
}

soc_edge = [
    (0, 1), (0, 2), (0, 4), (0, 5), (0, 8),
    (1, 2), (1, 3), (1, 5), (1, 6),
    (2, 4), (2, 7), (2, 9),
    (3, 5), (3, 8), (3, 9),
    (4, 5), (4, 6),
    (5, 7), (5, 8),
    (6, 9)
]

g = nx.Graph()
attr_edge = [('n' + str(n), 'a' + str(m)) for n, li in soc_node.items() for m in li]
edges = [('n' + str(n[0]), 'n' + str(n[1])) for n in soc_edge]
nodes = ['n' + str(n) for n in range(0, 10, 1)]
attr_node = ['a' + str(n) for n in range(0, 6, 1)]
g.add_nodes_from(nodes, lab='social node')
g.add_nodes_from(attr_node, lab='attribute node')
g.add_edges_from(attr_edge)
g.add_edges_from(edges)
r = RanGraph(nodes, attr_node, edges, attr_edge)
for n in r.soc_attr_net.neighbors('a0'):
    r.soc_attr_net[n]['a0']['weight'] = r.prob_given_feature('a0', [a for a in r.soc_attr_net.neighbors(n) if a != 'a0'])
for n in r.soc_attr_net.neighbors('a1'):
    r.soc_attr_net[n]['a1']['weight'] = r.prob_given_feature('a1', [a for a in r.soc_attr_net.neighbors(n) if a != 'a1'])
print 'done'
nx.write_gexf(r.soc_attr_net, 'g.gexf')
