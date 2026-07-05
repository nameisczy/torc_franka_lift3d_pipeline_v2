from random import uniform
import matplotlib.pyplot as plt
from task_planner.dep_graph import DepGraph
import networkx as nx

pickable = {1: 4, 5: 48}
obj_deps = {
    0: {
        3: {
            "g": 2,
            "d": "grasps blocked by"
        },
        9: {
            "g": 2,
            "d": "grasps blocked by"
        },
        4: {
            "d": "behind"
        },
        5: {
            "d": "behind"
        }
    },
    1: {
        4: {
            "g": 6,
            "d": "grasps blocked by"
        }
    },
    4: {},
    5: {
        4: {
            "g": 24,
            "d": "grasps blocked by"
        }
    },
    7: {
        1: {
            "d": "behind"
        }
    },
    8: {
        7: {
            "g": 2,
            "d": "grasps blocked by"
        },
        5: {
            "g": 5,
            "d": "grasps blocked by"
        },
        4: {
            "g": 3,
            "d": "grasps blocked by"
        },
        11: {
            "g": 1,
            "d": "grasps blocked by"
        },
        1: {
            "g": 1,
            "d": "grasps blocked by"
        }
    },
    2: {
        1: {
            "d": "behind"
        }
    },
    3: {
        5: {
            "d": "behind"
        },
        11: {
            "d": "behind"
        }
    },
    6: {
        1: {
            "d": "behind"
        },
        4: {
            "d": "behind"
        }
    },
    9: {
        4: {
            "d": "behind"
        },
        5: {
            "d": "behind"
        }
    },
    10: {
        5: {
            "d": "behind"
        },
        11: {
            "d": "below"
        }
    },
    11: {
        5: {
            "d": "behind"
        }
    }
}

G = DepGraph(obj_deps, pickable)
G.keep_only([1,2,3,4,8])
for i in G.nx_graph.nodes:
    G.nx_graph.nodes[i]['v'] = uniform(0, 100)
G.normalize_edges()
# print(G.sinks())
G.draw(fname='/tmp/test.png')
# print(G.gen_graphml(G.nx_graph))
# print(G.describe())
