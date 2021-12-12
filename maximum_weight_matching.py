import json
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

import networkx as nx
from click import secho
from networkx.algorithms import bipartite

def main():
    # Initialise the graph
    B = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    left_nodes = ["T1_q1", "T1_q2"]
    right_nodes = ["A", "B", "C", "D"]
    B.add_nodes_from(left_nodes, bipartite=0)
    B.add_nodes_from(right_nodes, bipartite=1)

    # Add edges with weights
    B.add_edge(1, "A", weight=1)
    B.add_edge(1, "B", weight=4)
    B.add_edge(1, "C", weight=2)
    B.add_edge(1, "D", weight=1)
    B.add_edge(2, "A", weight=3)
    B.add_edge(2, "B", weight=1)
    B.add_edge(2, "C", weight=2)
    B.add_edge(2, "D", weight=2)
    # Obtain the minimum weight full matching
    my_matching = bipartite.matching.minimum_weight_full_matching(B, left_nodes, "weight")

if __name__ == '__main__':
    main()