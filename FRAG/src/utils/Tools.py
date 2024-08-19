import igraph as ig
from utils.Query import Query
from typing import Generator
import json
from typing import List, Set


def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("type.type.") or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation or "sameas" in relation:
        return True


def get_k_hop_neighbors(G: ig.Graph, seed_list: List[str], hop: int) -> ig.Graph:
    if hop == -1:
        return G
    visited = set(seed_list)
    current_layer = set(seed_list)
    for _ in range(hop):
        next_layer = set()
        for node in current_layer:
            neighbors = G.neighbors(node, mode="out")
            new_neighbors = set(neighbors) - visited
            next_layer.update(new_neighbors)
            visited.update(new_neighbors)
        current_layer = next_layer
    return G.subgraph(visited)


def get_query(path: str) -> Generator[Query, None, None]:
    with open(path) as fp:
        for line in fp:
            data = json.loads(line)
            G = ig.Graph(directed=True)
            edges = []
            nodes = set()
            for triple in data["subgraph"]:
                head, rel, tail = triple
                nodes.add(head)
                nodes.add(tail)
                edges.append((head, tail, rel))
            G.add_vertices(list(nodes))
            G.add_edges([(edge[0], edge[1]) for edge in edges])
            G.es["name"] = [edge[2] for edge in edges]
            data["subgraph"] = G
            yield Query(data)
