from __future__ import division
import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
from student_utils_sp18 import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

#https://gist.github.com/mikkelam/ab7966e7ab1c441f947b#file-hamilton-py-L3
def hamilton(G):
    F = [(G,[G.nodes()[0]])]
    n = G.number_of_nodes()
    while F:
        graph,path = F.pop()
        confs = []
        for node in graph.neighbors(path[-1]):
            conf_p = path[:]
            conf_p.append(node)
            conf_g = nx.Graph(graph)
            conf_g.remove_node(path[-1])
            confs.append((conf_g,conf_p))
        for g,p in confs:
            if len(p)==n:
                return p
            else:
                F.append((g,p))
    # print('type:', type(F))
    return F #i think, im not sure


def index_satisfying(iterable, condition):
    """Returns the index of the first element in `iterable` that
    satisfies the given condition.

    If no such element is found (that is, when the iterable is
    exhausted), this returns the length of the iterable (that is, one
    greater than the last index of the iterable).

    `iterable` must not be empty. If `iterable` is empty, this
    function raises :exc:`ValueError`.

    """
    # Pre-condition: iterable must not be empty.
    for i, x in enumerate(iterable):
        if condition(x):
            return i
    # If we reach the end of the iterable without finding an element
    # that satisfies the condition, return the length of the iterable,
    # which is one greater than the index of its last element. If the
    # iterable was empty, `i` will not be defined, so we raise an
    # exception.
    try:
        return i + 1
    except NameError:
        raise ValueError('iterable must be non-empty')


def hamiltonian_path(G):
    """Returns a Hamiltonian path in the given tournament graph.
    """
    if len(G) == 0:
        return []
    if len(G) == 1:
        return [arbitrary_element(G)]
    v = arbitrary_element(G)
    hampath = hamiltonian_path(G.subgraph(set(G) - {v}))
    # Get the index of the first node in the path that does *not* have
    # an edge to `v`, then insert `v` before that node.
    index = index_satisfying(hampath, lambda u: v not in G[u])
    hampath.insert(index, v)
    return hampath

#https://github.com/MUSoC/Visualization-of-popular-algorithms-in-Python/tree/master/Travelling%20Salesman%20Problem
def traveling_salesman_problem(G, kingdoms_to_visit, starting_kingdom, weight=None):
    s_t = steiner_tree(G, kingdoms_to_visit)
    steiner_tree_vertices = s_t.nodes()
    odd_degree_vertices = []
    for v in steiner_tree_vertices:
        if len(G.edges(nbunch=v)) % 2 == 1:
            odd_degree_vertices.append((v))
    odd_subgraph = G.copy()

    temp_edges  = copy.deepcopy(odd_subgraph.edges())
    for e in temp_edges:
        # this may be and instead of or, check this
        if e[0] not in odd_degree_vertices or e[1] not in odd_degree_vertices:
            odd_subgraph.remove_edge(e[0], e[1])
    match = maximal_matching(odd_subgraph)
    for e in match:
        if e not in s_t.edges():
            s_t.add_edge(e)

    ham_path = hamiltonian_path(s_t)
    return ham_path

def steiner_tree(G, kingdoms_to_visit):
    s_t = nx.algorithms.approximation.steinertree.steiner_tree(G, list(kingdoms_to_visit))
    return s_t

def maximal_matching(G):
    r"""Find a maximal matching in the graph.
    """
    matching = set()
    nodes = set()
    for u, v in G.edges():
        # If the edge isn't covered, add it to the matching
        # then remove neighborhood of u and v from consideration.
        if u not in nodes and v not in nodes and u != v:
            matching.add((u, v))
            nodes.add(u)
            nodes.add(v)
    return matching


def new_min_weighted_dominating_set(starting_kingdom, G, weight=None):
    """Returns a dominating set that approximates the minimum weight node
    dominating set.
"""
    if len(G) == 0:
        return set()

    dom_set = set()
    dom_set.add(starting_kingdom)

    def _cost(node_and_neighborhood):
        v, neighborhood = node_and_neighborhood
        x = G.nodes[v]
        z = G.nodes[v].get(weight) / len(neighborhood - dom_set)
        return G.nodes[v].get(weight) / len(neighborhood - dom_set)

    vertices = set(G)

    neighborhoods = {v: {v} | set(G[v]) for v in G}

    while vertices:
        dom_node, min_set = min(neighborhoods.items(), key=_cost)
        dom_set.add(dom_node)
        del neighborhoods[dom_node]
        vertices -= min_set

    return dom_set

def solve(list_of_kingdom_names, starting_kingdom, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_kingdom_names: An list of kingdom names such that node i of the graph corresponds to name index i in the list
        starting_kingdom: The name of the starting kingdom for the walk
        adjacency_matrix: The adjacency matrix from the input file

    Output:
        Return 2 things. The first is a list of kingdoms representing the walk, and the second is the set of kingdoms that are conquered
    """
    #raise Exception('"solve" function not defined')
    G = adjacency_matrix_to_graph(adjacency_matrix)
    return new_min_weighted_dominating_set(G, weight="weight")
    # return closed_walk, conquered_kingdoms

    graph = adjacency_matrix_to_graph(adjacency_matrix)
    dominating_set = new_min_weighted_dominating_set(starting_kingdom, graph, weight='weight')
    path = traveling_salesman_problem(graph, dominating_set, starting_kingdom, weight='weight')
    return path, list(dominating_set)



"""
======================================================================
   No need to change any code below this line
======================================================================
"""


def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    number_of_kingdoms, list_of_kingdom_names, starting_kingdom, adjacency_matrix = data_parser(input_data)
    closed_walk, conquered_kingdoms = solve(list_of_kingdom_names, starting_kingdom, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    output_filename = utils.input_to_output(filename)
    output_file = '{output_directory}/{output_filename}'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    utils.write_data_to_file(output_file, closed_walk, ' ')
    utils.write_to_file(output_file, '\n', append=True)
    utils.write_data_to_file(output_file, conquered_kingdoms, ' ', append=True)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
