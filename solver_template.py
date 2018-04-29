import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
from student_utils_sp18 import *
import itertools as it
import numpy as np
import networkx as nx
import copy
from random import shuffle
"""
======================================================================
  Complete the following function.
======================================================================
"""

def new_min_weighted_dominating_set(G, weight=None):
    """Returns a dominating set that approximates the minimum weight node
    dominating set.

    References
    ----------
    .. [1] Vazirani, Vijay V.
           *Approximation Algorithms*.
           Springer Science & Business Media, 2001.
    """
    if len(G) == 0:
        return set()

    dom_set = set()
    # dom_set.add(starting_kingdom)

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

def best_set_permutation(subset, start, dist_dict):
    subset_list = [node for node in subset]
    shortest_dist =np.inf
    shortest_path = []
    permutations = []
    for i in range(100):
        random.shuffle(subset_list)
        x = copy.deepcopy(subset_list)
        permutations.append(x)
    # permutations = it.permutations(subset_list)
    for elem in permutations:
        dist = get_distance(dist_dict, start, elem[0])
        for i in range(0, len(elem) - 1):
            dist += get_distance(dist_dict, elem[i], elem[i+1])
        dist = get_distance(dist_dict, elem[-1], start)
        if dist < shortest_dist:
            shortest_path = elem
            shortest_dist = dist
    return shortest_path

def get_distance(dist_dict, start_vertex, end_vertex):
    start_dict = dist_dict[start_vertex]
    dist = start_dict[end_vertex]
    return dist

def recreate_shortest_path(path_dict, start_vertex, end_vertex):
    saved_end = end_vertex
    start_dict = path_dict[start_vertex]
    path = []
    while end_vertex != start_vertex:
        path.append(start_dict[end_vertex])
        end_vertex = start_dict[end_vertex]
    path.reverse()
    return path

def final_tour_creator(path_through_subset, path_dict, start):
    path = recreate_shortest_path(path_dict, start, path_through_subset[0])
    for i in range(len(path_through_subset) - 1):
        path.extend(recreate_shortest_path(path_dict, path_through_subset[i], path_through_subset[i+1]))
    path.extend(recreate_shortest_path(path_dict, path_through_subset[-1], start))
    path.extend([start])
    return path

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
    # raise Exception('"solve" function not defined')
    # return closed_walk, conquered_kingdoms

    graph = adjacency_matrix_to_graph(adjacency_matrix)
    for i in range(len(list_of_kingdom_names)):
        if list_of_kingdom_names[i] == starting_kingdom:
            start = i
            break
    dominating_set = new_min_weighted_dominating_set(graph, weight='weight')
    path_dict, dist_dict = nx.floyd_warshall_predecessor_and_distance(graph, weight='weight')
    shortest_path_through_dom_set = best_set_permutation(dominating_set, start, dist_dict)
    path = final_tour_creator(shortest_path_through_dom_set, path_dict, start)
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
    output_file = output_directory + '/' + output_filename
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
