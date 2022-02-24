#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml

graph = {
    0: [1],
    1: [0, 2, 3, 4],
    2: [1],
    3: [1],
    4: [1, 5, 7, 8],
    5: [4, 6],
    6: [5, 7],
    7: [4, 6],
    8: [4],
}


def n_swaps(cnot):
    """Count the minimum number of swaps needed to create the equivalent CNOT.

    Args:
        - cnot (qml.Operation): A CNOT gate that needs to be implemented on the hardware
        You can find out the wires on which an operator works by asking for the 'wires' attribute: 'cnot.wires'

    Returns:
        - (int): minimum number of swaps
    """

    # QHACK #

    def find_all_paths(start, end, graph = graph, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = find_all_paths(node, end, graph, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def find_min_swaps(start, end):
        # Find all paths from start to end
        paths = find_all_paths(start, end)
        # The first index considers the first path found. The second slice removes the first and last nodes form the paths, as these do not require a swap
        min_swaps = len(paths[0][1:-1])
        # If there is only one path, then we can return our answer
        if len(paths) == 1:
            return min_swaps
        # Else, check the length of all other paths to see if any are shorter
        for i in range(1, len(paths)):
            swaps = len(paths[i][1:-1])
            if swaps < min_swaps:
                min_swaps = swaps
        return min_swaps

    start = cnot.wires[0]
    end = cnot.wires[1]

    # Multiply by 2 as this considers the one way path, but we must also make the same number of swaps to obtain the original topology
    min_swaps = 2*find_min_swaps(start, end)

    return min_swaps
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = n_swaps(qml.CNOT(wires=[int(i) for i in inputs]))
    print(f"{output}")
