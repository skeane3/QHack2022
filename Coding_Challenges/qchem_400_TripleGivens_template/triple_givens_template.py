import sys
import pennylane as qml
from pennylane import numpy as np

NUM_WIRES = 6


def triple_excitation_matrix(gamma):
    """The matrix representation of a triple-excitation Givens rotation.

    Args:
        - gamma (float): The angle of rotation

    Returns:
        - (np.ndarray): The matrix representation of a triple-excitation
    """

    # QHACK #
    # The triple excitation operator must act on a 6 qubit state () and so will be 6x6
    n = 2**6 #  size of the 6 qubit state
    op = np.zeros(n*n).reshape(n, n)

    op[int(n/2)-1, int(n/2)-1] = np.cos(gamma)
    op[int(n/2)-1, int(n/2)] = -np.sin(gamma)
    op[int(n/2), int(n/2)-1] = np.sin(gamma)
    op[int(n/2), int(n/2)] = np.cos(gamma)

    return op
    # QHACK #


dev = qml.device("default.qubit", wires=6)


@qml.qnode(dev)
def circuit(angles):
    """Prepares the quantum state in the problem statement and returns qml.probs

    Args:
        - angles (list(float)): The relevant angles in the problem statement in this order:
        [alpha, beta, gamma]

    Returns:
        - (np.tensor): The probability of each computational basis state
    """

    # QHACK #
    # Prepare the state in |111000>
    qml.BasisState(np.array([1, 1, 1, 0, 0, 0]), wires=[0, 1, 2, 3, 4, 5])
    TripleExcitation = triple_excitation_matrix(angles[2])

    # First, act with G1
    qml.SingleExcitation(angles[0], wires=[0, 5])
    qml.ctrl(qml.DoubleExcitation, control = 0)(angles[1], wires = [0, 1, 4, 5])
    qml.ControlledQubitUnitary(TripleExcitation, control_wires=[0], wires = [0, 1, 2, 3, 4, 5])
    #qml.ctrl(qml.TripleExcitation, control = 0)(angles[2])


    # QHACK #

    return qml.probs(wires=range(NUM_WIRES))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float)
    probs = circuit(inputs).round(6)
    print(*probs, sep=",")
