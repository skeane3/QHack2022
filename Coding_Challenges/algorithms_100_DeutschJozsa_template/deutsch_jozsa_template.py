import sys
import pennylane as qml
from pennylane import numpy as np


def deutsch_jozsa(oracle):
    """This function will determine whether an oracle defined by a function f is constant or balanced.

    Args:
        - oracle (function): Encoding of the f function as a quantum gate. The first two qubits refer to the input and the third to the output.

    Returns:
        - (str): "constant" or "balanced"
    """

    dev = qml.device("default.qubit", wires=3, shots=1)

    @qml.qnode(dev)
    def circuit():
        """Implements the Deutsch Jozsa algorithm."""

        # QHACK #

        # Insert any pre-oracle processing here
        # Put our ancilla in state |1>
        qml.PauliX(wires=2)

        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)

        oracle()  # DO NOT MODIFY this line

        # Insert any post-oracle processing here
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        # QHACK #

        return qml.sample(wires=range(2))

    sample = circuit()
    # QHACK #
    f = 'constant' # Assumes all entries in sample is a 0
    for s in sample:
        if s != 0:
            f = 'balanced'
            break
    # From `sample` (a single call to the circuit), determine whether the function is constant or balanced.
    return f
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    def oracle():
        for i in numbers:
            qml.CNOT(wires=[i, 2])

    output = deutsch_jozsa(oracle)
    print(output)
