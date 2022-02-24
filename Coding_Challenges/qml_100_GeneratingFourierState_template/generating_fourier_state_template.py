#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def generating_fourier_state(n_qubits, m):
    """Function which, given the number of qubits and an integer m, returns the circuit and the angles that generate the state
    QFT|m> following the above template.

    Args:
        - n_qubits (int): number of qubits in the circuit.
        - m (int): basis state that we generate. For example, for 'm = 3' and 'n_qubits = 4'
        we would generate the state QFT|0011> (3 in binary is 11).

    Returns:
       - (qml.QNode): circuit used to generate the state.
       - (list[float]): angles that generate the state QFT|m>.
    """

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(angles):
        """This is the quantum circuit that we will use."""

        # QHACK #
        '''
        # Convert m to a binary list
        m_binary = []
        k = int(2**(n_qubits-1))
        for i in range(n_qubits):
            if m/k >= 1:
                m -= k
                m_binary.append(1)
            else:
                m_binary.append(0)
            k = k/2
        '''

        # First, prepare the register in the state |m>
        m_binary = np.binary_repr(m, n_qubits)

        for i in range(len(m_binary)):
            if m_binary[i] == '1':
                qml.PauliX(wires = i)


        # Add the template of the statement with the angles passed as an argument.
        for i in range(len(angles)):
            qml.Hadamard(wires = i)
            qml.RZ(angles[i], wires = i)
        # QHACK #

        # We apply QFT^-1 to return to the computational basis.
        # This will help us to see how well we have done.
        qml.adjoint(qml.QFT)(wires=range(n_qubits))

        # We return the probabilities of seeing each basis state.
        return qml.probs(wires=range(n_qubits))

    def error(angles):
        """This function will determine, given a set of angles, how well it approximates
        the desired state. Here it will be necessary to call the circuit to work with these results.
        """

        probs = circuit(angles)
        # QHACK #

        # The return error should be smaller when the state m is more likely to be obtained.
        error = 0
        n = len(probs)

        for i in range(n):
            if i != m:
                #error += abs(probs[i])
                error += (probs[i])**2

        #return error
        return np.sqrt(error)
        # QHACK #

    # This subroutine will find the angles that minimize the error function.
    # Do not modify anything from here.

    opt = qml.AdamOptimizer(stepsize=0.8)
    epochs = 5000

    angles = np.zeros(n_qubits, requires_grad=True)

    for epoch in range(epochs):
        print(epoch)
        angles = opt.step(error, angles)
        angles = np.clip(opt.step(error, angles), -2 * np.pi, 2 * np.pi)

    return circuit, angles


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    n_qubits = int(inputs[0])
    m = int(inputs[1])

    output = generating_fourier_state(n_qubits, m)
    output[0](output[1])
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def check_with_arbitrary_state():
        for i in range(n_qubits):
            qml.RY(i, wires=i)
        for op in output[0].qtape.operations:
            qml.apply(op)
        return qml.state()

    print(",".join([f"{p.real.round(5)},{p.imag.round(5)}" for p in check_with_arbitrary_state()]))
