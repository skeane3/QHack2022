import sys
import pennylane as qml
from pennylane import numpy as np
from pennylane import hf


def ground_state_VQE(H):
    """Perform VQE to find the ground state of the H2 Hamiltonian.

    Args:
        - H (qml.Hamiltonian): The Hydrogen (H2) Hamiltonian

    Returns:
        - (float): The ground state energy
        - (np.ndarray): The ground state calculated through your optimization routine
    """

    # QHACK #
    qubits = 4
    # Define a device
    dev = qml.device("default.qubit", wires = qubits)
    # Define a circuit to prepare the trial state
    def circuit(param, wires):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

    # Define the cost function
    @qml.qnode(dev)
    def cost_fn(param):
        circuit(param, wires=range(qubits))
        return qml.expval(H)
    # Define an optimiser
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)



    max_iterations = 100
    conv_tol = 1e-06

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)

        energy = cost_fn(theta)
        angle = theta

        conv = np.abs(energy - prev_energy)

        if conv <= conv_tol:
            break

    print("\n" f"Final value of the ground-state energy = {energy:.8f} Ha")
    print("\n" f"Optimal value of the circuit parameter = {angle:.4f}")


    # Construct the ground state using the optimised parameter
    @qml.qnode(dev)
    def ground_state(param):
        circuit(param, wires=range(qubits))
        return qml.state()

    gs = ground_state(angle)
    #print(f'\n Ground state = {gs}')
    print(f'\n  size Ground state = {gs.shape}')

    return energy, gs
    # QHACK #


def create_H1(ground_state, beta, H):
    """Create the H1 matrix, then use `qml.Hermitian(matrix)` to return an observable-form of H1.

    Args:
        - ground_state (np.ndarray): from the ground state VQE calculation
        - beta (float): the prefactor for the ground state projector term
        - H (qml.Hamiltonian): the result of hf.generate_hamiltonian(mol)()

    Returns:
        - (qml.Observable): The result of qml.Hermitian(H1_matrix)
    """

    # QHACK #

    #print(f'Type H = {type(H)}')
    #print(H.matrix)
    #print(f'Type new_H = {type(new_H)}')
    #new_H = qml.Hermitian(beta*np.outer(ground_state, ground_state))
    H_per_matrix = beta*np.outer(ground_state, ground_state)
    H_per = qml.Hermitian(H_per_matrix, wires=[0, 1, 2, 3])

    print(f'Type H = {type(H)}')
    print(f'Type H_per = {type(H_per)}')

    H1 = H + H_per
    print(f'Type H1 = {type(H1)}')
    #print(H == H1)

    return H1
    # QHACK #


def excited_state_VQE(H1):
    """Perform VQE using the "excited state" Hamiltonian.

    Args:
        - H1 (qml.Observable): result of create_H1

    Returns:
        - (float): The excited state energy
    """

    # QHACK #
    qubits = 4
    # Define a device
    dev = qml.device("default.qubit", wires = qubits)
    # Define a circuit to prepare the trial state
    def circuit(param, wires):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

    # Define the cost function
    @qml.qnode(dev)
    def cost_fn(param):
        circuit(param, wires=range(qubits))
        return qml.expval(H1)
    # Define an optimiser
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)



    max_iterations = 100
    conv_tol = 1e-06

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)

        energy = cost_fn(theta)
        angle = theta

        conv = np.abs(energy - prev_energy)

        if conv <= conv_tol:
            break

    print("\n" f"Final value of the ground-state energy = {energy:.8f} Ha")
    print("\n" f"Optimal value of the circuit parameter = {angle:.4f}")


    return energy
    # QHACK #


if __name__ == "__main__":
    coord = float(sys.stdin.read())
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, -coord], [0.0, 0.0, coord]], requires_grad=False)
    mol = hf.Molecule(symbols, geometry)

    H = hf.generate_hamiltonian(mol)()
    E0, ground_state = ground_state_VQE(H)

    print()

    beta = 15.0
    H1 = create_H1(ground_state, beta, H)
    E1 = excited_state_VQE(H1)

    answer = [np.real(E0), E1]
    print(*answer, sep=",")
