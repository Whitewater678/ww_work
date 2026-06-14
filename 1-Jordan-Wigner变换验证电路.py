import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Statevector

def jordan_wigner_transform(state_str):
    n = len(state_str)
    qc = QuantumCircuit(n)

    for i, bit in enumerate(state_str):
        if bit == '1':
            qc.x(i)

    return qc


if __name__ == "__main__":
    test_state = "0101"
    n_qubits = len(test_state)

    original_qc = QuantumCircuit(n_qubits)
    for i, bit in enumerate(test_state):
        if bit == '1':
            original_qc.x(i)

    jw_qc = jordan_wigner_transform(test_state)

    simulator = Aer.get_backend('statevector_simulator')

    state_original = simulator.run(original_qc).result().get_statevector()
    state_jw = simulator.run(jw_qc).result().get_statevector()

    print(f"原始自旋态 |{test_state}> 的振幅:")
    print(state_original)

    print(f"\nJordan-Wigner 变换后（费米子表示）的振幅:")
    print(state_jw)

    is_same = np.allclose(np.abs(state_original), np.abs(state_jw))
    print(f"\n验证结果: 两个态的系数是否一致? {is_same}")

    index = int(test_state, 2)
    print(f"\n对应的计算基态索引: {index} (二进制: {test_state})")