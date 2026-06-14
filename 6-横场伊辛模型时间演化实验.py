import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import QFT
import matplotlib.pyplot as plt

# ============================================================
# 1. 基础组件（与之前保持一致）
# ============================================================
def jw_fswap(qc, i, j):
    qc.h(i)
    qc.h(j)
    qc.cx(i, j)
    qc.h(i)
    qc.h(j)
    qc.z(i)

def theta_k(lam, k, n):
    kk = 2 * np.pi * k / n
    return 0.5 * np.arctan2(
        2 * np.sin(kk),
        lam - np.cos(kk)
    )

def bogo_gate(qc, i, j, theta):
    qc.x(i)
    qc.x(j)
    qc.crx(2 * theta, j, i)
    qc.x(i)
    qc.x(j)

# ============================================================
# 2. 构造对角化电路 U_dis^\dagger
# ============================================================
def diagonalizing_circuit(lam, n):
    qc = QuantumCircuit(n)
    # Step 1: QFT
    qc.append(QFT(n, inverse=False), range(n))
    # Step 2: JW fSWAP
    for i in range(0, n, 2):
        if i + 1 < n:
            jw_fswap(qc, i, i + 1)
    # Step 3: Bogoliubov
    for k_idx in range(n // 2):
        theta = theta_k(lam, k_idx, n)
        bogo_gate(qc, 2 * k_idx, 2 * k_idx + 1, theta)
    # Step 4: IQFT
    qc.append(QFT(n, inverse=True), range(n))
    return qc

# ============================================================
# 3. 主程序：复现论文 Fig 6
# ============================================================
def main():
    n = 4
    lambdas = [0.5, 1.0, 1.5]
    times = np.linspace(0, 2.0, 100)

    plt.figure(figsize=(8, 6))
    colors = {0.5: 'blue', 1.0: 'orange', 1.5: 'green'}

    for lam in lambdas:
        results = []

        # 论文公式 (21) 的参数
        prefactor = (1 + 2 * lam**2) / (2 + 2 * lam**2)
        freq = 4 * np.sqrt(1 + lam**2)

        for t in times:
            # Step 1: |0000>
            qc = QuantumCircuit(n)

            # Step 2: U_dis^\dagger |0000>
            diag_inv = diagonalizing_circuit(lam, n).inverse()
            qc.compose(diag_inv, inplace=True)

            # Step 3: 时间演化相位 e^{-iHt}
            # 论文：相位 = 4 t sqrt(1+λ²)
            qc.rz(freq * t, 0)

            # Step 4: U_dis
            qc.compose(diagonalizing_circuit(lam, n), inplace=True)

            # Step 5: 测量 <σ_z>（第一个比特）
            sv = Statevector.from_instruction(qc)
            pauli_z = SparsePauliOp('Z' + 'I'*(n-1))
            exp_val = sv.expectation_value(pauli_z).real

            results.append(exp_val)

        # 画模拟曲线
        plt.plot(times, results, 'o-', color=colors[lam],
                 linewidth=2, markersize=6, label=f'Simulation λ={lam}')

        # 画理论曲线
        theory = prefactor + (1 / (2 + 2 * lam**2)) * np.cos(freq * times)
        plt.plot(times, theory, '--', color=colors[lam],
                 linewidth=2, label=f'Theory λ={lam}')

    plt.xlabel("Time (t)")
    plt.ylabel("⟨σ_z⟩")
    plt.title("Time Evolution of Transverse Magnetization (Fig 6)")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()