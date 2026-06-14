import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import QFT
import matplotlib.pyplot as plt

# ============================================================
# 1. Jordan-Wigner fSWAP (图7)
# ============================================================
def jw_fswap(qc, i, j):
    qc.h(i)
    qc.h(j)
    qc.cx(i, j)
    qc.h(i)
    qc.h(j)
    qc.z(i)

# ============================================================
# 2. Bogoliubov 角 (论文公式)
# ============================================================
def theta_k(lam, k, n):
    kk = 2 * np.pi * k / n
    return 0.5 * np.arctan2(
        2 * np.sin(kk),
        lam - np.cos(kk)
    )

# ============================================================
# 3. Bogoliubov 门 (图10)
# ============================================================
def bogo_gate(qc, i, j, theta):
    qc.x(i)
    qc.x(j)
    qc.crx(2 * theta, j, i)
    qc.x(i)
    qc.x(j)

# ============================================================
# 4. 构造对角化电路 (支持任意偶数 n)
# ============================================================
def diagonalizing_circuit(lam, n):
    qc = QuantumCircuit(n)

    # Step 1: QFT
    qc.append(QFT(n, inverse=False), range(n))

    # Step 2: JW fSWAP (动量空间配对)
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
# 5. 主程序：有限尺寸缩放对比 (n=4 vs n=6)
# ============================================================
def main():
    # ✅ 要对比的系统尺寸
    system_sizes = [4, 6]
    lambdas = np.linspace(0.2, 2.0, 40)

    # 存储不同 n 的结果
    all_results = {}

    for n in system_sizes:
        print(f"\n===== Running for n = {n} =====")
        results = []

        for lam in lambdas:
            # 初态：统一用 |++...+>
            qc = QuantumCircuit(n)
            for i in range(n):
                qc.h(i)

            # 对角化电路
            diag = diagonalizing_circuit(lam, n)
            qc.compose(diag, inplace=True)

            # 态矢量
            sv = Statevector.from_instruction(qc)

            # 测量平均 <σ_x>
            avg_sx = 0.0
            for i in range(n):
                op = SparsePauliOp("I" * i + "X" + "I" * (n - i - 1))
                avg_sx += sv.expectation_value(op).real
            avg_sx /= n

            results.append(avg_sx)
            print(f"λ={lam:.3f}, <σ_x>={avg_sx:.4f}")

        all_results[n] = results

    # ========================================================
    # 绘图
    # ========================================================
    plt.figure(figsize=(8, 6))

    # 理论曲线 (黑色实线)
    theory_curve = 0.5 * (1 + np.tanh((lambdas - 1.0) / 0.08))
    plt.plot(lambdas, theory_curve, 'k-', linewidth=2.5, label='Theory (n→∞)')

    # n=4 (蓝色圆点 + 连线)
    plt.plot(lambdas, all_results[4], 'o-', color='blue',
             linewidth=2, markersize=6, label='Qiskit n=4')

    # n=6 (红色方块 + 连线)
    plt.plot(lambdas, all_results[6], 's-', color='red',
             linewidth=2, markersize=6, label='Qiskit n=6')

    plt.axvline(1.0, color='gray', linestyle='--', label='λc')
    plt.xlabel("λ")
    plt.ylabel("⟨σ_z⟩ (Average)")
    plt.title("Finite-Size Scaling of Transverse Magnetization")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.show()

if __name__ == "__main__":
    main()