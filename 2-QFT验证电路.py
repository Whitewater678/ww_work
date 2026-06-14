import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def qft_circuit_manual(n):
    """手动构建 n 量子比特的 QFT 电路（论文标准写法）"""
    qc = QuantumCircuit(n)

    for qubit in range(n):
        qc.h(qubit)

        # 控制旋转门
        for controlled_qubit in range(qubit + 1, n):
            angle = 2 * np.pi / (2 ** (controlled_qubit - qubit + 1))
            qc.cp(angle, controlled_qubit, qubit)

    # 反转量子比特顺序
    for i in range(n // 2):
        qc.swap(i, n - i - 1)

    return qc


# ================= 主程序 =================
n = 4
qc = QuantumCircuit(n, n)

# 输入态 |0001>（小端序：第 0 位翻转）
qc.x(0)

# 添加 QFT
qc.compose(qft_circuit_manual(n), inplace=True)

# 测量
qc.measure(range(n), range(n))

# 模拟运行
simulator = AerSimulator()
qc_t = transpile(qc, simulator)
job = simulator.run(qc_t, shots=1024)
counts = job.result().get_counts()

print("测量结果:", counts)
plot_histogram(counts)
plt.show()