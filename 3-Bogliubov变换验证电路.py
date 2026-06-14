import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator
# ================================
# 1. 2-qubit 索引翻转（和你 _bitrev2 等价）
# ================================
def bitrev2(i: int) -> int:
    return ((i & 1) << 1) | ((i >> 1) & 1)

def lowfirst_to_highfirst(U_low):
    U = np.zeros((4,4), dtype=complex)
    for i in range(4):
        for j in range(4):
            U[bitrev2(i), bitrev2(j)] = U_low[i, j]
    return U

# ================================
# 2. B_k^n 矩阵（论文公式 14）
# ================================
def Bk_matrix_lowfirst(lam, k, n):
    kk = 2 * np.pi * k / n
    num = lam - np.cos(kk)
    den = np.sqrt((lam - np.cos(kk))**2 + np.sin(kk)**2)
    ratio = float(np.clip(num / den, -1.0, 1.0))
    theta = np.arccos(ratio)

    c = np.cos(theta / 2)
    s = np.sin(theta / 2)

    B = np.eye(4, dtype=complex)
    B[0, 0] = c
    B[0, 3] = 1j * s   # ✅ 与论文一致（+i）
    B[3, 0] = 1j * s
    B[3, 3] = c
    return B

# ================================
# 3. 构造 Qiskit 门
# ================================
def Bk_gate_qiskit(lam, k, n):
    U_low = Bk_matrix_lowfirst(lam, k, n)
    U_high = lowfirst_to_highfirst(U_low)
    return U_high

# ================================
# 4. 验证（态矢量级）
# ================================
def verify_Bk_qiskit(lam=1.6, k=1, n=4):
    U = Bk_gate_qiskit(lam, k, n)

    # 理论演化
    init = np.array([1,0,0,0], dtype=complex)
    exact = U @ init

    # Qiskit 模拟
    qc = QuantumCircuit(2)
    qc.unitary(U, [0, 1])  # qubit0 = q0, qubit1 = q1

    sim = AerSimulator()
    qc_sv = Statevector.from_label("00").evolve(qc)

    fid = state_fidelity(qc_sv, exact)

    print(f"λ={lam}, k={k}, θ={np.arccos((lam-np.cos(2*np.pi*k/n))/np.sqrt((lam-np.cos(2*np.pi*k/n))**2+np.sin(2*np.pi*k/n)**2)):.6f}")
    print("保真度:", fid)

    return fid

# ================================
# 5. 运行
# ================================
if __name__ == "__main__":
    verify_Bk_qiskit(lam=1.6, k=1, n=4)