import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, SparsePauliOp, DensityMatrix
from qiskit.circuit.library import QFT
import matplotlib.pyplot as plt
from tqdm import tqdm

# ======================
# 1. 模型参数
# ======================
n_qubits = 4  # 4 量子比特
lambda_range = np.linspace(0.01, 2.0, 50)  # λ 范围
beta_range = np.linspace(0.01, 20.0, 50)   # β 范围（逆温度）

# ======================
# 2. 构建哈密顿量（TFIM）
# ======================
def build_tfim_hamiltonian(n, lam):
    """构建横场伊辛模型哈密顿量"""
    paulis = []
    coeffs = []
    
    # 横向场项：-λ * Σ σ_i^x
    for i in range(n):
        pauli_str = ['I'] * n
        pauli_str[i] = 'X'
        paulis.append(''.join(pauli_str))
        coeffs.append(-lam)
    
    # 纵向耦合项：-Σ σ_i^z σ_{i+1}^z
    for i in range(n - 1):
        pauli_str = ['I'] * n
        pauli_str[i] = 'Z'
        pauli_str[i + 1] = 'Z'
        paulis.append(''.join(pauli_str))
        coeffs.append(-1.0)
    
    return SparsePauliOp(paulis, coeffs)

# ======================
# 3. 计算热态期望值（密度矩阵方法）
# ======================
def compute_thermal_expectation(n, lam, beta):
    from qiskit.quantum_info import Operator
    import numpy as np

    # 构建哈密顿量
    H = build_tfim_hamiltonian(n, lam)

    # 转成 Operator，再算本征值
    op = Operator(H)
    eigvals, eigvecs = np.linalg.eigh(op.data)

    # 配分函数
    Z = np.sum(np.exp(-beta * eigvals))

    # 横向磁化算符 <σ^x>
    sigma_x = SparsePauliOp(
        ['X' + 'I' * (n - 1),
         'I' + 'X' + 'I' * (n - 2),
         'I' * 2 + 'X' + 'I' * (n - 3),
         'I' * 3 + 'X'],
        [1.0] * n
    )

    expectation = 0.0
    for i in range(len(eigvals)):
        p_i = np.exp(-beta * eigvals[i]) / Z
        exp_val_i = eigvecs[:, i].conj().T @ sigma_x.to_matrix() @ eigvecs[:, i]
        expectation += p_i * exp_val_i.real

    return expectation / n
# ======================
# 4. 生成相图数据
# ======================
heatmap_data = np.zeros((len(beta_range), len(lambda_range)))

for i, beta in enumerate(tqdm(beta_range, desc="Computing β rows")):
    for j, lam in enumerate(lambda_range):
        heatmap_data[i, j] = compute_thermal_expectation(n_qubits, lam, beta)

# ======================
# 5. 绘制热力图（Figure 2 风格）
# ======================
# ======================
# 5. 绘制热力图（Figure 2 风格 · viridis）
# ======================
plt.figure(figsize=(8, 6))

im = plt.imshow(
    heatmap_data,
    extent=[lambda_range.min(), lambda_range.max(), beta_range.min(), beta_range.max()],
    cmap='viridis',        # ✅ 改回 viridis
    aspect='auto',
    origin='lower'
)

cbar = plt.colorbar(im)
cbar.set_label(r'$\langle \sigma^z \rangle$', fontsize=14)

plt.xlabel(r'$\lambda$ (Transverse field)', fontsize=14)
plt.ylabel(r'$\beta$ (Inverse Temperature)', fontsize=14)
plt.title(r'Transverse Magnetization of $n=4$ 1D Transverse-Field Ising Model', fontsize=16)

plt.grid(False)

# 白色等高线，突出临界区
plt.contour(
    lambda_range,
    beta_range,
    heatmap_data,
    levels=20,
    colors='white',
    alpha=0.35,
    linewidths=0.8
)

plt.tight_layout()
plt.show()