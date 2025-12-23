import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
import matplotlib.pyplot as plt
import copy


# ==========================================
# 1. 积分计算 (保持精确的数值积分)，与理论推导完全一致
# ==========================================
def get_discrete_matrices(A, B, t_start, t_end):
    """
    【功能】计算线性系统离散化过程中的输入积分项。

    【数学原理】
    对于连续系统 dx = Ax + Bu，在采样周期内的离散化形式为：
    x(k+1) = e^{AT}x(k) + (\int e^{A\tau}B d\tau)u(k)
    此函数计算的就是积分项 \Phi = \int_{t_start}^{t_end} e^{At} B dt

    【实现方式】
    使用 scipy.integrate.quad 对矩阵的每一个元素分别进行数值积分。
    虽然速度较慢，但数学含义直观且精确。
    """
    # 积分时间长度 dt
    dt = t_end - t_start
    if dt <= 0:
        return np.zeros(B.shape)

    n = A.shape[0]  # 状态维数 (6维: 3位置+3速度)
    m = B.shape[1]  # 输入维数 (3维: 3加速度)

    # 初始化结果矩阵 Phi
    Phi = np.zeros((n, m))

    # 定义被积函数矩阵: F(t) = e^{At} * B
    def integrand_matrix(t):
        return expm(A * t) @ B

    # 遍历结果矩阵的每一行 i 和每一列 j，逐个元素积分
    for i in range(n):
        for j in range(m):
            # 定义标量被积函数
            def integrand_element(t):
                mat_val = integrand_matrix(t)
                return mat_val[i, j]

            # 执行数值积分 quad 只能积标量，所以需要对 Phi 矩阵的每一个元素 (i, j) 分别积分
            # 执行数值积分，区间 [t_start, t_end]。val 是积分值，error 是估计误差
            val, error = quad(integrand_element, t_start, t_end)
            Phi[i, j] = val

    return Phi


# ==========================================
# 2. 用户与 UAV 类定义
# ==========================================
class User:
    """
    【Follower】
    代表地面幸存者。在博弈中，根据 UAV 的定价调整自己购买的带宽。
    """
    def __init__(self, uid, loc, vel, data_size_S, priority_weight_theta, max_power_P):
        self.id = uid  # 用户唯一ID
        self.loc = np.array(loc, dtype=float)  # 位置坐标 [x, y, 0] (米)
        self.S = data_size_S  # 待传输数据包大小 (bits)
        self.vel = np.array(vel, dtype=float)  # [新增] 用户速度，例如人或车辆的移动
        self.theta = priority_weight_theta  # 信息优先级权重 (theta)，越大越舍得花钱买带宽
        self.P_tx = max_power_P  # 最大发射功率 (Watts)

        # 运行时变量
        self.H_i = 0.0
        self.current_snr = 0.0  # 信噪比
        self.B_min = 0.0  # 最小需求带宽
        self.assigned_bandwidth = 0.0  # 最终分配到的带宽 (Hz)


class UAV:
    """
    【Leader】
    代表搜救无人机。
    1. 负责维护物理控制系统的稳定性 (通过 Lyapunov 理论)。
    2. 负责制定带宽价格 p，以最大化收益并满足约束。
    """

    def __init__(self, nid, loc, vel, B_total, cost_per_bw, target_s_n):
        self.id = nid  # UAV ID
        self.loc = np.array(loc, dtype=float)  # 位置 [x, y, z]
        self.vel = np.array(vel, dtype=float)  # 速度 [vx, vy, vz]
        self.B_total = B_total  # 总可用带宽 (Hz)
        self.c_n = cost_per_bw  # 单位带宽成本 (cost)
        self.s_n = target_s_n  # 采样周期 (秒)，控制指令更新间隔

        # 控制理论参数
        self.tau = 0.1  # 电机响应时间常数 (决定物理回路时延)
        self.rho = 0.9  # Lyapunov 收敛速率 (越小越严苛，要求能量衰减越快)
        self.u_last = np.zeros(3)  # 上一时刻控制输入

        # 二阶动力学 A, B
        mu = 10

        self.A = np.zeros((6, 6))
        self.A[0:3, 3:6] = np.eye(3)
        self.A[3:6, 3:6] = -mu * np.eye(3)

        self.B = np.zeros((6, 3))
        self.B[3:6, :] = np.eye(3)

        self.P_lyap = np.eye(9)
        self.Q_noise = 0.01 * np.eye(6)
        self.K = -0.5 * np.ones((3, 9))

        self.T_budget = 0.1  # 初始占位

    # def update_time_budget(self, active_users):
    #     """
    #     根据 UAV 与当前所有活跃用户的实时位置/速度误差，计算最严苛的 T_budget
    #     """
    #     if not active_users:
    #         self.T_budget = 0.4 * self.s_n
    #         return
    #
    #     # 1. 预计算离散化矩阵
    #     s_bar = 0.6 * self.s_n  # 假设新指令执行占 60%，简化
    #     Phi_0 = get_discrete_matrices(self.A, self.B, 0, s_bar)
    #     Phi_1 = get_discrete_matrices(self.A, self.B, s_bar, self.s_n)
    #     Omega_k = expm(self.A * self.s_n)
    #     Phi_total = get_discrete_matrices(self.A, self.B, 0, self.s_n)
    #
    #     # 闭环与开环增广矩阵 (9x9)
    #     Omega_cl = np.zeros((9, 9))
    #     Omega_cl[:6, :6] = Omega_k
    #     Omega_cl[:6, 6:] = Phi_1
    #     Phi_d = np.zeros((9, 3))
    #     Phi_d[:6, :] = Phi_0
    #     Phi_d[6:, :] = np.eye(3)
    #     Omega_cl = Omega_cl + Phi_d @ self.K
    #
    #     Omega_op = np.zeros((9, 9))
    #     Omega_op[:6, :6] = Omega_k
    #     Omega_op[:6, 6:] = Phi_total
    #     Omega_op[6:, 6:] = np.eye(3)
    #
    #     Q_aug = np.zeros((9, 9))
    #     Q_aug[:6, :6] = self.Q_noise
    #     Tr_PQ = np.trace(self.P_lyap @ Q_aug)
    #
    #     # 2. 遍历用户寻找最大 Gamma (最差物理情况)
    #     max_Gamma = 0.01
    #     for user in active_users:
    #         # 状态误差向量 xi = [pos_err, vel_err, u_last]
    #         xi = np.concatenate([self.loc - user.loc, self.vel - user.vel, self.u_last])
    #         V_curr = xi.T @ self.P_lyap @ xi
    #         V_close = xi.T @ Omega_cl.T @ self.P_lyap @ Omega_cl @ xi
    #         V_open = xi.T @ Omega_op.T @ self.P_lyap @ Omega_op @ xi
    #
    #         denom = V_open - V_close
    #         num = V_open - self.rho * V_curr - Tr_PQ
    #         Gamma_i = np.clip(num / denom, 0.01, 0.99) if denom != 0 else 0.99
    #         max_Gamma = max(max_Gamma, Gamma_i)
    #
    #     # 3. 转换回时间预算
    #     D_req = (1 - max_Gamma) * self.s_n
    #     T_fixed = 0.03 + 4 * self.tau  # 感知+计算+物理回路
    #     self.T_budget = max(0.001, D_req - T_fixed)

    def update_time_budget(self, active_users):
        """
        按照文字逻辑严密求解：
        1. 计算固定时延 T_fixed (感知 + 计算基础 + 控制响应)
        2. 基于 T_fixed 确定离散化切换点 s_bar
        3. 求解 Lyapunov 稳定性要求的 Gamma
        4. 推导通信时间预算 T_budget
        """
        if not active_users:
            self.T_budget = 0.5 * self.s_n
            return

        # --- 步骤 1: 计算固定时延 T_fixed ---
        # 这里的参数应与你文字定义一致
        t_sense = 0.01  # T_i^sense (假设 10ms)
        t_base = 0.005  # T_n^base (假设 5ms)
        f_0 = 1000  # 每 bit 所需 CPU 周期
        f_n = 1e9  # UAV 计算能力 (1 GHz)

        # 找到最严苛（固定时延最大）的用户
        max_t_fixed = 0
        for user in active_users:
            # T_fixed = T_sense + T_comp_base + T_control
            t_comp = t_base + (user.S * f_0) / f_n
            t_control = 4 * self.tau
            t_fixed_i = t_sense + t_comp + t_control
            if t_fixed_i > max_t_fixed:
                max_t_fixed = t_fixed_i

        # --- 步骤 2: 确定离散化矩阵切换点 ---
        # 根据公式 s_n = s_bar + T_n，在理想闭环情况下 T_n = T_fixed
        # 所以 s_bar (新指令执行时间) = s_n - T_fixed
        s_bar = max(0.001, self.s_n - max_t_fixed)

        # 预计算离散化矩阵
        Phi_0 = get_discrete_matrices(self.A, self.B, 0, s_bar)  # 执行新指令 [0, s_bar]
        Phi_1 = get_discrete_matrices(self.A, self.B, s_bar, self.s_n)  # 执行旧指令 [s_bar, s_n]
        Omega_k = expm(self.A * self.s_n)
        Phi_total = get_discrete_matrices(self.A, self.B, 0, self.s_n)

        # 构造广义闭环矩阵 Omega_cl (9x9)
        # xi(k+1) = [Omega_k, Phi_1; 0, 0] * xi(k) + [Phi_0; I] * u(k)
        # 加入 u(k) = K * xi(k)
        Omega_cl_base = np.zeros((9, 9))
        Omega_cl_base[:6, :6] = Omega_k
        Omega_cl_base[:6, 6:] = Phi_1

        Phi_d = np.zeros((9, 3))
        Phi_d[:6, :] = Phi_0
        Phi_d[6:, :] = np.eye(3)

        Omega_cl = Omega_cl_base + Phi_d @ self.K

        # 构造广义开环矩阵 Omega_op (9x9)
        # 没有新指令，系统按惯性演变，旧指令 u(k-1) 作用全周期
        Omega_op = np.zeros((9, 9))
        Omega_op[:6, :6] = Omega_k
        Omega_op[:6, 6:] = Phi_total
        Omega_op[6:, 6:] = np.eye(3)

        # 噪声项
        Q_aug = np.zeros((9, 9))
        Q_aug[:6, :6] = self.Q_noise
        Tr_PQ = np.trace(self.P_lyap @ Q_aug)

        # --- 步骤 3: 遍历用户寻找最大 Gamma (最差物理情况) ---
        max_Gamma = 0.01
        for user in active_users:
            # 状态误差向量 xi = [pos_err, vel_err, u_last]
            xi = np.concatenate([self.loc - user.loc, self.vel - user.vel, self.u_last])
            V_curr = xi.T @ self.P_lyap @ xi
            V_close = xi.T @ Omega_cl.T @ self.P_lyap @ Omega_cl @ xi
            V_open = xi.T @ Omega_op.T @ self.P_lyap @ Omega_op @ xi

            denom = V_open - V_close
            num = V_open - self.rho * V_curr - Tr_PQ

            # 防止分母为0或计算出不合理的概率
            if denom > 1e-9:
                Gamma_i = num / denom
            else:
                Gamma_i = 0.99

            max_Gamma = max(max_Gamma, np.clip(Gamma_i, 0.01, 0.99))

        # --- 步骤 4: 转换回通信时间预算 T_budget ---
        # 根据马尔可夫不等式推导：D_req = (1 - Gamma) * s_n
        D_req = (1 - max_Gamma) * self.s_n

        # T_budget 是留给传输时延 (S/R) 的部分
        # T_budget = D_req - T_fixed
        self.T_budget = max(0.001, D_req - max_t_fixed)

    def update_channel(self, user):
        dist = max(np.linalg.norm(self.loc - user.loc), 1.0)
        h0, alpha, N_pow = 1e-4, 2.5, 1e-13
        gain = h0 * (dist ** (-alpha))
        user.current_snr = (user.P_tx * gain) / N_pow
        user.H_i = np.log2(1 + user.current_snr)


# ==========================================
# 3. 博弈求解核心
# ==========================================
def solve_stackelberg_game(uav, users_list):
    active_users = copy.copy(users_list)
    print(f"\n--- 开始博弈求解 (UAV {uav.id}) ---")

    while True:
        if not active_users:
            return None, None

        # A. 动态计算当前活跃用户群体下的物理约束
        uav.update_time_budget(active_users)

        # B. 更新每个用户的最小带宽 B_min (基于当前的 T_budget)
        Theta_sum, Inv_H_sum = 0.0, 0.0
        for user in active_users:
            uav.update_channel(user)
            user.B_min = user.S / (uav.T_budget * user.H_i)
            Theta_sum += user.theta
            Inv_H_sum += (1.0 / user.H_i)

        # C. 确定价格可行域
        p_bars = [u.theta / (u.B_min + 1.0 / u.H_i) for u in active_users]  # 最大值
        p_max = min(p_bars)  # 稳定性上限，通过控制约束转变求解
        p_min = Theta_sum / (uav.B_total + Inv_H_sum)  # 容量下限

        if p_min > p_max:
            # 资源冲突，移除优先级最小的用户
            active_users.sort(key=lambda x: x.theta)
            removed = active_users.pop(0)
            print(f"  [资源不足] 移除用户 U{removed.id}, 重新计算...")
            continue

        # D. 求解最优价格并映射到可行域
        p_opt_uncons = np.sqrt((uav.c_n * Theta_sum) / Inv_H_sum)  # 价格的理论求解
        p_final = np.clip(p_opt_uncons, p_min, p_max)

        # E. 计算最终结果
        results = []
        for user in active_users:
            B_i = (user.theta / p_final) - (1.0 / user.H_i)  # 带宽的理论值
            user.assigned_bandwidth = max(B_i, user.B_min)  # 满足稳定性
            results.append({
                "user_id": user.id,
                "B_assigned": user.assigned_bandwidth,
                "B_min_req": user.B_min,
                "SNR_dB": 10 * np.log10(user.current_snr),
                "Theta": user.theta
            })

        print(f"  收敛! 价格: {p_final:.2e}, 时延预算: {uav.T_budget:.4f}s")
        return p_final, results


# ==========================================
# 4. 绘图函数
# ==========================================
def plot_results(uav, allocation_results):
    if not allocation_results: return

    uids = [res['user_id'] for res in allocation_results]
    thetas = [res['Theta'] for res in allocation_results]
    snrs = [res['SNR_dB'] for res in allocation_results]
    b_allocs = np.array([res['B_assigned'] / 1e6 for res in allocation_results])
    b_mins = np.array([res['B_min_req'] / 1e6 for res in allocation_results])
    b_extras = b_allocs - b_mins

    # --- 图 1：用户属性散点图 ---
    # 圆圈越大：意味着该用户是“高价值”用户（例如正处于紧急状态的幸存者或传输关键指令的终端）。在博弈逻辑中，优先级越大的用户，其效用函数对带宽越敏感，越舍得花钱购买带宽。
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(snrs, thetas, s=[t * 15 for t in thetas], c=b_allocs, cmap='viridis', alpha=0.8,
                          edgecolors='k')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Priority Weight (Theta)')
    plt.title(f'User Attributes & Allocation (UAV {uav.id})')
    plt.grid(True, linestyle='--', alpha=0.6)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Total Allocated Bandwidth (MHz)')
    for i, txt in enumerate(uids):
        plt.annotate(f"U{txt}", (snrs[i], thetas[i]), xytext=(5, 5), textcoords='offset points')
    plt.tight_layout()
    plt.show()

    # --- 图 2：带宽分配堆叠柱状图 (颜色分明) ---
    plt.figure(figsize=(7, 5))

    x = np.arange(len(uids))
    width = 0.6

    # 1. 绘制底层：最小需求 (红色)
    p1 = plt.bar(x, b_mins, width, color='#d62728', label='Min Requirement (Base)', alpha=0.7, edgecolor='black')

    # 2. 绘制上层：额外盈余 (绿色)，底部垫在 b_mins 之上
    p2 = plt.bar(x, b_extras, width, bottom=b_mins, color='#2ca02c', label='Extra Allocation (Surplus)', alpha=0.5,
                 edgecolor='black')

    # 用蓝色阶梯线表示由于稳定性要求的"生死线"
    plt.errorbar(x, b_mins, xerr=width / 2, fmt='none', ecolor='blue', lw=2, capsize=5,
                 label='Min Requirement (Stability)')

    plt.xlabel('User ID', fontsize=12)
    plt.ylabel('Bandwidth (MHz)', fontsize=12)
    plt.title(f'Bandwidth Allocation Breakdown (UAV {uav.id})', fontsize=14)
    plt.xticks(x, [f"U{i}" for i in uids])
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 在柱子顶端标注“总带宽”数值
    for i in range(len(x)):
        total_height = b_allocs[i]
        plt.text(x[i], total_height, f'{total_height:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 在红色柱子中间标注最小需求数值
    for rect in p1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., height / 2.,
                 f'{height:.4f}', ha='center', va='center', fontweight='bold', color='white', fontsize=8)

    plt.tight_layout()
    plt.show()


# ==========================================
# 5. 主程序运行
# ==========================================
if __name__ == "__main__":
    np.random.seed(0)

    # 初始化单架 UAV
    my_uav = UAV(nid=1, loc=[0, 0, 100], vel=[2, 2, 0],
                 B_total=20e6, cost_per_bw=1e-7, target_s_n=0.5)

    # 初始化 8 个用户 (带位置和速度)
    my_users = []
    for i in range(8):
        loc = [np.random.uniform(-100, 100), np.random.uniform(-100, 100), 0]
        vel = [np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5), 0]
        theta = np.random.uniform(20, 80)
        u = User(uid=i + 1, loc=loc, vel=vel, data_size_S=5 * 1024 * 8, priority_weight_theta=theta, max_power_P=0.2)
        my_users.append(u)

    # 博弈求解
    final_p, alloc_data = solve_stackelberg_game(my_uav, my_users)

    # 绘图
    plot_results(my_uav, alloc_data)
