import numpy as np
from scipy.linalg import expm, block_diag
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import copy
import matplotlib.pyplot as plt


# ==========================================
# 1. 积分计算，计算 Phi
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
        term = expm(A * t) @ B
        return term

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
# 2. 用户和 UAV 类定义
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
        self.assigned_bandwidth = 0.0  # 最终分配到的带宽 (Hz)

        # --- 物理参数 ---
        self.N0 = 1e-20  # 噪声功率谱密度 (W/Hz)
        self.cached_Prx = None  # 缓存变量

    # def get_received_power(self, uav_loc):
    #     """计算路损后的接收功率"""
    #     if self.cached_Prx is not None:
    #         return self.cached_Prx
    #     dist = np.linalg.norm(self.loc - uav_loc)
    #     dist = max(dist, 1.0)
    #     h0 = 1e-4
    #     alpha = 2.5
    #     channel_gain = h0 * (dist ** (-alpha))
    #     P_rx = self.P_tx * channel_gain
    #     self.cached_Prx = P_rx
    #     return P_rx

    # # 此函数在本代码逻辑中主要作为辅助，核心逻辑使用了解析公式
    # def _utility_func(self, B, P_rx, price):
    #     if B < 1e-6:
    #         return 1e9
    #     noise_power = self.N0 * B
    #     snr = P_rx / noise_power
    #     rate = B * np.log2(1 + snr)
    #     utility = self.theta * np.log(1 + rate) - price * B
    #     return -utility
    #
    # # 此函数在本代码逻辑中主要作为辅助，核心逻辑使用了解析公式
    # def optimize_bandwidth(self, uav, price):
    #     P_rx = self.get_received_power(uav.loc)
    #     safe_price = max(price, 1e-10)
    #     bnds = (1e3, uav.B_total)
    #     res = minimize_scalar(
    #         self._utility_func,
    #         bounds=bnds,
    #         args=(P_rx, safe_price),
    #         method='bounded',
    #         options={'xatol': 1e-4}
    #     )
    #     best_B = res.x
    #     max_util = -res.fun
    #     if max_util < 0:
    #         return 0.0, 0.0, 0.0
    #     noise = self.N0 * best_B
    #     final_snr = P_rx / noise
    #     return best_B, final_snr, max_util


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

        # --- 控制与稳定性参数 ---
        self.tau = 0.1  # 电机响应时间常数 (决定物理回路时延)
        self.rho = 0.9  # Lyapunov 收敛速率 (越小越严苛，要求能量衰减越快)

        # --- 动力学模型 dx = Ax + Bu ---
        # 状态 x = [位置误差(3), 速度误差(3)]^T
        self.dim_state = 6
        self.dim_input = 3
        # 带阻尼的二阶动力学模型
        mu = 10  # 空气阻力系数

        # 构造 A 矩阵 (系统内部演化)
        # [0   0   0    1   0   0] <- x位置 随 vx 变化
        # [0   0   0    0   1   0] <- y位置 随 vy 变化
        # [0   0   0    0   0   1] <- z位置 随 vz 变化
        # [0   0   0   -µ   0   0] <- vx 随自身衰减
        # [0   0   0    0  -µ   0] <- vy 随自身衰减
        # [0   0   0    0   0  -µ] <- vz 随自身衰减
        self.A = np.zeros((6, 6))
        # x 位置只受 vx 影响，y 受 vy 影响
        self.A[0:3, 3:6] = np.eye(3)  # d(位置)/dt = 速度，右上角块设为单位矩阵
        # -mu 越大，无人机停下来的越快，系统越“粘稠”
        self.A[3:6, 3:6] = -mu * np.eye(3)  # d(速度)/dt = -mu * 速度 (阻尼)

        # 构造 B 矩阵 (输入影响)
        # [0   0   0] <- 推力不直接改坐标
        # [0   0   0]
        # [0   0   0]
        # [1   0   0] <- 推力 ax 直接增加 vx 的导数
        # [0   1   0]
        # [0   0   1]
        self.B = np.zeros((6, 3))
        # 上半部分全为0：加速度指令 u 不能直接改变位置（位置是速度积分出来的，不能突变）
        # 下半部分为单位矩阵：加速度指令 u 直接改变速度的导数（即加速度）
        self.B[3:6, :] = np.eye(3)  # 输入 u 直接改变加速度 (速度的导数)，下半部分设为单位矩阵，np.eye 用于设置单位矩阵

        # --- Lyapunov 矩阵 ---
        self.P_lyap = np.eye(9)  # 状态权重矩阵 P (正定，9维是因为包含上一时刻控制量)
        self.Q_noise = 0.01 * np.eye(6)  # 过程噪声协方差 Q，针对状态噪声 sigma
        self.K = -0.5 * np.ones((3, 9))  # 反馈控制增益矩阵 K (简单 LQR 或 极点配置得到的增益，这里随机初始化一个稳定的)

    def update_user_distance_and_channel(self, user):
        """
        【关键简化假设】
        为了得到 Stackelberg 博弈的解析解，
        这里假设了底噪 N_pow 是固定的，不随带宽 B 变化。
        """
        dist = np.linalg.norm(self.loc - user.loc)
        dist = max(dist, 1.0)  # 避免除零

        # 路径损耗
        h0 = 1e-4  # 参考距离增益
        alpha = 2.5  # 路径损耗指数

        # --- 固定噪声模型 ---
        # 假设带宽很大，N0*B 近似为固定底噪 Power N_pow，可解释
        N_pow = 1e-13  # (W) 假设固定底噪

        channel_gain = h0 * (dist ** (-alpha))
        rx_power = user.P_tx * channel_gain
        snr = rx_power / N_pow  # 这里的 SNR 与带宽无关

        user.current_snr = snr
        user.H_i = np.log2(1 + snr)  # 香农公式中的对数项 (log_2(1+SNR))
        return dist

# ==========================================
# 3. 核心求解逻辑
# ==========================================

def solve_stackelberg_game(uav: UAV, users_list):
    """
    【求解器核心】
    结合控制理论稳定性约束和 Stackelberg 博弈优化。

    流程：
    1. 估算状态系统离散化参数 (\Phi, \Omega)。
    2. 计算为了维持稳定所需的最小传输成功率 (\Gamma_n)。
    3. 根据 \Gamma_n 推导时延预算 (T_budget)。
    4. 根据 T_budget 计算每个用户所需的最小带宽 (B_min)。
    5. 求解博弈：在满足 B_min 的前提下，寻找最优价格 p*。
    """

    print(f"\n--- 开始求解 UAV {uav.id} 的资源分配 ---")

    # ==========================================
    # Phase 1: 系统动力学与稳定性约束计算
    # ==========================================

    active_users = copy.copy(users_list)

    # 1. 时延估算
    # 假设旧指令执行占据了 40% 的周期 (简化处理，实际应基于历史数据)
    T_n = 0.4 * uav.s_n
    s_bar = uav.s_n - T_n  # 新指令执行时长

    # 2. 计算离散化系统参数 (利用 quad 积分)
    # Phi^0: 对应新指令 u(k) 的影响
    Phi_0 = get_discrete_matrices(uav.A, uav.B, 0, s_bar)
    # Phi^1: 对应旧指令 u(k-1) 的影响
    Phi_1 = get_discrete_matrices(uav.A, uav.B, s_bar, uav.s_n)
    # Omega: 状态转移部分 e^{As}
    Omega_k = expm(uav.A * uav.s_n)

    # 3. 构建增广状态矩阵 (9维状态: 6维物理状态 + 3维旧控制量)
    # 闭环演化矩阵 (通信成功时)
    Omega_d = np.zeros((9, 9))
    Omega_d[:6, :6] = Omega_k
    Omega_d[:6, 6:] = Phi_1

    Phi_d = np.zeros((9, 3))
    Phi_d[:6, :] = Phi_0
    Phi_d[6:, :] = np.eye(3)

    # 闭环系统矩阵：x(k+1) = (Omega_d + Phi_d * K) * x(k)
    Omega_cl = Omega_d + Phi_d @ uav.K

    # 开环演化矩阵 (通信失败时，没有新指令 u(k))
    Phi_total = get_discrete_matrices(uav.A, uav.B, 0, uav.s_n)
    Omega_op = np.zeros((9, 9))
    Omega_op[:6, :6] = Omega_k
    Omega_op[:6, 6:] = Phi_total
    Omega_op[6:, 6:] = np.eye(3)  # 保持旧控制量不变

    # 4. Lyapunov 能量计算
    # 随机生成一个当前状态 xi_k
    xi_k = np.array([5.0, 5.0, 2.0, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1]).T

    # 计算当前能量 V(k)
    V_curr = xi_k.T @ uav.P_lyap @ xi_k
    # 计算闭环下一时刻期望能量 V_cl(k+1)
    V_close = xi_k.T @ Omega_cl.T @ uav.P_lyap @ Omega_cl @ xi_k
    # 计算开环下一时刻期望能量 V_op(k+1)
    V_open = xi_k.T @ Omega_op.T @ uav.P_lyap @ Omega_op @ xi_k

    # 噪声能量项
    Q_aug = np.zeros((9, 9))
    Q_aug[:6, :6] = uav.Q_noise
    Tr_PQ = np.trace(uav.P_lyap @ Q_aug)

    # 5. 计算最小传输成功概率 Gamma_n
    # 依据：E[V(k+1)] <= rho * V(k) + Noise
    # 即：Gamma * V_close + (1-Gamma) * V_open <= rho * V_curr + Noise
    """ ① 理论上根据所有已知的参数，可以计算出\Gamma_n(xi_k) """
    denominator = V_open - V_close
    numerator = V_open - uav.rho * V_curr - Tr_PQ

    if denominator == 0:
        Gamma_n = 0.99
    else:
        Gamma_n = numerator / denominator

    # 概率必须在 [0, 1] 之间
    Gamma_n = np.clip(Gamma_n, 0.01, 0.99)

    print(f"  当前状态能量 V_curr: {V_curr:.2f}")
    print(f"  需要的最小传输成功率 Gamma_n: {Gamma_n:.4f}")

    # 6. 计算时延预算
    # 数学关系：如果要求成功率 Gamma，则平均时延需满足 E[T] <= (1-Gamma)*s_n
    """ ② 根据所计算出的$\Gamma_n(xi_k)$，可以进一步计算出$D_{req}(Gamma_n)$ """
    D_req = (1 - Gamma_n) * uav.s_n

    # 计算固定开销 (感知+计算+控制响应)
    T_sense = 0.02
    T_comp = 0.01
    T_control = 4 * uav.tau  # 物理回路响应
    T_fixed = T_sense + T_comp + T_control

    # 得到留给通信的时间预算 T_budget
    """ ③ 根据所计算出的$D_{req}(Gamma_n)$，可以进一步计算出$T_{budget}$，然后求解出$B_i^{min}$ """
    T_budget = D_req - T_fixed

    print(f"  最大允许平均时延 D_req: {D_req:.4f}s")
    print(f"  通信时延预算 T_budget: {T_budget:.4f}s")

    # 如果预算为负，说明物理和计算太慢了，光是固定开销就超过了稳定性要求
    if T_budget <= 0:
        print("  [警告] 系统物理时延已超过稳定性约束容限！")
        T_budget = 0.001  # 强制给一个小值以防除零，但结果可能无效

    # ==========================================
    # Phase 2: 迭代筛选用户与求解博弈
    # ==========================================

    while True:
        if not active_users:
            print("  [失败] 没有用户可以被服务。")
            return None, None

        # 2.1 更新统计量与带宽约束
        Theta_sum = 0.0  # 所有用户的优先级之和
        Inv_H_sum = 0.0  # 所有用户的信道条件倒数之和

        for user in active_users:
            uav.update_user_distance_and_channel(user)

            # 计算最小带宽需求 B_min
            # 传输时间 T_tran = S / (B * H_i) <= T_budget
            # 所以 B >= S / (T_budget * H_i)
            if user.H_i <= 1e-6:
                user.B_min = 1e9  # 信道太差，需要无穷大带宽
            else:
                user.B_min = user.S / (T_budget * user.H_i)

            Theta_sum += user.theta
            Inv_H_sum += (1.0 / user.H_i)

        # 2.2 计算价格边界
        # 价格必须足够低，才能保证用户购买的带宽 B_i >= B_min
        # 用户策略：B_i = theta/p - 1/H
        # 约束：theta/p - 1/H >= B_min  =>  p <= theta / (B_min + 1/H)
        """ ④ 将 Follower 的最优解代入到 Leader 的效用函数里面 """
        p_bars = []
        for user in active_users:
            val = user.theta / (user.B_min + 1.0 / user.H_i)
            p_bars.append(val)

        # 稳定性约束上界 (必须满足所有用户)
        """ ⑤ 根据所计算出的$B_i^{min}$，可以求解出$\overline {p_n}$，进而最后求解出 Leader 的最优解 """
        p_max = min(p_bars)

        # 容量约束下界：所有用户购买带宽之和 <= 总带宽
        # Sum(theta/p - 1/H) <= B_total  =>  p >= Sum(theta) / (B_total + Sum(1/H))
        p_min = Theta_sum / (uav.B_total + Inv_H_sum)

        print(f"  用户数: {len(active_users)}")
        print(f"  价格下界 (容量) p_min: {p_min:.6f}")
        print(f"  价格上界 (稳定) p_max: {p_max:.6f}")

        # 2.3 检查可行性
        if p_min > p_max:
            # 如果最低价都比最高价高，说明没有可行解（带宽不够分）
            print("  -> 冲突: 资源不足。剔除优先级最低的用户...")
            # 剔除策略：去掉 theta 最小的用户
            active_users.sort(key=lambda x: x.theta)
            removed_user = active_users.pop(0)
            print(f"     移除用户 ID {removed_user.id} (Theta: {removed_user.theta})")
            continue  # 进入下一轮循环，重新计算
        else:
            # 2.4 求解最优价格
            # Leader 利润最大化的一阶导数零点：
            # p_opt = sqrt( cost * Sum(theta) / Sum(1/H) )
            p_opt_uncons = np.sqrt((uav.c_n * Theta_sum) / Inv_H_sum)
            print(f"  无约束最优价格 p_opt: {p_opt_uncons:.6f}")

            # 考虑边界约束
            p_final = 0.0
            if p_opt_uncons < p_min:
                p_final = p_min
                print("  -> 激活容量约束 (带宽被买光了)")
            elif p_opt_uncons > p_max:
                p_final = p_max
                print("  -> 激活稳定性约束 (为了救人不得不降价)")
            else:
                p_final = p_opt_uncons
                print("  -> 采用无约束最优")

            if p_final <= uav.c_n:
                print("  [警告] 价格低于成本，亏本服务。")

            # 2.5 计算最终带宽分配 (Follower Response)
            results = []
            total_bw_used = 0.0
            for user in active_users:
                # 代入最优价格计算带宽
                # B_i = theta_i / p - 1 / H_i
                B_i = (user.theta / p_final) - (1.0 / user.H_i)

                # 理论上应该满足 B_i >= B_min，这里做数值修正
                B_i = max(B_i, user.B_min)

                user.assigned_bandwidth = B_i
                total_bw_used += B_i

                results.append({
                    "user_id": user.id,
                    "B_assigned": B_i,
                    "B_min_req": user.B_min,
                    "SNR_dB": 10 * np.log10(user.current_snr),
                    "Theta": user.theta
                })

            print(f"  --- 分配完成 ---")
            print(f"  总带宽使用: {total_bw_used:.2f} / {uav.B_total:.2f}")
            print(f"  最终定价: {p_final:.6f}")
            return p_final, results

# ==========================================
# 3. 可视化函数 (优化版：堆叠柱状图)
# ==========================================

def plot_single_uav_results(uav, allocation_results):
    if not allocation_results:
        print("无数据可绘图")
        return

    # 提取数据并转为 numpy 数组方便计算
    uids = [res['user_id'] for res in allocation_results]
    thetas = [res['Theta'] for res in allocation_results]
    snrs = [res['SNR_dB'] for res in allocation_results]

    b_allocs = np.array([res['B_assigned'] / 1e6 for res in allocation_results])  # MHz
    b_mins = np.array([res['B_min_req'] / 1e6 for res in allocation_results])  # MHz

    # 计算“额外盈余带宽” (上层绿色部分)
    b_extras = b_allocs - b_mins

    # --- 图 1：用户属性散点图 (保持不变) ---
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(snrs, thetas, s=[t * 15 for t in thetas], c=b_allocs, cmap='viridis', alpha=0.8,
                          edgecolors='k')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Priority Weight (Theta)', fontsize=12)
    plt.title(f'User Attributes & Allocation (UAV {uav.id})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Total Allocated Bandwidth (MHz)')
    for i, txt in enumerate(uids):
        plt.annotate(f"U{txt}", (snrs[i], thetas[i]), xytext=(5, 5), textcoords='offset points', fontsize=10,
                     fontweight='bold')
    plt.tight_layout()
    plt.show()

    '''Theta 越大（Y轴越高）且 SNR 越好（X轴越靠右）的用户，颜色越亮（分到的带宽越多）。这完美验证了 Stackelberg 博弈的“按需分配”特性。'''

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
    plt.errorbar(x, b_mins, xerr=width / 2, fmt='none', ecolor='blue', lw=2, capsize=5, label='Min Requirement (Stability)')

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

    '''绿色的柱子（实际分配）永远高于或等于红色的阴影/横线（最小需求）'''

# ==========================================
# 4. 主程序运行示例
# ==========================================

if __name__ == "__main__":
    np.random.seed(10)

    # 4.1 初始化 UAV
    # 带宽 20MHz, 成本 1e-7, 周期 0.5s
    uav = UAV(nid=1, loc=[0, 0, 100], vel=[0, 0, 0],
              B_total=20e6, cost_per_bw=1e-7, target_s_n=0.5)

    # 4.2 初始化用户集合
    users = []
    for i in range(5):
        # 随机分布用户
        loc = [np.random.uniform(-200, 200), np.random.uniform(-200, 200), 0]
        # 速度
        vel = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0]
        # 数据量 5KB
        S = 5 * 8 * 1024
        # 随机优先级
        theta = np.random.uniform(10, 50)
        # 发射功率
        P_tx = 0.1

        u = User(uid=i + 1, loc=loc, vel=vel, data_size_S=S, priority_weight_theta=theta, max_power_P=P_tx)
        users.append(u)

    # 4.3 运行求解
    price, allocation = solve_stackelberg_game(uav, users)

    # 4.4 打印结果表和绘图
    if allocation:
        print("\n详细分配表:")
        print(
            f"{'User ID':<10} | {'Theta':<10} | {'SNR (dB)':<10} | {'B_min (Hz)':<15} | {'B_alloc (Hz)':<15} | {'Satisfied?'}")
        print("-" * 80)
        for res in allocation:
            # 检查是否满足最小需求 (允许微小误差)
            sat = "YES" if res['B_assigned'] >= res['B_min_req'] - 1.0 else "NO"
            print(
                f"{res['user_id']:<10} | {res['Theta']:<10.2f} | {res['SNR_dB']:<10.2f} | {res['B_min_req']:<15.2f} | {res['B_assigned']:<15.2f} | {sat}")
        plot_single_uav_results(uav, allocation)
