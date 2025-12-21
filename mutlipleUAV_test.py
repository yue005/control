import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
import matplotlib.pyplot as plt

# ==========================================
# 1. 积分计算
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

            # 执行数值积分 quad，quad 只能积标量，所以需要对 Phi 矩阵的每一个元素 (i, j) 分别积分
            # 执行数值积分，区间 [t_start, t_end]。val 是积分值，error 是估计误差
            val, error = quad(integrand_element, t_start, t_end)
            Phi[i, j] = val

    return Phi
# def get_discrete_matrices(A, B, t_start, t_end):
#     """
#     计算积分项 \Phi = \int_{t_start}^{t_end} e^{At} B dt
#     """
#     dt = t_end - t_start
#     if dt <= 0:
#         return np.zeros(B.shape)
#     n, m = A.shape[0], B.shape[1]
#
#     # 为了演示速度，这里改回了 Van Loan 方法 (比 quad 快很多，且适合动态更新)
#     # 如果你坚持要用 quad，可以替换回之前的代码，但会显著变慢
#     aug = np.zeros((n + m, n + m))
#     aug[:n, :n] = A
#     aug[:n, n:] = B
#     exp_aug = expm(aug * dt)
#     return exp_aug[:n, n:]


# ==========================================
# 2. 类定义
# ==========================================

class User:
    def __init__(self, uid, loc, vel, data_size_S, priority_weight_theta, max_power_P):
        self.id = uid
        self.loc = np.array(loc, dtype=float)
        self.vel = np.array(vel, dtype=float)  # 新增：用户速度
        self.S = data_size_S
        self.theta = priority_weight_theta
        self.P_tx = max_power_P

        self.connected_uav_id = -1
        self.assigned_bandwidth = 0.0
        self.current_utility = -np.inf

        self.N_pow = 1e-13
        self.cached_H = {}

    def get_channel_H(self, uav):
        if uav.id in self.cached_H:
            return self.cached_H[uav.id]
        dist = np.linalg.norm(self.loc - uav.loc)
        dist = max(dist, 1.0)
        h0 = 1e-4
        alpha = 2.5
        channel_gain = h0 * (dist ** (-alpha))
        snr = (self.P_tx * channel_gain) / self.N_pow
        H_i = np.log2(1 + snr)
        self.cached_H[uav.id] = H_i
        return H_i

    def calculate_potential_utility(self, uav, price):
        H_i = self.get_channel_H(uav)
        safe_price = max(price, 1e-10)
        B_opt = (self.theta / safe_price) - (1.0 / H_i)

        if B_opt <= 0:
            return -1e9, 0.0
        utility = self.theta * np.log(1 + B_opt * H_i) - safe_price * B_opt
        return utility, B_opt


class UAV:
    def __init__(self, nid, loc, vel, B_total, cost_per_bw, target_s_n):
        self.id = nid
        self.loc = np.array(loc, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.B_total = B_total
        self.c_n = cost_per_bw
        self.s_n = target_s_n

        self.price = 1e-5
        self.connected_users = []
        self.history_prices = []

        self.tau = 0.1
        self.rho = 0.9

        mu = 0.5
        self.A = np.zeros((6, 6))
        self.A[0:3, 3:6] = np.eye(3)
        self.A[3:6, 3:6] = -mu * np.eye(3)
        self.B = np.zeros((6, 3))
        self.B[3:6, :] = np.eye(3)

        self.P_lyap = np.eye(9)
        self.Q_noise = 0.01 * np.eye(6)
        self.K = -0.5 * np.ones((3, 9))

        # 上一时刻的控制输入 (假设初始为0或悬停推力)
        self.u_last = np.zeros(3)

        # 预计算固定矩阵
        self.Omega_k = expm(self.A * self.s_n)
        self.Phi_total = get_discrete_matrices(self.A, self.B, 0, self.s_n)

        # 初始默认预算 (还没用户时)
        self.T_budget = 0.2

    def update_time_budget(self):
        """
        【关键修改】动态计算 T_budget
        根据当前连接的用户的实际位置和速度，构建真实的状态误差向量 xi，
        计算每个用户需要的 Gamma，取最大值（最坏情况）作为系统要求。
        """
        if not self.connected_users:
            # 如果没有用户，使用宽松的默认值
            self.T_budget = 0.4 * self.s_n
            return

        # 1. 估算系统矩阵 (依赖于 T_n，这里先假设 T_n 占 40% 周期来计算矩阵参数)
        # 注意：严格来说这是一个不动点问题，这里做简化解耦处理
        T_n_est = 0.4 * self.s_n
        s_bar = self.s_n - T_n_est

        Phi_0 = get_discrete_matrices(self.A, self.B, 0, s_bar)
        Phi_1 = get_discrete_matrices(self.A, self.B, s_bar, self.s_n)

        # 构建增广矩阵
        Omega_d = np.zeros((9, 9))
        Omega_d[:6, :6] = self.Omega_k
        Omega_d[:6, 6:] = Phi_1
        Phi_d = np.zeros((9, 3))
        Phi_d[:6, :] = Phi_0
        Phi_d[6:, :] = np.eye(3)
        Omega_cl = Omega_d + Phi_d @ self.K  # 闭环

        Omega_op = np.zeros((9, 9))
        Omega_op[:6, :6] = self.Omega_k
        Omega_op[:6, 6:] = self.Phi_total
        Omega_op[6:, 6:] = np.eye(3)  # 开环

        Q_aug = np.zeros((9, 9))
        Q_aug[:6, :6] = self.Q_noise
        Tr_PQ = np.trace(self.P_lyap @ Q_aug)

        max_Gamma = 0.01

        # 2. 遍历每个用户，计算基于真实物理状态的 Gamma
        for user in self.connected_users:
            # --- 构建真实的状态误差向量 xi ---
            pos_err = self.loc - user.loc  # 位置误差 (3维)
            vel_err = self.vel - user.vel  # 速度误差 (3维)

            # xi = [pos_err, vel_err, u_last] (9维)
            xi = np.concatenate([pos_err, vel_err, self.u_last])

            # 计算能量
            V_curr = xi.T @ self.P_lyap @ xi
            V_close = xi.T @ Omega_cl.T @ self.P_lyap @ Omega_cl @ xi
            V_open = xi.T @ Omega_op.T @ self.P_lyap @ Omega_op @ xi

            # 计算 Gamma
            denom = V_open - V_close
            num = V_open - self.rho * V_curr - Tr_PQ

            if denom != 0:
                Gamma_i = np.clip(num / denom, 0.01, 0.99)
            else:
                Gamma_i = 0.99

            # 系统必须满足最坏情况（最难稳定的那个用户）
            if Gamma_i > max_Gamma:
                max_Gamma = Gamma_i

        # 3. 更新 T_budget
        D_req = (1 - max_Gamma) * self.s_n
        T_fixed = 0.03 + 4 * self.tau

        self.T_budget = max(0.001, D_req - T_fixed)

        # (可选) 打印调试信息，看看现在的 Gamma 是多少
        # print(f"UAV {self.id} Max Gamma: {max_Gamma:.2f}, Budget: {self.T_budget:.3f}s")


# ==========================================
# 3. 多 UAV 博弈核心逻辑
# ==========================================

def solve_single_uav_pricing(uav):
    """
    单 UAV 优化策略
    """
    # --- 关键：在定价前，根据当前用户状态更新约束 ---
    uav.update_time_budget()

    active_users = uav.connected_users
    if not active_users:
        return max(uav.price * 0.9, uav.c_n * 1.1)

    while True:
        if not active_users:
            return uav.c_n * 1.5

        Theta_sum = 0.0
        Inv_H_sum = 0.0

        for user in active_users:
            H_i = user.get_channel_H(uav)
            user.B_min_temp = user.S / (uav.T_budget * H_i)
            Theta_sum += user.theta
            Inv_H_sum += (1.0 / H_i)

        p_bars = [u.theta / (u.B_min_temp + 1.0 / u.get_channel_H(uav)) for u in active_users]
        p_max = min(p_bars)
        p_min = Theta_sum / (uav.B_total + Inv_H_sum)

        if p_min > p_max:
            # 资源不足，剔除 theta 最小的用户
            active_users.sort(key=lambda x: x.theta)
            removed = active_users.pop(0)
            continue

        p_opt = np.sqrt((uav.c_n * Theta_sum) / Inv_H_sum)

        p_final = p_opt
        if p_final < p_min:
            p_final = p_min
        if p_final > p_max:
            p_final = p_max

        return max(p_final, uav.c_n * 1.01)


def run_multi_uav_game(uavs, users, max_iter=200):
    print(f"\n--- 启动多 UAV 资源分配博弈 (UAVs: {len(uavs)}, Users: {len(users)}) ---")

    for uav in uavs:
        uav.price = np.random.uniform(2.0, 5.0) * 1e-5
        uav.history_prices.append(uav.price)

    for iteration in range(max_iter):
        print(f"\n[Iteration {iteration + 1}]")

        # Stage 1: 用户关联
        for uav in uavs:
            uav.connected_users = []
        for user in users:
            best_util = -np.inf
            best_uav = None
            for uav in uavs:
                util, _ = user.calculate_potential_utility(uav, uav.price)
                if util > best_util:
                    best_util = util
                    best_uav = uav

            if best_uav and best_util > 0:
                user.connected_uav_id = best_uav.id
                user.current_utility = best_util
                best_uav.connected_users.append(user)
            else:
                user.connected_uav_id = -1

        # Stage 2: UAV 定价更新
        max_price_change = 0.0

        for uav in uavs:
            old_price = uav.price
            new_price = solve_single_uav_pricing(uav)

            # 平滑更新
            alpha = 0.4
            updated_price = alpha * new_price + (1 - alpha) * old_price

            uav.price = updated_price
            uav.history_prices.append(updated_price)

            change = abs(updated_price - old_price) / (old_price + 1e-9)
            max_price_change = max(max_price_change, change)

            print(
                f"  UAV {uav.id}: Users={len(uav.connected_users)}, Price={uav.price:.2e}, Budget={uav.T_budget:.3f}s")

        if max_price_change < 1e-3:
            print(f"\n>>> 博弈在第 {iteration + 1} 轮收敛！ <<<")
            break

    return uavs, users


# ==========================================
# 4. 可视化函数
# ==========================================
def plot_results(uavs, users):
    # 图 1: 价格收敛
    plt.figure(figsize=(7, 5))
    for uav in uavs:
        plt.plot(uav.history_prices, marker='o', markersize=4, label=f'UAV {uav.id}')
    plt.xlabel('Iteration')
    plt.ylabel('Price')
    plt.title('Price Convergence')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 图 2: 带宽分配
    plt.figure(figsize=(7, 5))
    users.sort(key=lambda u: u.id)
    user_ids = [u.id for u in users]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, uav in enumerate(uavs):
        connected_uids = []
        bandwidths = []
        for u in users:
            if u.connected_uav_id == uav.id:
                connected_uids.append(u.id)
                _, B_opt = u.calculate_potential_utility(uav, uav.price)
                H_i = u.get_channel_H(uav)
                B_min = u.S / (uav.T_budget * H_i)
                u.assigned_bandwidth = max(B_opt, B_min)
                bandwidths.append(u.assigned_bandwidth / 1e6)

        if connected_uids:
            plt.bar(connected_uids, bandwidths, color=colors[i % len(colors)], label=f'UAV {uav.id}', width=0.6)

    unconnected = [u.id for u in users if u.connected_uav_id == -1]
    if unconnected:
        plt.bar(unconnected, [0] * len(unconnected), color='black', label='Unconnected')

    plt.xlabel('User ID')
    plt.ylabel('Bandwidth (MHz)')
    plt.title('Allocated Bandwidth')
    plt.xticks(user_ids)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. 主程序
# ==========================================

# if __name__ == "__main__":
#     np.random.seed(4)
#
#     # 5.1 初始化 3 个 UAV (位置和速度)
#     # 修改说明：
#     # 1. 位置 x,y 均设置为非零值，模拟分布在不同空域。
#     # 2. z=100 保持不变，表示飞行高度。
#     # 3. 速度方向大致指向坐标原点(幸存者聚集区)，模拟搜救过程。
#     uavs = [
#         # UAV 1: 位于左下方，向右上方飞行搜索
#         UAV(nid=1, loc=[-120, -80, 100], vel=[5, 2, 0],
#             B_total=20e6, cost_per_bw=1e-7, target_s_n=0.5),
#
#         # UAV 2: 位于右下方，向左飞行搜索
#         UAV(nid=2, loc=[120, -60, 100], vel=[-4, 1, 0],
#             B_total=20e6, cost_per_bw=1e-7, target_s_n=0.5),
#
#         # UAV 3: 位于正上方偏右，慢速巡航
#         UAV(nid=3, loc=[40, 130, 100], vel=[-1, -3, 0],
#             B_total=20e6, cost_per_bw=1e-7, target_s_n=0.5)
#     ]
#
#     # 5.2 初始化 15 个用户 (位置和速度随机)
#     users = []
#     for i in range(15):
#         # 用户分布在中心区域 [-150, 150]
#         loc = [np.random.uniform(-150, 150), np.random.uniform(-150, 150), 0]
#
#         # 随机速度 (模拟幸存者在地面缓慢移动，z轴速度为0)
#         # 速度范围设小一点，比如 +/- 1.5 m/s (步行速度)
#         vel = [np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5), 0]
#
#         S = 5 * 1024 * 8
#         theta = np.random.uniform(20, 80)
#
#         # 实例化
#         u = User(uid=i + 1, loc=loc, vel=vel, data_size_S=S, priority_weight_theta=theta, max_power_P=0.2)
#         users.append(u)
#
#     # 5.3 运行博弈
#     uavs, users = run_multi_uav_game(uavs, users)
#
#     # 5.4 结果可视化
#     plot_results(uavs, users)

# ==========================================
# 5. 主程序
# ==========================================

if __name__ == "__main__":
    np.random.seed(40)

    # uavs = [
    #     UAV(1, [-100, 0, 100], [5, 0, 0], 20e6, 1e-7, 0.5),  # 向右飞
    #     UAV(2, [100, 0, 100], [-5, 0, 0], 20e6, 1e-7, 0.5),  # 向左飞
    #     UAV(3, [0, 100, 100], [0, 0, 0], 20e6, 1e-7, 0.5)  # 悬停
    # ]

    # 5.1 初始化 UAV (位置和速度)
    uavs = []
    for i in range(3):
        loc = [np.random.uniform(-150, 150), np.random.uniform(-150, 150), 100]
        vel = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0]
        B_total = 20e6
        cost_per_bw = 1e-7
        target_s_n = 0.5
        uav = UAV(i + 1, loc, vel, B_total, cost_per_bw, target_s_n)
        uavs.append(uav)

    print(f"{'ID':<5} | {'Location':<20} | {'Velocity':<20} | {'BW (MHz)':<10} | {'Cost':<10} | {'Cycle(s)':<8}")
    print("-" * 90)

    for u in uavs:
        # 注意：numpy array 直接打印可能会有换行，这里用 array2string 或直接转 list 更好看
        loc_str = np.array2string(u.loc, precision=1, separator=',', suppress_small=True)
        vel_str = np.array2string(u.vel, precision=1, separator=',', suppress_small=True)

        print(f"{u.id:<5} | {loc_str:<20} | {vel_str:<20} | {u.B_total / 1e6:<10.1f} | {u.c_n:<10.2e} | {u.s_n:<8.2f}")

    # 5.2 初始化用户 (位置和速度随机)
    users = []
    for i in range(15):
        loc = [np.random.uniform(-150, 150), np.random.uniform(-50, 150), 0]
        # 随机速度 (模拟幸存者在移动)
        vel = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0]

        S = 5 * 1024 * 8
        theta = np.random.uniform(20, 80)

        # 实例化用户时传入 vel
        u = User(i + 1, loc, vel, S, theta, 0.2)
        users.append(u)

    print(f"\n{'ID':<5} | {'Location':<25} | {'Velocity':<25} | {'Theta':<6} | {'Data(KB)':<8} | {'P_tx(W)':<7}")
    print("-" * 100)

    for u in users:
        # 格式化 numpy 数组，去掉不必要的空格，保留1位小数
        # suppress_small=True 会把极小的数(如 1e-15)显示为 0，让表格更整洁
        loc_str = np.array2string(u.loc, precision=1, separator=', ', suppress_small=True)
        vel_str = np.array2string(u.vel, precision=1, separator=', ', suppress_small=True)

        # 打印每一行
        # u.S / (1024 * 8) 把 bits 转换为 KB
        print(
            f"{u.id:<5} | {loc_str:<25} | {vel_str:<25} | {u.theta:<6.1f} | {u.S / (1024 * 8):<8.1f} | {u.P_tx:<7.2f}")

    # 5.3 运行
    uavs, users = run_multi_uav_game(uavs, users)

    # 5.4 结果
    plot_results(uavs, users)
