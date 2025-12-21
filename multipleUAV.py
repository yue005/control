import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
import matplotlib.pyplot as plt
import copy


# ==========================================
# 1. 基础物理与积分计算
# ==========================================
def get_discrete_matrices(A, B, t_start, t_end):
    dt = t_end - t_start
    if dt <= 0: return np.zeros(B.shape)
    n, m = A.shape[0], B.shape[1]
    Phi = np.zeros((n, m))

    def integrand_matrix(t):
        return expm(A * t) @ B

    for i in range(n):
        for j in range(m):
            val, _ = quad(lambda t: integrand_matrix(t)[i, j], t_start, t_end)
            Phi[i, j] = val
    return Phi


# ==========================================
# 2. 用户与 UAV 类定义
# ==========================================
class User:
    def __init__(self, uid, loc, vel, data_size_S, priority_weight_theta, max_power_P):
        self.id = uid
        self.loc = np.array(loc, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.S = data_size_S
        self.theta = priority_weight_theta
        self.P_tx = max_power_P

        # 运行时变量
        self.associated_uav_id = -1
        self.H_i = 0.0
        self.current_snr = 0.0
        self.B_min = 0.0
        self.assigned_bandwidth = 0.0


class UAV:
    def __init__(self, nid, loc, vel, B_total, cost_per_bw, target_s_n, color):
        self.id = nid
        self.loc = np.array(loc, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.B_total = B_total
        self.c_n = cost_per_bw
        self.s_n = target_s_n
        self.color = color

        # 控制理论参数
        self.tau, self.rho = 0.1, 0.9
        self.u_last = np.zeros(3)
        mu = 0.5
        self.A = np.zeros((6, 6))
        self.A[0:3, 3:6] = np.eye(3);
        self.A[3:6, 3:6] = -mu * np.eye(3)
        self.B = np.zeros((6, 3));
        self.B[3:6, :] = np.eye(3)
        self.P_lyap = np.eye(9)
        self.Q_noise = 0.01 * np.eye(6)
        self.K = -0.5 * np.ones((3, 9))
        self.T_budget = 0.1

    def update_time_budget(self, active_users):
        if not active_users:
            self.T_budget = 0.5 * self.s_n
            return

        s_bar = 0.6 * self.s_n
        Phi_0 = get_discrete_matrices(self.A, self.B, 0, s_bar)
        Phi_1 = get_discrete_matrices(self.A, self.B, s_bar, self.s_n)
        Omega_k = expm(self.A * self.s_n)
        Phi_total = get_discrete_matrices(self.A, self.B, 0, self.s_n)

        Omega_cl = np.zeros((9, 9))
        Omega_cl[:6, :6] = Omega_k;
        Omega_cl[:6, 6:] = Phi_1
        Phi_d = np.zeros((9, 3));
        Phi_d[:6, :] = Phi_0;
        Phi_d[6:, :] = np.eye(3)
        Omega_cl += Phi_d @ self.K

        Omega_op = np.zeros((9, 9))
        Omega_op[:6, :6] = Omega_k;
        Omega_op[:6, 6:] = Phi_total;
        Omega_op[6:, 6:] = np.eye(3)

        Tr_PQ = np.trace(self.P_lyap @ np.block([[self.Q_noise, np.zeros((6, 3))], [np.zeros((3, 9))]]))

        max_Gamma = 0.01
        for user in active_users:
            xi = np.concatenate([self.loc - user.loc, self.vel - user.vel, self.u_last])
            V_curr = xi.T @ self.P_lyap @ xi
            V_close = xi.T @ Omega_cl.T @ self.P_lyap @ Omega_cl @ xi
            V_open = xi.T @ Omega_op.T @ self.P_lyap @ Omega_op @ xi
            denom = V_open - V_close
            num = V_open - self.rho * V_curr - Tr_PQ
            Gamma_i = np.clip(num / denom, 0.01, 0.99) if denom > 0 else 0.99
            max_Gamma = max(max_Gamma, Gamma_i)

        self.T_budget = max(0.005, (1 - max_Gamma) * self.s_n - (0.03 + 4 * self.tau))

    def get_channel_gain(self, user):
        dist = max(np.linalg.norm(self.loc - user.loc), 1.0)
        h0, alpha, N_pow = 1e-4, 2.5, 1e-13
        gain = h0 * (dist ** (-alpha))
        snr = (user.P_tx * gain) / N_pow
        return np.log2(1 + snr), snr


# ==========================================
# 3. 博弈与关联逻辑
# ==========================================
def multi_uav_association(uavs, users):
    """ 用户寻找信道质量最好的 UAV """
    for user in users:
        best_h = -1
        best_uav_idx = -1
        for i, uav in enumerate(uavs):
            h, snr = uav.get_channel_gain(user)
            if h > best_h:
                best_h = h
                best_uav_idx = i
        user.associated_uav_id = uavs[best_uav_idx].id
        user.H_i = best_h
        # 记录瞬时 SNR 用于绘图
        _, user.current_snr = uavs[best_uav_idx].get_channel_gain(user)


def solve_system(uavs, users):
    # 1. 关联
    multi_uav_association(uavs, users)

    all_results = {}

    # 2. 对每个 UAV 独立求解博弈
    for uav in uavs:
        uav_users = [u for u in users if u.associated_uav_id == uav.id]
        if not uav_users:
            all_results[uav.id] = {"price": 0, "users": [], "T_budget": uav.T_budget}
            continue

        active_list = copy.copy(uav_users)
        while True:
            if not active_list: break

            uav.update_time_budget(active_list)
            Theta_sum, Inv_H_sum = 0.0, 0.0
            for u in active_list:
                u.B_min = u.S / (uav.T_budget * u.H_i)
                Theta_sum += u.theta
                Inv_H_sum += (1.0 / u.H_i)

            p_bars = [u.theta / (u.B_min + 1.0 / u.H_i) for u in active_list]
            p_max = min(p_bars)
            p_min = Theta_sum / (uav.B_total + Inv_H_sum)

            if p_min > p_max:
                active_list.sort(key=lambda x: x.theta)
                active_list.pop(0)
                continue

            p_opt = np.clip(np.sqrt((uav.c_n * Theta_sum) / (Inv_H_sum + 1e-9)), p_min, p_max)

            res_list = []
            for u in active_list:
                u.assigned_bandwidth = (u.theta / p_opt) - (1.0 / u.H_i)
                res_list.append({
                    "uid": u.id, "B": u.assigned_bandwidth, "B_min": u.B_min, "theta": u.theta
                })

            all_results[uav.id] = {"price": p_opt, "users": res_list, "T_budget": uav.T_budget}
            break

    return all_results


# ==========================================
# 4. 绘图
# ==========================================
def plot_multi_uav_results(uavs, users, results):
    fig = plt.figure(figsize=(15, 10))

    # --- 图1: 空间分布与关联 ---
    ax1 = fig.add_subplot(221)
    for uav in uavs:
        ax1.scatter(uav.loc[0], uav.loc[1], c=uav.color, marker='s', s=100, label=f'UAV {uav.id}')
        # 绘制服务范围内的用户
        associated_users = [u for u in users if u.associated_uav_id == uav.id]
        for uu in associated_users:
            ax1.scatter(uu.loc[0], uu.loc[1], edgecolors=uav.color, facecolors='none', s=50)
            ax1.plot([uav.loc[0], uu.loc[0]], [uav.loc[1], uu.loc[1]], c=uav.color, alpha=0.3, linestyle='--')
    ax1.set_title("UAV-User Association Map")
    ax1.set_xlabel("X (m)");
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- 图2: 带宽分配堆叠图 (按UAV分组) ---
    ax2 = fig.add_subplot(222)
    uav_labels = [f"UAV {u.id}" for u in uavs]
    x = np.arange(len(uav_labels))
    bottom_val = np.zeros(len(uavs))

    # 获取所有UID以固定颜色
    all_uids = [u.id for u in users]
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_uids)))
    uid_to_color = dict(zip(all_uids, colors))

    for u_idx, uav in enumerate(uavs):
        res = results.get(uav.id, {}).get("users", [])
        for user_res in res:
            bw_mhz = user_res['B'] / 1e6
            ax2.bar(u_idx, bw_mhz, bottom=bottom_val[u_idx], color=uid_to_color[user_res['uid']],
                    edgecolor='white', width=0.6)
            # 在条形图中间标注用户ID
            ax2.text(u_idx, bottom_val[u_idx] + bw_mhz / 2, f"U{user_res['uid']}",
                     ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            bottom_val[u_idx] += bw_mhz

    ax2.set_xticks(x)
    ax2.set_xticklabels(uav_labels)
    ax2.set_ylabel("Total Bandwidth (MHz)")
    ax2.set_title("Bandwidth Allocation per UAV")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # --- 图3: 价格对比 ---
    ax3 = fig.add_subplot(223)
    prices = [results[u.id]['price'] for u in uavs]
    ax3.bar(uav_labels, prices, color=[u.color for u in uavs], alpha=0.7)
    ax3.set_yscale('log')
    ax3.set_title("Equilibrium Price $p^*_n$ (Log Scale)")
    ax3.set_ylabel("Price")

    # # --- 图4: 时间预算 T_budget ---
    # ax4 = fig.add_subplot(224)
    # t_budgets = [results[u.id]['T_budget'] for u in uavs]
    # ax4.step(uav_labels, t_budgets, where='mid', marker='o', color='purple', lw=2)
    # ax4.set_title("UAV Control-Aware Time Budget ($T_{budget}$)")
    # ax4.set_ylabel("Seconds")
    # ax4.set_ylim(0, max(t_budgets) * 1.2)
    # ax4.grid(True, alpha=0.3)

    # ==========================================
    # --- 子图 4: 用户 ID vs 分配带宽 (修正字典报错问题) ---
    # ==========================================
    ax4 = fig.add_subplot(224)

    # 1. 因为 results 是字典 {uav_id: {users: [...]}}, 我们需要提取出所有用户到一个列表
    flat_results = []
    for uav in uavs:
        # 检查该 UAV 是否有分配结果
        if uav.id in results and "users" in results[uav.id]:
            for u_res in results[uav.id]["users"]:
                flat_results.append({
                    'u_id': u_res['uid'],  # 对应之前代码里的 'uid'
                    'bw': u_res['B'],  # 对应之前代码里的 'B'
                    'uav_color': uav.color,
                    'uav_id': uav.id
                })

    # 2. 现在 flat_results 是列表了，可以进行排序
    flat_results.sort(key=lambda x: x['u_id'])

    # 3. 准备绘图数据
    u_ids_str = [f"U{r['u_id']}" for r in flat_results]
    bw_mhz = [r['bw'] / 1e6 for r in flat_results]
    bar_colors = [r['uav_color'] for r in flat_results]

    # 4. 绘图
    bars = ax4.bar(u_ids_str, bw_mhz, color=bar_colors, edgecolor='black', alpha=0.8)

    # 5. 设置标签和标题
    ax4.set_xlabel("User ID", fontsize=10)
    ax4.set_ylabel("Bandwidth (MHz)", fontsize=10)
    ax4.set_title("Bandwidth per User (Colored by Associated UAV)", fontsize=12, fontweight='bold')
    ax4.grid(axis='y', linestyle='--', alpha=0.4)

    # 6. 自定义图例 (区分 UAV)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=u.color, lw=6, label=f'UAV {u.id}')
        for u in uavs
    ]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # 7. 柱状图上方数值标注
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.show()


# ==========================================
# 6. 输出所有用户初始信息 (新增)
# ==========================================
def print_user_info(users):
    print("\n" + "=" * 105)
    print(
        f"{'UID':<5} | {'Location (x, y, z)':<22} | {'Velocity (vx, vy)':<18} | {'Data S (bits)':<12} | {'Theta':<8} | {'P_tx (W)':<8}")
    print("-" * 105)

    # 按照用户 ID 排序打印
    sorted_users = sorted(users, key=lambda x: x.id)

    for u in sorted_users:
        loc_str = f"({u.loc[0]:>6.1f}, {u.loc[1]:>6.1f}, {u.loc[2]:>4.1f})"
        # 只取速度的前两个维度 (x, y) 方便观察
        vel_str = f"({u.vel[0]:>5.2f}, {u.vel[1]:>5.2f})"

        print(f"{u.id:<5} | {loc_str:<22} | {vel_str:<18} | {u.S:<12,d} | {u.theta:<8.2f} | {u.P_tx:<8.2f}")
    print("=" * 105 + "\n")

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    np.random.seed(4)

    # 初始化 3 架 UAV，分布在不同象限
    uav_configs = [
        {'id': 1, 'loc': [80, 80, 100], 'color': '#1f77b4', 'B': 20e6, 'c': 1.2e-7},
        {'id': 2, 'loc': [-80, 80, 100], 'color': '#ff7f0e', 'B': 15e6, 'c': 1.0e-7},
        {'id': 3, 'loc': [0, -80, 100], 'color': '#2ca02c', 'B': 25e6, 'c': 1.5e-7},
    ]

    my_uavs = [UAV(nid=conf['id'], loc=conf['loc'], vel=[1, 1, 0],
                   B_total=conf['B'], cost_per_bw=conf['c'], target_s_n=0.5,
                   color=conf['color']) for conf in uav_configs]

    # 初始化 15 个随机分布的用户
    my_users = []
    for i in range(15):
        loc = [np.random.uniform(-150, 150), np.random.uniform(-150, 150), 0]
        vel = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0]
        theta = np.random.uniform(30, 100)
        u = User(uid=i + 1, loc=loc, vel=vel, data_size_S=8 * 1024 * 8,
                 priority_weight_theta=theta, max_power_P=0.2)
        my_users.append(u)

    # 运行多UAV博弈系统
    game_results = solve_system(my_uavs, my_users)

    print_user_info(my_users)

    # 打印简报
    print(f"{'UAV ID':<10} | {'Users':<10} | {'Price':<12} | {'T_budget':<10}")
    print("-" * 50)
    for un in my_uavs:
        r = game_results[un.id]
        print(f"{un.id:<10} | {len(r['users']):<10} | {r['price']:<12.2e} | {r['T_budget']:<10.4f}")

    # 绘图
    plot_multi_uav_results(my_uavs, my_users, game_results)
