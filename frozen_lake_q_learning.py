import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from QLearning import QLearning, EpsilonGreedy

STATE_SIZE = 16
ACTION_SIZE = 4


ALPHAs   = [0.05, 0.1, 0.2]
GAMMAs   = [0.7, 0.9, 0.99]
EPSILONs = [0.05, 0.2, 0.5]

MAP_DESC = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]

TRAIN_EPISODES = 8000         # adjust as needed
EVAL_EPISODES = 100
EVAL_INTERVAL = 10            # fewer eval points = faster
MOVING_AVG_WINDOW = 200

SLIPPERY = True
SEED = 0

def evaluate_greedy(env, q_table, n_episodes=100):
    successes = 0
    for _ in range(n_episodes):
        s, _ = env.reset()
        terminated = truncated = False
        while not terminated and not truncated:
            a = int(np.argmax(q_table[s, :]))
            s, r, terminated, truncated, _ = env.step(a)
        successes += (r == 1)
    return successes / n_episodes

def moving_average(x, window):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return np.array([])
    return np.convolve(x, np.ones(window)/window, mode="valid")

def plot_policy_arrows(Q, lake_map, title, save_path):
    """
    lake_map: list of strings like ['SFFF','FHFH','FFFH','HFFG'].
    """
    n_rows = len(lake_map)
    n_cols = len(lake_map[0])

    # Action mapping: 0=LEFT,1=DOWN,2=RIGHT,3=UP
    arrow = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    policy = np.argmax(Q, axis=1)

    fig, ax = plt.subplots(figsize=(5, 5))

    # grid lines
    for i in range(n_cols + 1):
        ax.plot([i, i], [0, n_rows], color="black", linewidth=1)
    for j in range(n_rows + 1):
        ax.plot([0, n_cols], [j, j], color="black", linewidth=1)

    # cell content
    for r in range(n_rows):
        for c in range(n_cols):
            s = r * n_cols + c
            cell = lake_map[r][c]

            # invert y so row 0 is at top
            x = c + 0.5
            y = (n_rows - 1 - r) + 0.5

            if cell in ["S", "G", "H"]:
                ax.text(x, y, cell, ha="center", va="center", fontsize=16)
            else:
                ax.text(x, y, arrow[int(policy[s])], ha="center", va="center", fontsize=16)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

def run_one(alpha, gamma, epsilon):
    env = gym.make("FrozenLake-v1",map_name="4x4", desc=MAP_DESC, is_slippery=SLIPPERY, )
    env.reset(seed=SEED)
    np.random.seed(SEED)

    learner = QLearning(STATE_SIZE, ACTION_SIZE, alpha, gamma)
    explorer = EpsilonGreedy(epsilon)

    training_rewards = []
    eval_x, eval_success = [], []

    for ep in range(1, TRAIN_EPISODES + 1):
        s, _ = env.reset()
        terminated = truncated = False
        ep_reward = 0.0

        while not terminated and not truncated:
            a = explorer.choose_action(env.action_space, s, learner.q_table)
            ns, r, terminated, truncated, _ = env.step(a)
            learner.update(s, a, r, ns)
            s = ns
            ep_reward += r

        training_rewards.append(ep_reward)

        if ep % EVAL_INTERVAL == 0:
            sr = evaluate_greedy(env, learner.q_table, n_episodes=EVAL_EPISODES)
            eval_x.append(ep)
            eval_success.append(sr)

    final_sr = evaluate_greedy(env, learner.q_table, n_episodes=EVAL_EPISODES)
    env.close()

    return {
        "alpha": alpha, "gamma": gamma, "epsilon": epsilon,
        "final_success": final_sr,
        "eval_x": np.array(eval_x),
        "eval_success": np.array(eval_success),
        "training_rewards": np.array(training_rewards),
        "q_table": learner.q_table.copy(),
    }

def main():
    results = []

    # --- Run 27 experiments ---
    idx = 0
    for a in ALPHAs:
        for g in GAMMAs:
            for e in EPSILONs:
                idx += 1
                print(f"[{idx:02d}/27] alpha={a}, gamma={g}, epsilon={e} ...")
                out = run_one(a, g, e)
                print(f"    final success rate = {out['final_success']:.2f}")
                results.append(out)

    # --- Sort & choose best ---
    results_sorted = sorted(results, key=lambda d: d["final_success"], reverse=True)
    best = results_sorted[0]
    print("\nTop 5 combos:")
    for r in results_sorted[:5]:
        print(f"  a={r['alpha']}, g={r['gamma']}, e={r['epsilon']} -> final={r['final_success']:.2f}")

    # --- Plot 2: Success Rate vs Training Episodes (27 curves) ---
    plt.figure(figsize=(10, 6))
    for r in results:
        label = f"a={r['alpha']}, g={r['gamma']}, e={r['epsilon']}"
        plt.plot(r["eval_x"], r["eval_success"], label=label)
    plt.xlabel("Training Episodes")
    plt.ylabel("Success Rate (Greedy Evaluation)")
    plt.title("Success Rate vs Training Episodes (27 experiments)")
    plt.grid(True)
    plt.legend(fontsize=7, ncol=2)  # crowded legend; adjust ncol if needed
    plt.savefig("success_rate_27.png", dpi=200, bbox_inches="tight")
    plt.show()

    # --- Plot 3: Moving Avg Reward vs Training Episodes (27 curves) ---
    plt.figure(figsize=(10, 6))
    for r in results:
        ma = moving_average(r["training_rewards"], MOVING_AVG_WINDOW)
        if len(ma) == 0:
            continue
        x = np.arange(MOVING_AVG_WINDOW, MOVING_AVG_WINDOW + len(ma))
        label = f"a={r['alpha']}, g={r['gamma']}, e={r['epsilon']}"
        plt.plot(x, ma, label=label)
    plt.xlabel("Training Episodes")
    plt.ylabel(f"Moving Avg Reward (window={MOVING_AVG_WINDOW})")
    plt.title("Moving Average Reward vs Training Episodes (27 experiments)")
    plt.grid(True)
    plt.legend(fontsize=7, ncol=2)
    plt.savefig("moving_avg_reward_27.png", dpi=200, bbox_inches="tight")
    plt.show()

    # --- Plot 1: Policy visualization (best only) ---
    print(best["q_table"])
    plot_policy_arrows(
        best["q_table"],
        MAP_DESC,
        title=f"Best Policy (a={best['alpha']}, g={best['gamma']}, e={best['epsilon']}, final={best['final_success']:.2f})",
        save_path="policy_plot_best.png"
    )

if __name__ == "__main__":
    main()