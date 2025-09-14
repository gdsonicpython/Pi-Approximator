import random
import numpy as np

pi = np.pi
learning_rate = 0.05
episodes = 500000
num_agents = 10

# --- Reward system ---
def decimal_digits_correct(ans, pi_val=pi):
    s_ans = f"{ans:.15f}"
    s_pi = f"{pi_val:.15f}"
    count = 0
    for a, b in zip(s_ans, s_pi):
        if a == b:
            count += 1
        else:
            break
    return count

def improved_reward(n, x):
    try:
        ans = x**(1/n)
        rel_error = abs(pi - ans) / pi
        digits_correct = decimal_digits_correct(ans)
        reward_value = (digits_correct + 1) / (rel_error + 1e-10)
        reward_value /= (1 + np.log10(x))
        return reward_value
    except:
        return -100

x_start_values = [random.randint(10**i, 10**(i+1)-1) for i in range(2, 12)]

agents_results = []

for agent_idx in range(num_agents):
    # Initialize n and x
    n = round(random.uniform(2, 20), 2)
    x = x_start_values[agent_idx]

    best_n, best_x, best_r = n, x, improved_reward(n, x)

    for episode in range(episodes):
        # Propose changes around current best
        n_new = best_n + random.uniform(-0.5, 0.5)
        n_new = max(0.01, round(n_new, 2))

        factor = 1 + random.uniform(-0.05, 0.05)  # ±5% change
        x_new = max(1, int(round(best_x * factor)))

        r_new = improved_reward(n_new, x_new)

        if r_new > best_r:
            best_r = r_new
            best_n = n_new
            best_x = x_new
        
        if episode % 1000 == 0:
            ans = best_x**(1/best_n)
            print(f"Agent {agent_idx+1}, Episode {episode}: n={best_n}, x={best_x}, ans={ans:.15f}, reward={best_r:.4f}")

    agents_results.append((best_n, best_x, best_x**(1/best_n), best_r))

print("\n=== Final results for all agents ===")
for idx, (n, x, ans, r) in enumerate(agents_results):
    print(f"Agent {idx+1}: n={n}, x={x}, x^(1/n)={ans:.15f}, reward={r:.4f}")
