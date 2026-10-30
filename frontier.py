"""Reproduce the Almgren-Chriss efficient frontier (E[IS] vs sqrt(Var[IS]))
by sweeping risk aversion lam, with TWAP marked for reference.

Run:  python frontier.py   ->  writes frontier.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from execution_env import ACParams, ExecutionEnv
from baselines import twap_schedule, ac_schedule

lams = np.logspace(-8, -5, 40)
E, SD = [], []
for lam in lams:
    p = ACParams(lam=lam)
    env = ExecutionEnv(p)
    s = ac_schedule(p)
    E.append(env.expected_cost(s))
    SD.append(np.sqrt(env.cost_variance(s)))

p0 = ACParams()
env0 = ExecutionEnv(p0)
tw = twap_schedule(p0)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(np.array(SD) / 1e3, np.array(E) / 1e3, color="teal", lw=2, label="AC frontier (varying $\\lambda$)")
ax.scatter(
    [np.sqrt(env0.cost_variance(tw)) / 1e3],
    [env0.expected_cost(tw) / 1e3],
    color="maroon", zorder=5, label="TWAP",
)
s_def = ac_schedule(p0)
ax.scatter(
    [np.sqrt(env0.cost_variance(s_def)) / 1e3],
    [env0.expected_cost(s_def) / 1e3],
    color="goldenrod", zorder=5, label=f"AC @ default $\\lambda$={p0.lam:.0e}",
)
ax.set_xlabel("Std. dev. of implementation shortfall ($ thousands)")
ax.set_ylabel("Expected implementation shortfall ($ thousands)")
ax.set_title("Almgren-Chriss efficient frontier")
ax.legend()
fig.tight_layout()
fig.savefig("frontier.png", dpi=150)
print("wrote frontier.png")
