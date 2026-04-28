import json
import matplotlib.pyplot as plt

with open('./cache/speed_logs/pid_vehicle_49.json') as f:
    pid = json.load(f)

with open('./cache/speed_logs/eco_vehicle_49.json') as f:
    eco = json.load(f)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot([x['speed'] for x in pid], label='PID')
axes[0].plot([x['speed'] for x in eco], label='ECO')
axes[0].set_ylabel('Speed')
axes[0].legend()

axes[1].plot([x['throttle'] for x in pid], label='PID')
axes[1].plot([x['throttle'] for x in eco], label='ECO')
axes[1].set_ylabel('Throttle')
axes[1].legend()

axes[2].plot([x['brake'] for x in pid], label='PID')
axes[2].plot([x['brake'] for x in eco], label='ECO')
axes[2].set_ylabel('Brake')
axes[2].legend()

plt.xlabel('Time Step')
plt.tight_layout()
plt.savefig('./cache/speed_logs/pid_vs_eco_comparison.png', dpi=150, bbox_inches='tight')
plt.show()