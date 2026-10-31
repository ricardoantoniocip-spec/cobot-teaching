# Lesson 1: look inside a LeRobot demo dataset.
# Run in Google Colab (CPU is fine). Each "# %%" block is one notebook cell.

# %% 1. Install LeRobot (takes 2-4 minutes). Restart the runtime if Colab asks.
# !pip install -q "lerobot[dataset]==0.6.1"

# %% 2. Load a small simulated dataset: push a T-shaped block into place.
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot/pusht")
print(ds)  # summary
print("episodes:", ds.num_episodes)
print("frames:  ", ds.num_frames)
print("fps:     ", ds.fps)

# %% 3. What is recorded each tick? (the "columns" of the station log)
for name, info in ds.features.items():
    print(f"{name:25s} {info['dtype']:8s} {info['shape']}")

# %% 4. One frame = observation + action at one moment.
frame = ds[0]
print("state :", frame["observation.state"])  # where the pusher is (x, y)
print("action:", frame["action"])             # where it was told to go (x, y)

import matplotlib.pyplot as plt
plt.imshow(frame["observation.image"].permute(1, 2, 0))  # channels last for plotting
plt.title("What the camera saw at frame 0")
plt.axis("off")
plt.show()

# %% 5. Episode lengths, like cycle times per run.
lengths = ds.meta.episodes["length"]
print("first 10 episode lengths (frames):", lengths[:10])

# %% 6. Plot one episode: state vs action over time (no video decoding, fast).
ep = 0
start = ds.meta.episodes["dataset_from_index"][ep]
end = ds.meta.episodes["dataset_to_index"][ep]
rows = ds.hf_dataset.select(range(start, end))

state_x = [float(s[0]) for s in rows["observation.state"]]
action_x = [float(a[0]) for a in rows["action"]]
t = [i / ds.fps for i in range(len(state_x))]

plt.plot(t, state_x, label="state x (where it is)")
plt.plot(t, action_x, label="action x (where it was told to go)")
plt.xlabel("time [s]")
plt.ylabel("x position")
plt.legend()
plt.title(f"Episode {ep}")
plt.show()

# %% 7. YOUR EXERCISE (see lesson.md)
# a) shortest and longest episode, in frames and seconds
# b) mean and standard deviation of episode length
# c) plot state x and action x for the longest episode
