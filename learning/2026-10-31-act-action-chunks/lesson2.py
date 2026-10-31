# Lesson 2: train ACT for a short run and look at a predicted action chunk.
# Run in Google Colab with a T4 GPU (Runtime > Change runtime type > T4 GPU).
# Each "# %%" block is one notebook cell.

# %% 1. Install LeRobot (2-4 minutes). Restart the session if Colab asks.
# !pip install -q "lerobot[training]==0.6.1"

# %% 2. Check you have a GPU. It should print True.
import torch
print("GPU available:", torch.cuda.is_available())

# %% 3. Train ACT for 2,000 steps (10-20 minutes). Watch the "loss" values in the log.
# !lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --policy.device=cuda --policy.push_to_hub=false --steps=2000 --batch_size=8 --log_freq=200 --save_freq=2000 --output_dir=outputs/lesson2_act

# %% 4. YOUR EXERCISE part 1: fill in the loss you saw at each step, then plot it.
import matplotlib.pyplot as plt
steps = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
losses = []  # type your 10 loss values here, e.g. [1.2, 0.8, ...]
if losses:
    plt.plot(steps, losses, marker="o")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("ACT training loss")
    plt.show()

# %% 5. Load your trained model and the dataset.
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ckpt = "outputs/lesson2_act/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(ckpt)
policy.to("cuda")
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=ckpt)

ds = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")
print("chunk size:", policy.config.chunk_size, "actions =", policy.config.chunk_size / ds.fps, "seconds")

# %% 6. Ask ACT for one chunk and compare it with what the human did next.
start = 100      # frame to predict from (try other values)
joint = 8        # 0-6 = left arm, 7-13 = right arm; 8 = right shoulder

frame = ds[start]
batch = preprocessor(frame)
chunk = postprocessor(policy.predict_action_chunk(batch))  # shape (1, 100, 14)
predicted = chunk[0, :, joint].cpu().numpy()

recorded = [float(a[joint]) for a in ds.hf_dataset.select(range(start, start + len(predicted)))["action"]]

plt.plot(recorded, label="human (recorded)")
plt.plot(predicted, label="ACT (predicted)")
plt.xlabel("steps ahead (50 per second)")
plt.ylabel(f"joint {joint} target")
plt.legend()
plt.title(f"One ACT chunk from frame {start}")
plt.show()

# %% 7. YOUR EXERCISE part 2: try 2 other start frames and one other joint.
# Write your one-sentence verdict here as a comment.
