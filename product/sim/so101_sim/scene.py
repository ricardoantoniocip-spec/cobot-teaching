"""The digital twin: SO-101 on a table with a part to pick, a place zone and two cameras.

The scene file is written next to the arm model (MuJoCo looks for meshes relative to
the main file), then loaded. Edit PART_POS / PLACE_POS / PART_SIZE to match your bench.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .fetch import MODEL_DIR, ensure_model

# Positions in the robot base frame, metres (x forward, y left). The `demos` command varies them.
PART_POS = np.array([0.24, 0.10])
PLACE_POS = np.array([0.24, -0.10])
PART_SIZE = (0.02, 0.02, 0.04)  # part size x, y, z in metres (a small printed block)

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TCP_SITE = "gripperframe"  # same point as gripper_frame_link in the URDF

# Gripper joint angle (rad) for "open" and "closed". Closed is past the part so it squeezes.
GRIPPER_OPEN_RAD = 1.2
GRIPPER_CLOSED_RAD = -0.15

SCENE_XML = """<mujoco model="so101_kitting">
  <include file="so101_new_calib.xml"/>
  <option timestep="0.002" cone="elliptic" impratio="10"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="150" elevation="-25" offwidth="640" offheight="480"/>
  </visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.45 0.42 0.38" rgb2="0.42 0.39 0.35" markrgb="0.3 0.3 0.3" width="300" height="300"/>
    <material name="table" texture="grid" texuniform="true" texrepeat="10 10"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="table" type="plane" size="0.6 0.6 0.01" material="table" friction="1 0.005 0.0001"/>
    <body name="part" pos="{px} {py} {pz}" euler="0 0 {yaw}">
      <freejoint name="part"/>
      <geom name="part" type="box" size="{hx} {hy} {hz}" rgba="0.15 0.45 0.85 1" mass="0.01" friction="1.5 0.01 0.001" condim="4"/>
    </body>
    <site name="place_zone" type="cylinder" pos="{qx} {qy} 0.0005" size="0.03 0.0005" rgba="0.2 0.8 0.3 0.5"/>
    <camera name="top" pos="0.22 0 0.75" xyaxes="0 -1 0 1 0 0"/>
    <camera name="front" pos="0.7 0 0.35" xyaxes="0 1 0 -0.4 0 0.9"/>
  </worldbody>
</mujoco>
"""


def scene_path(part_pos=PART_POS, place_pos=PLACE_POS, part_size=PART_SIZE) -> Path:
    ensure_model()
    hx, hy, hz = (v / 2 for v in part_size)
    xml = SCENE_XML.format(px=part_pos[0], py=part_pos[1], pz=hz, yaw=np.arctan2(part_pos[1], part_pos[0]), hx=hx, hy=hy, hz=hz, qx=place_pos[0], qy=place_pos[1])
    path = MODEL_DIR / "scene_kitting.xml"
    path.write_text(xml)
    return path


def load(**kw) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(scene_path(**kw)))
    # More grip on the jaws (collision geoms are group 3). MuJoCo uses the convex hull of each mesh.
    jaws = (model.body("gripper").id, model.body("moving_jaw_so101_v1").id)
    for i in range(model.ngeom):
        if model.geom_bodyid[i] in jaws and model.geom_group[i] == 3:
            model.geom_friction[i] = (1.5, 0.01, 0.001)
    data = mujoco.MjData(model)
    return model, data


def gripper_angle(gripper_open: float) -> float:
    """0 closed .. 1 open (retarget convention) -> gripper joint angle in rad."""
    return GRIPPER_CLOSED_RAD + float(gripper_open) * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)


def set_pose(model, data, q: np.ndarray, gripper_open: float = 1.0) -> None:
    """Teleport the arm (no physics), for the start pose."""
    for name, v in zip(ARM_JOINTS, q):
        data.joint(name).qpos[0] = v
        data.actuator(name).ctrl[0] = v
    data.joint("gripper").qpos[0] = gripper_angle(gripper_open)
    data.actuator("gripper").ctrl[0] = gripper_angle(gripper_open)
    mujoco.mj_forward(model, data)
