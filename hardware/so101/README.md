# SO-101 robot model

`so101_new_calib.urdf` is the official SO-101 URDF, copied unchanged from
TheRobotStudio/SO-ARM100 (Apache 2.0), file `Simulation/SO101/so101_new_calib.urdf`,
commit a758567c3978dfeefe282ede0500085a48fe8f78:
https://github.com/TheRobotStudio/SO-ARM100/blob/a758567c3978dfeefe282ede0500085a48fe8f78/Simulation/SO101/so101_new_calib.urdf

The mesh files it refers to are not copied. Kinematics code only needs the joint frames.
The tool point (TCP) is the `gripper_frame_link`.
