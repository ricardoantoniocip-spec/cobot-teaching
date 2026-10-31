---
name: robotics-kinematics
description: Robotics kinematics — forward/inverse kinematics, DH parameters, transformation matrices, workspace analysis, singularities, Jacobian, velocity kinematics, robot types (serial/parallel/SCARA).
metadata:
  priority: 7
  promptSignals:
    phrases:
      - "robotics"
      - "robot kinematics"
      - "forward kinematics"
      - "inverse kinematics"
      - "DH parameters"
      - "Denavit-Hartenberg"
      - "Jacobian"
      - "robot arm"
      - "workspace"
      - "end effector"
    minScore: 3
---

# Robotics Kinematics — Complete Skill

## Coordinate Frames and Transformations

### Homogeneous Transformation Matrix
4×4 matrix combining rotation and translation:
T = [R | p; 0 0 0 | 1]
R = 3×3 rotation matrix [orthogonal: Rᵀ = R⁻¹]
p = 3×1 position vector [m]

**Composition:** T_A→C = T_A→B × T_B→C (multiply left to right)

**Inverse transformation:**
T⁻¹ = [Rᵀ | -Rᵀp; 0 0 0 | 1]

### Elementary Rotations
R_x(α) = [1, 0, 0; 0, cα, -sα; 0, sα, cα]
R_y(β) = [cβ, 0, sβ; 0, 1, 0; -sβ, 0, cβ]
R_z(γ) = [cγ, -sγ, 0; sγ, cγ, 0; 0, 0, 1]
c = cosine, s = sine

## Denavit-Hartenberg (DH) Parameters

### Modified DH Convention (Craig, common in robotics)
Each link described by 4 parameters: [a_i, α_i, d_i, θ_i]
a_i = link length (distance along x_i)
α_i = link twist (rotation about x_i, from z_{i-1} to z_i)
d_i = link offset (distance along z_{i-1})
θ_i = joint angle (rotation about z_{i-1})

For revolute joint: θ_i is variable; d_i is fixed
For prismatic joint: d_i is variable; θ_i is fixed

### DH Transformation Matrix
ᵢ₋₁Tᵢ = R_x(α_{i-1}) × D_x(a_{i-1}) × R_z(θ_i) × D_z(d_i)

**Expanded:**
ᵢ₋₁Tᵢ = [cθᵢ, -sθᵢ, 0, aᵢ₋₁;
           sθᵢ cαᵢ₋₁, cθᵢ cαᵢ₋₁, -sαᵢ₋₁, -sαᵢ₋₁ dᵢ;
           sθᵢ sαᵢ₋₁, cθᵢ sαᵢ₋₁, cαᵢ₋₁, cαᵢ₋₁ dᵢ;
           0, 0, 0, 1]

**Base to end-effector:**
⁰Tₙ = ⁰T₁ × ¹T₂ × ... × ⁿ⁻¹Tₙ

## Forward Kinematics

### Definition
Given joint variables q = [q₁, q₂, ..., qₙ], find end-effector pose {R, p}
⁰Tₙ(q) = product of DH matrices → position p and orientation R

### Example — 2-DOF Planar Robot
Links: a₁, a₂; joint angles θ₁, θ₂
p_x = a₁ cos(θ₁) + a₂ cos(θ₁+θ₂)
p_y = a₁ sin(θ₁) + a₂ sin(θ₁+θ₂)
φ = θ₁ + θ₂ (end-effector angle)

### Example — Standard 6-DOF Robot (Puma-like)
Build DH table: n=6 rows with parameters [aᵢ, αᵢ, dᵢ, θᵢ]
Wrist singularity: when axes 4, 5, 6 intersect at same point → ⁰Tₙ = ⁰T₃ × ³T₆
Standard: joints 1-3 position wrist; joints 4-6 orient end-effector

## Inverse Kinematics (IK)

### Overview
Given desired end-effector pose {R_d, p_d}, find joint variables q
**Challenges:** non-unique solutions, singularities, no-solution regions, nonlinear equations

**Number of solutions:** 6-DOF robot typically has 8 solutions (elbow up/down, wrist flip, shoulder L/R)

### Analytical IK (Preferred When Possible)
**Step 1 — Decouple (wrist-partitioned robots):**
Wrist center: p_wc = p_d - d₆ × R_d[:,2] (d₆ = wrist-to-tool offset along z₆)
Joints 1-3 from p_wc using geometric/algebraic relations
Joints 4-6 from R₃₋₆ = ⁰R₃ᵀ × R_d (use ZYZ or ZYX Euler angles for R₃₋₆)

**2-DOF planar solution:**
c₂ = (p_x² + p_y² - a₁² - a₂²) / (2a₁a₂)
s₂ = ±√(1-c₂²) → two solutions (elbow up/down)
θ₂ = atan2(s₂, c₂)
θ₁ = atan2(p_y, p_x) - atan2(a₂ s₂, a₁ + a₂ c₂)

### Numerical IK (General)
Newton-Raphson: q_{k+1} = q_k + J⁺(q_k) × Δx
J⁺ = Jᵀ(JJᵀ)⁻¹ (right pseudo-inverse, for overdetermined)
Δx = x_d - f(q_k) (task space error)

**Damped least squares (DLS):**
q̇ = Jᵀ(JJᵀ + λ²I)⁻¹ ẋ (avoids singularity: λ = 0.1-1.0)

## Jacobian and Velocity Kinematics

### Geometric Jacobian
ẋ = J(q) q̇
ẋ = [v; ω] (6×1 velocity vector, linear + angular)
J(q) = 6×n matrix (n = DOF)

**For column i (revolute joint):**
Jv_i = z_{i-1} × (p_n - p_{i-1}) (linear velocity contribution)
Jω_i = z_{i-1} (angular velocity contribution)
z_{i-1} = rotation axis of joint i (third column of ⁰R_{i-1})

**For prismatic joint column i:**
Jv_i = z_{i-1} (linear only)
Jω_i = 0

### Force Transformation
τ = Jᵀ × F (joint torques from task-space forces/moments)
F = wrench [f; m] applied at end-effector

## Singularities

### Definition
Singularity: rank(J) < min(m, n) → det(J) = 0 (for square J)
At singularity: lost DOF in task space; infinite joint velocities for finite task velocity

### Types
**Workspace boundary singularity:** arm fully extended or fully folded — expected
**Interior singularity (wrist):** when wrist axes are aligned → infinite solutions in null space
Puma-type at θ₅ = 0°: all solutions with any θ₄ + θ₆ = const

**Singularity avoidance:**
Monitor manipulability: w = √(det(J Jᵀ)) — decreases near singularity
Damped least squares: automatically degrades gracefully near singularity

## Workspace

### Reachable vs. Dexterous Workspace
Reachable workspace: all positions the end-effector can reach with at least one orientation
Dexterous workspace: positions where all orientations are achievable
Typical 6-DOF: dexterous workspace ⊂ reachable workspace

**Joint limits and workspace:**
With joint limits: non-spherical reachable workspace
Collision constraints: further reduce practical workspace

### Workspace Calculation
Monte Carlo sampling: random q in joint limits → compute FK → scatter positions
Discretized joint space: grid of joint angles → FK → boundary detection

## Robot Types Summary

| Type | DOF | Configuration | Workspace Shape | Accuracy | Application |
|------|-----|--------------|----------------|---------|------------|
| Cartesian (gantry) | 3 | PPP | Rectangular box | ±0.01 mm | Large parts, CNC |
| Cylindrical | 3 | RPP | Cylinder | ±0.1 mm | Palletizing |
| Spherical (polar) | 3 | RRP | Partial sphere | ±0.1 mm | Older design |
| SCARA | 4 | RRPR | Cylinder, vertical z | ±0.01 mm | Assembly, pick&place |
| Serial 6-DOF | 6 | RRRRRR | Irregular sphere | ±0.05 mm | General industrial |
| Parallel (Delta) | 3-6 | Multiple chains | Limited volume | ±0.05 mm | High speed pick, food |
| 7-DOF redundant | 7 | RRRRRRR | Enhanced sphere | ±0.05 mm | Human-collaborative |

**Payload to weight ratio (serial):** typically 1:10 to 1:5 (100 kg robot carries 10-20 kg)
**Repeatability vs. accuracy:** robots are highly repeatable (±0.02-0.1 mm) but may have poor absolute accuracy without calibration

## Output
Provide: DH table with parameters, ⁰Tₙ FK result (position [m] + rotation matrix), IK solutions (all configurations), Jacobian J at given q, manipulability w, singularity detection (det(J) = 0 check), joint torques τ = Jᵀ F for given wrench.
