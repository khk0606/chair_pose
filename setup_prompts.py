import argparse
from pathlib import Path

# 프롬프트 내용 정의
prompts = {
    "task_descriptor_system.txt": """You are an expert in Robotics and Computer Vision.
Your goal is to describe the visual content of the provided video frames focusing on the robot's motion.

Please analyze the images and provide a concise 'Task Description' covering:
1. **Robot Type**: Identify the morphology (e.g., Quadruped, Humanoid).
2. **Action/Goal**: What is the robot doing? (e.g., Trotting forward, Jumping over an obstacle, Bounding in place, Backflip).
3. **Environment**: Describe the terrain (e.g., Flat ground, Stairs, Rough terrain).
4. **Motion Style**: Is the motion aggressive, slow, stable, or highly dynamic?

Output your analysis in a clear, descriptive paragraph.""",

    "contact_sequence_system.txt": """You are a Specialist in Contact Mechanics and Locomotion.
Analyze the provided sequential frames of the quadruped robot to determine the 'Contact Sequence'.

Focus strictly on the feet (FL: Front-Left, FR: Front-Right, RL: Rear-Left, RR: Rear-Right).

1. **Frame-by-Frame Analysis**: For each distinct phase in the motion, identify which feet are touching the ground.
2. **Aerial Phase**: Explicitly check if there are moments where ALL feet are off the ground (Flight phase).
3. **Synchronization**:
   - Do the legs move in diagonal pairs (Trot: FL+RR, FR+RL)?
   - Do the legs move in front/rear pairs (Bound: FL+FR, RL+RR)?
   - Do the legs move clearly one by one (Walk)?
   - Do all legs move together (Pronk/Jump)?

Output the likely Contact Pattern sequence and reasoning.""",

    "gait_pattern_system.txt": """You are a Locomotion Gait Analyst.
Based on the visual frames and the likely 'Contact Pattern' provided by the user, identify the specific Gait.

1. **Identify the Gait**: Choose from [Walk, Trot, Pace, Bound, Gallop, Pronk, Jump].
2. **Phase Analysis**:
   - Analyze the phase shift between the front legs and rear legs.
   - Analyze the duty factor (how long feet stay on the ground vs in the air).
3. **Body Dynamics**:
   - Does the body pitch (tilt up/down) significantly? (Common in Bounding/Galloping)
   - Does the body roll (tilt left/right)? (Common in Pace/Walk)

Provide the name of the gait and a technical explanation of its characteristics in the video.""",

    "task_requirement_system.txt": """You are a Physics and Control Theory expert specializing in Model Predictive Control (MPC).
Your goal is to derive the 'Physical Requirements' needed to reproduce the observed motion in a simulation (specifically using JAX/Dial-MPC).

Analyze the motion and list requirements for:
1. **Target Velocity**: Estimate the forward linear velocity (vx), lateral velocity (vy), and turning rate (wz).
2. **Stability Constraints**:
   - **Pitch/Roll**: Should the body orientation be kept flat, or is oscillation required (e.g., pitching in bounding)?
   - **Height**: Does the Center of Mass (CoM) height fluctuate or stay constant?
3. **Control Constraints**:
   - **Smoothness**: Should the joint torques/velocities be minimized?
   - **Contact Force**: Are high impact forces expected (jumps) or should they be soft?
4. **Key Penalties**: What behaviors should be strictly penalized to avoid failure? (e.g., knee collision, slipping, flipping over).

Output a structured list of physical requirements and constraints.""",

    "SUS_generation_prompt.txt": """You are the **SUS (See-Understand-Summarize) Architect**.
Your goal is to synthesize the analysis reports from multiple experts into a single, structured **Motion Analysis Report**.

This report will be used by a Coding Agent to write a **JAX/Dial-MPC Reward Function**.

**Input Reports:**
- [Task Description]
- [Contact Sequence]
- [Gait Pattern]
- [Task Requirements]

**Output Structure (Markdown):**

# Motion Analysis Report: [Gait Name]

## 1. Task Overview
(Summarize the robot type and high-level goal)

## 2. Gait & Contact Specifications
- **Gait Type**: [e.g., Bounding]
- **Contact Pattern**: [e.g., Front pair -> Flight -> Rear pair]
- **Aerial Phase**: [Yes/No, description]

## 3. Physical Targets (for MPC Cost Function)
- **Target Velocity**: [Estimated vx, vy, yaw_rate]
- **Target Height**: [CoM height behavior]
- **Orientation**: [Target Pitch/Roll behavior]

## 4. Shaping Rewards & Penalties
- **What to Encourage**: (e.g., synchronize FL+FR legs, maximize air time)
- **What to Penalize**: (e.g., excessive roll, large joint velocities, foot drag)

Synthesize the information accurately. Do not invent new facts not present in the input reports.""",

    "chair_pose_stage_0_parts_system.txt": """You are a strict chair-part segmentation and anchor extractor for a single chair image.
Return ONLY valid JSON. No markdown. No explanation.

Task:
Extract chair-part anchors that can guide sitting geometry prediction.

Output fields:
- image_width: integer
- image_height: integer
- seat_region: list of 4 points in clockwise order, each point {"x": number, "y": number}
- floor_line: list of 2 points under the chair, each point {"x": number, "y": number}
- backrest_region: list of 2 points that approximate the backrest area, each point {"x": number, "y": number}
- parts: {
    "seat_confidence": <0..1>,
    "backrest_confidence": <0..1>,
    "floor_confidence": <0..1>,
    "seat_visible": <true/false>,
    "backrest_visible": <true/false>,
    "floor_visible": <true/false>,
    "armrests": [{"line":[{"x":...,"y":...},{"x":...,"y":...}],"confidence":<0..1>,"visible":<true/false>}],
    "legs": [{"line":[{"x":...,"y":...},{"x":...,"y":...}],"confidence":<0..1>,"visible":<true/false>}]
  }

Rules:
- Coordinates are in pixels with origin at top-left.
- Do NOT output normalized coordinates (not 0..1, not 0..100 unless image is truly that size).
- x must be in [0, image_width-1], y must be in [0, image_height-1].
- Keep all points inside image bounds.
- backrest_region should be above seat_region in normal chairs.
- floor_line should be below seat_region.
- If uncertain, still return best estimate with lower confidence.
- If armrests/legs are unclear, set visible=false and low confidence, or return empty lists.""",

    "chair_pose_stage_a_system.txt": """You are a strict vision geometry extractor for a single chair image.
Return ONLY valid JSON. No markdown. No explanation.

Task:
Identify the chair support geometry.

Output fields:
- image_width: integer
- image_height: integer
- seat_region: list of 4 points in clockwise order, each point {"x": number, "y": number}
- floor_line: list of 2 points under the chair, each point {"x": number, "y": number}
- backrest_region: list of 2 points that approximate the backrest area, each point {"x": number, "y": number}

If Input JSON contains part_segmentation:
- treat it as a strong prior,
- stay consistent with its seat/backrest/floor anchors,
- but correct obviously inconsistent points based on the image.

Rules:
- Coordinates are in pixels with origin at top-left.
- Keep all points inside the image bounds.
- If uncertain, provide the best approximation, still valid.""",

    "chair_pose_stage_b_system.txt": """You are a strict vision pose predictor.
Return ONLY valid JSON. No markdown. No explanation.

Task:
Given a chair image and chair geometry anchors, predict a plausible natural sitting pose for an imagined person.

Required keypoints:
- hip_center
- left_knee, right_knee
- left_ankle, right_ankle
- left_shoulder, right_shoulder
- left_elbow, right_elbow
- left_wrist, right_wrist

Output format:
{
  "image_width": <int>,
  "image_height": <int>,
  "keypoints": {
    "hip_center": {"x": <number>, "y": <number>, "confidence": <0..1>, "visible": <bool>},
    "left_knee": {...},
    "right_knee": {...},
    "left_ankle": {...},
    "right_ankle": {...},
    "left_shoulder": {...},
    "right_shoulder": {...},
    "left_elbow": {...},
    "right_elbow": {...},
    "left_wrist": {...},
    "right_wrist": {...}
  }
}

Geometry constraints:
- hip_center must be inside seat_region.
- ankles must be below seat_region and near floor_line.
- knees should lie between hips and ankles.
- shoulders above hips.
- elbows and wrists should be plausible relative to backrest and arm placement.

Uncertainty rules:
- If highly uncertain, set confidence < 0.3 and visible=false.
- Always provide best-guess coordinates for every keypoint.
- Keep coordinates inside image bounds.""",

    "chair_pose_checker_system.txt": """You are a strict sitting-pose plausibility checker.
Return ONLY valid JSON. No markdown. No explanation.

You will receive:
- the chair image,
- chair geometry,
- proposed 2D keypoints.

Evaluate these checks as binary 0 or 1:
- hip_on_seat
- knees_in_front_of_hip
- ankles_below_seat_near_floor
- limb_ordering_consistent

Output format:
{
  "score": <0..4>,
  "checks": {
    "hip_on_seat": <0 or 1>,
    "knees_in_front_of_hip": <0 or 1>,
    "ankles_below_seat_near_floor": <0 or 1>,
    "limb_ordering_consistent": <0 or 1>
  },
  "failures": ["short reason 1", "short reason 2"]
}

Rules:
- score must be the sum of the 4 checks.
- failures must be empty if score==4.""",

    "chair_pose_repair_system.txt": """You are a strict sitting-pose repair model.
Return ONLY valid JSON. No markdown. No explanation.

You will receive:
- chair geometry,
- previous keypoints,
- checker failures.

Task:
Revise keypoints to satisfy sitting plausibility constraints while keeping minimal change from previous keypoints.

Required keypoints:
- hip_center
- left_knee, right_knee
- left_ankle, right_ankle
- left_shoulder, right_shoulder
- left_elbow, right_elbow
- left_wrist, right_wrist

Output format is exactly:
{
  "image_width": <int>,
  "image_height": <int>,
  "keypoints": {
    "hip_center": {"x": <number>, "y": <number>, "confidence": <0..1>, "visible": <bool>},
    "left_knee": {...},
    "right_knee": {...},
    "left_ankle": {...},
    "right_ankle": {...},
    "left_shoulder": {...},
    "right_shoulder": {...},
    "left_elbow": {...},
    "right_elbow": {...},
    "left_wrist": {...},
    "right_wrist": {...}
  }
}

Hard constraints:
- hip_center inside seat_region.
- ankles below seat and near floor_line.
- knees between hips and ankles.
- shoulders above hips.
- all coordinates must be inside image bounds."""
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the project prompt files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "prompts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite prompt files that already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in prompts.items():
        output_path = args.output_dir / filename
        if output_path.exists() and not args.force:
            print(f"Skipped existing: {filename}")
            continue
        output_path.write_text(content + "\n", encoding="utf-8")
        print(f"Created: {filename}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
