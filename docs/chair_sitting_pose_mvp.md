# Chair Sitting Pose MVP (VLM)

This MVP extracts plausible 2D sitting keypoints from a single chair-only image.

## Scope

Input:
- one chair image

Output:
- 2D keypoints for imagined sitting person
- confidence and visibility per keypoint
- checker score and failure reasons

Keypoints:
- hip_center
- left_knee, right_knee
- left_ankle, right_ankle
- left_shoulder, right_shoulder
- left_elbow, right_elbow
- left_wrist, right_wrist

## Pipeline

1. Stage 0 (`chair_pose_stage_0_parts_system.txt`)
- extract coarse chair part anchors (`seat/backrest/floor`, optional `armrests/legs`)

2. Stage A (`chair_pose_stage_a_system.txt`)
- extract `seat_region`, `floor_line`, `backrest_region`
- optionally use Stage 0 anchors as prior

3. Stage B (`chair_pose_stage_b_system.txt`)
- predict keypoints anchored to Stage A geometry

4. Checker (`chair_pose_checker_system.txt`)
- return sitting plausibility score in `[0..4]`

5. Failure handling (`chair_pose_repair_system.txt`)
- retry on invalid JSON or schema mismatch
- run repair rounds if score below threshold

## Output contract

Main output file includes:
- `status`: `ok`, `low_score`, or `failed`
- `part_segmentation`: Stage 0 JSON
- `geometry`: Stage A JSON
- `pose`: Stage B JSON
- `checker`: VLM checker JSON
- `local_checker`: deterministic local checker JSON
- `effective_score`: `min(checker.score, local_checker.score)`
- `stage_logs`: attempt counts and errors

Strict schema files:
- `schemas/chair_parts_v1.schema.json`
- `schemas/chair_geometry_v1.schema.json`
- `schemas/sitting_keypoints_v1.schema.json`
- `schemas/sitting_pose_check_v1.schema.json`

## Run

Requirements:
- For `--backend gemini`: `GEMINI_API_KEY` env var set
- For `--backend ollama`: local Ollama server running (`http://127.0.0.1:11434`)

Example (Gemini):

```bash
python run_chair_sitting_pose.py \
  --backend gemini \
  --image /absolute/path/to/chair.jpg \
  --output runs/chair_pose_result.json \
  --model gemini-2.5-pro \
  --part-segmentation \
  --parts-min-confidence 0.2 \
  --persona "tired: hip posterior on seat, lower shoulders, feet slightly forward" \
  --min-score 3
```

Example (Ollama local):

```bash
python run_chair_sitting_pose.py \
  --backend ollama \
  --model qwen2.5vl:3b \
  --image /absolute/path/to/chair.jpg \
  --output runs/chair_pose_result.json \
  --no-part-segmentation \
  --min-score 3 \
  --timeout 420
```

## Notes

- This is imagined pose generation, not observed human pose estimation.
- Confidence is mandatory because chair-only images are ambiguous.
- For better stability, keep Stage A -> Stage B separation.
- On Apple Silicon CPU backends, first vision inference can be slow; increase `--timeout`.
