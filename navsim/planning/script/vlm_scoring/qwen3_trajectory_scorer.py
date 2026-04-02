from __future__ import annotations

import json
import re
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

def parse_trajectory_txt(txt_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parse trajectory text file.

    Expected format per line:
    color_name: x1,y1;x2,y2;x3,y3;...

    Returns:
        {
            "red": [(x, y), ...],
            "blue": [(x, y), ...],
            ...
        }
    """
    candidates = {}

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            color, coord_blob = line.split(":", 1)
            color = color.strip()

            points = []
            for token in coord_blob.strip().split(";"):
                token = token.strip()
                if not token:
                    continue
                if "," not in token:
                    continue

                x_str, y_str = token.split(",", 1)
                try:
                    x = int(float(x_str.strip()))
                    y = int(float(y_str.strip()))
                    points.append((x, y))
                except ValueError:
                    continue

            if points:
                candidates[color] = points

    return candidates


def summarize_candidate(points: List[Tuple[int, int]]) -> Dict:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    start = points[0]
    end = points[-1]

    summary = {
        "num_points": len(points),
        "start": start,
        "end": end,
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "delta_x": end[0] - start[0],
        "delta_y": end[1] - start[1],
    }
    return summary


def build_candidate_table(candidates: Dict[str, List[Tuple[int, int]]]) -> List[Dict]:
    rows = []
    for idx, (color, pts) in enumerate(candidates.items()):
        stats = summarize_candidate(pts)
        rows.append({
            "candidate_index": idx,
            "color": color,
            "points": pts,
            "summary": stats,
        })
    return rows


def format_candidate_text(
    candidate_rows: List[Dict],
    include_raw_points: bool = False,
    max_points_preview: int = 12,
) -> str:
    lines = []
    for row in candidate_rows:
        s = row["summary"]
        line = (
            f"- candidate_{row['candidate_index']} | color={row['color']} | "
            f"num_points={s['num_points']} | "
            f"start={s['start']} | end={s['end']} | "
            f"x_range=[{s['min_x']},{s['max_x']}] | "
            f"y_range=[{s['min_y']},{s['max_y']}] | "
            f"delta=({s['delta_x']},{s['delta_y']})"
        )
        lines.append(line)

        if include_raw_points:
            preview = row["points"][:max_points_preview]
            lines.append(f"  preview_points={preview}")

    return "\n".join(lines)


def discover_pairs(search_dirs: List[Path]) -> List[Dict]:
    pairs = []

    for d in search_dirs:
        if not d.exists() or not d.is_dir():
            continue

        image_files = {}
        for ext in ("*.png", "*.jpg"):
            for p in d.glob(ext):
                image_files[p.stem] = p

        txt_files = {p.stem: p for p in d.glob("*.txt")}
        common_stems = sorted(set(image_files.keys()) & set(txt_files.keys()))

        for stem in common_stems:
            pairs.append({
                "stem": stem,
                "image_path": image_files[stem],
                "traj_path": txt_files[stem],
                "parent_dir": d,
            })

    return pairs

def build_scoring_prompt(
    candidate_rows: List[Dict],
    route_instruction: str,
    padms_text: Optional[str] = None,
    extra_context: Optional[str] = None,
    include_raw_points: bool = False,
) -> str:
    candidate_text = format_candidate_text(
        candidate_rows,
        include_raw_points=include_raw_points,
        max_points_preview=12,
    )

    padms_section = ""
    if padms_text is not None and len(padms_text.strip()) > 0:
        padms_section = f"\nPADMS or planner metric context:\n{padms_text}\n"

    extra_section = ""
    if extra_context is not None and len(extra_context.strip()) > 0:
        extra_section = f"\nAdditional context:\n{extra_context}\n"

    prompt = f"""
You are a cautious autonomous-driving trajectory scorer.

You are given:
1. A front-facing driving image with multiple color-coded candidate trajectories overlaid.
2. Structured candidate trajectory metadata extracted from a matching trajectory file.
3. A route-level driving instruction.

Your job:
- Select the SINGLE best candidate trajectory.
- Judge based on visible scene context and trajectory plausibility.
- Prefer:
  - staying in the correct drivable region,
  - maintaining lane alignment,
  - avoiding nearby vehicles/obstacles,
  - avoiding sidewalk/off-road/wrong-way behavior,
  - smooth and realistic motion,
  - following the route instruction as much as possible.

Important constraints:
- Use only what is visible in the image and candidate metadata.
- Do not invent hidden objects, maps, or traffic rules not supported by the scene.
- If multiple candidates are similar, choose the safest one.
- If all candidates are imperfect, choose the least risky one.

Route instruction:
{route_instruction}
{padms_section}
{extra_section}

Candidate trajectories:
{candidate_text}

Return ONLY valid JSON in this exact schema:
{{
  "best_candidate_index": <int>,
  "best_candidate_color": "<str>",
  "confidence": <float between 0 and 1>,
  "reasoning": {{
    "safety": "<primary safety considerations (obstacles, collisions, etc.)>",
    "lane_alignment": "<how well it stays in the lane and follows road geometry>",
    "confidence_justification": "<brief explanation of why this confidence score, not higher or lower>"
  }}
}}
""".strip()

    return prompt

def try_parse_json(text: str) -> Optional[Dict]:
    raw = text.strip()

    # direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # fenced code block parse
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # first {...} blob parse
    blob = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
    if blob:
        candidate = blob.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None

def build_messages(image_path: Path, prompt: str) -> List[Dict]:
    image_uri = f"file://{image_path.resolve()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages

class Qwen3TrajectoryScorer:
    """
    VLM trajectory scorer using Qwen3-VL-8B-Instruct.

    Scores trajectory candidates in images with reasoning.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 300,
    ):

        logger.info(f"Loading model {model_id} on device {device}...")
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto",
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        print(f"✓ Model loaded: {model_id}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")

    def _run_inference(self, messages: List[Dict]) -> str:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images, videos = process_vision_info(
            messages,
            image_patch_size=16,
        )

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            do_resize=False,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        trimmed_ids = []
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids):
            trimmed_ids.append(out_ids[len(in_ids):])

        output_text = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text

    def score_pair(
        self,
        image_path: Path,
        traj_path: Path,
        route_instruction: str = (
            "Drive safely and select the most reasonable forward trajectory "
            "that stays in the proper drivable area, avoids collisions, and "
            "follows lane geometry."
        ),
        padms_text: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> Dict:

        start_time = time.time()

        try:
            # Parse trajectory file
            candidates = parse_trajectory_txt(traj_path)
            if not candidates:
                logger.warning(f"No valid candidates found in {traj_path}")
                return {
                    "image_path": str(image_path),
                    "traj_path": str(traj_path),
                    "error": "No valid candidates parsed",
                    "num_candidates": 0,
                }

            candidate_rows = build_candidate_table(candidates)

            # Build prompt
            prompt = build_scoring_prompt(
                candidate_rows=candidate_rows,
                route_instruction=route_instruction,
                padms_text=padms_text,
                extra_context=extra_context,
                include_raw_points=False,
            )

            # Build messages and run inference
            messages = build_messages(image_path, prompt)
            raw_output = self._run_inference(messages)

            # Parse JSON
            parsed_output = try_parse_json(raw_output)

            elapsed = time.time() - start_time

            # Package result
            result = {
                "image_path": str(image_path),
                "traj_path": str(traj_path),
                "num_candidates": len(candidate_rows),
                "raw_output": raw_output,
                "parsed_output": parsed_output,
                "inference_time_sec": elapsed,
            }

            return result

        except Exception as e:
            logger.error(f"Error scoring pair {image_path}: {e}", exc_info=True)
            return {
                "image_path": str(image_path),
                "traj_path": str(traj_path),
                "error": str(e),
            }

    def batch_score(
        self,
        pairs: List[Dict],
        output_dir: Optional[Path] = None,
        checkpoint_every: int = 10,
        verbose: bool = True,
    ) -> List[Dict]:

        results = []

        # Create output dir if needed
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for i, pair in enumerate(pairs):
            if verbose:
                print(f"[{i+1}/{len(pairs)}] Scoring {pair['stem']}...", end=" ", flush=True)

            result = self.score_pair(
                image_path=pair["image_path"],
                traj_path=pair["traj_path"],
            )
            results.append(result)

            if verbose:
                if "error" in result:
                    print(f"ERROR: {result['error']}")
                else:
                    confidence = result.get("parsed_output", {}).get("confidence", "?")
                    print(f"✓ confidence={confidence}")

            # Checkpoint periodically
            if output_dir and (i + 1) % checkpoint_every == 0:
                checkpoint_path = output_dir / f"checkpoint_{i+1}.json"
                with open(checkpoint_path, "w") as f:
                    json.dump(results, f, indent=2)
                if verbose:
                    print(f"  → Checkpoint saved to {checkpoint_path.name}")

        return results
