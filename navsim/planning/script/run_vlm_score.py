#!/usr/bin/env python
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm

from vlm_scoring.qwen3_trajectory_scorer import (
    Qwen3TrajectoryScorer,
    discover_pairs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_statistics(results: List[Dict]) -> Dict:
    confidences = []

    for result in results:
        if "error" in result:
            continue

        parsed = result.get("parsed_output", {})
        if parsed and "confidence" in parsed:
            confidences.append(parsed["confidence"])

    stats = {
        "num_successful": len([r for r in results if "error" not in r]),
        "num_errors": len([r for r in results if "error" in r]),
    }

    if confidences:
        mean_conf = sum(confidences) / len(confidences)
        stats["confidence"] = {
            "mean": mean_conf,
            "min": min(confidences),
            "max": max(confidences),
            "std": (sum((x - mean_conf) ** 2 for x in confidences) / len(confidences)) ** 0.5
            if len(confidences) > 1
            else 0,
        }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate trajectories with Qwen3-VL-8B-Instruct"
    )
    parser.add_argument(
        "--candidate-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Directories containing trajectory overlays (PNG + TXT pairs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for VLM scoring results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max number of samples to process (default: all)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save checkpoint every N samples (default: 5)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model ID from Hugging Face",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--route-instruction",
        type=str,
        default=(
            "Drive safely and select the most reasonable forward trajectory "
            "that stays in the proper drivable area, avoids collisions, and "
            "follows lane geometry."
        ),
        help="Route instruction for the scorer",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if output file already exists",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "results.json"
    if args.skip_existing and output_file.exists():
        logger.info(f"Output file exists, skipping: {output_file}")
        return

    # Discover pairs
    logger.info(f"Discovering pairs in {args.candidate_dirs}...")
    pairs = discover_pairs(args.candidate_dirs)
    logger.info(f"Found {len(pairs)} pairs")

    if not pairs:
        logger.error("No pairs found!")
        return

    # Limit samples
    if args.max_samples:
        pairs = pairs[:args.max_samples]
        logger.info(f"Limiting to {len(pairs)} samples")

    # Initialize scorer
    logger.info(f"Initializing scorer with model {args.model_id}...")
    scorer = Qwen3TrajectoryScorer(
        model_id=args.model_id,
        device=args.device,
    )

    # Run batch scoring
    logger.info(f"Starting batch evaluation of {len(pairs)} pairs...")
    results = []
    start_time = datetime.now()

    for i, pair in enumerate(tqdm(pairs, desc="Scoring")):
        try:
            result = scorer.score_pair(
                image_path=pair["image_path"],
                traj_path=pair["traj_path"],
                route_instruction=args.route_instruction,
            )
            results.append(result)

            # Save checkpoint
            if (i + 1) % args.checkpoint_every == 0:
                checkpoint_file = output_dir / f"checkpoint_{i+1}.json"
                with open(checkpoint_file, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Checkpoint saved: {checkpoint_file}")

        except Exception as e:
            logger.error(f"Error processing pair {i}: {e}", exc_info=True)
            results.append({
                "stem": pair.get("stem", "unknown"),
                "image_path": str(pair["image_path"]),
                "traj_path": str(pair["traj_path"]),
                "error": str(e),
            })

    elapsed = datetime.now() - start_time

    # Compute statistics
    logger.info("Computing statistics...")
    statistics = compute_statistics(results)

    # Prepare final output
    output_data = {
        "metadata": {
            "timestamp": start_time.isoformat(),
            "model_id": args.model_id,
            "num_pairs": len(pairs),
            "num_results": len(results),
            "elapsed_seconds": elapsed.total_seconds(),
            "candidate_dirs": [str(d) for d in args.candidate_dirs],
        },
        "statistics": statistics,
        "results": results,
    }

    # Save results
    logger.info(f"Saving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Pairs processed: {len(results)}")
    print(f"Successful: {statistics.get('num_successful', 0)}")
    print(f"Errors: {statistics.get('num_errors', 0)}")
    print(f"Total time: {elapsed}")

    if "confidence" in statistics:
        conf = statistics["confidence"]
        print(f"\nConfidence scores:")
        print(f"  Mean: {conf['mean']:.3f} ± {conf['std']:.3f}")
        print(f"  Range: [{conf['min']:.3f}, {conf['max']:.3f}]")

    if "risk_flags" in statistics:
        print(f"\nTop risk flags:")
        for risk, count in list(statistics["risk_flags"].items())[:5]:
            print(f"  {risk}: {count}")

    print(f"\nResults saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
