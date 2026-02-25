from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import os
import sys
import json
from pathlib import Path
import numpy as np

# default: Load the model on the available device(s)
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-8B-Thinking", dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Thinking",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")

out_dir = os.getenv('NAVSIM_EXP_ROOT')
k = 5

if out_dir is None:
    user_input = input("Enter path to experiment root (or press Enter to exit): ").strip()
    if not user_input:
        print("Error: No path provided. Exiting.")
        sys.exit(1)
    base_dir = Path(user_input)
else:
    base_dir = Path(out_dir)

data_dir = base_dir / f"{k}_proposals"
if not data_dir.exists():
    print(f"Error: data directory {data_dir} does not exist. Exiting.")
    sys.exit(1)

# Discover text and image files (search recursively)
text_exts = {'.txt', '.json', '.csv', '.npy', '.npz'}
img_exts = {'.png', '.jpg', '.jpeg'}
texts = {}
images = {}
for p in data_dir.rglob('*'):
    if not p.is_file():
        continue
    ext = p.suffix.lower()
    if ext in text_exts:
        texts[p.stem] = p
    if ext in img_exts:
        images[p.stem] = p

# Build pairs by matching stems when possible, otherwise try positional pairing
pairs = []
common = sorted(set(texts.keys()) & set(images.keys()))
if common:
    for stem in common:
        pairs.append((texts[stem], images[stem]))
else:
    if texts and images and len(texts) == len(images):
        texts_sorted = sorted(texts.values())
        imgs_sorted = sorted(images.values())
        pairs = list(zip(texts_sorted, imgs_sorted))
    else:
        print("Error: Could not find matching text-image pairs in data directory.\n"
              "Make sure each trajectory text file has a corresponding image with the same stem,\n"
              "or an equal number of text and image files for positional pairing.")
        sys.exit(1)

def load_text_file(p: Path):
    ext = p.suffix.lower()
    try:
        if ext in ('.txt', '.csv'):
            return p.read_text()
        if ext == '.json':
            return json.loads(p.read_text())
        if ext in ('.npy', '.npz'):
            arr = np.load(p, allow_pickle=True)
            try:
                return arr.tolist()
            except Exception:
                return str(arr)
    except Exception as e:
        return f"<failed to load {p}: {e}>"
    return p.read_text()

# Loop inference over all found pairs
for text_path, image_path in pairs:
    print(f"Processing pair: {text_path.name}  <->  {image_path.name}")
    text_trajectories = load_text_file(text_path)
    # Ensure text is string for injection into prompt
    if not isinstance(text_trajectories, str):
        try:
            text_for_prompt = json.dumps(text_trajectories)
        except Exception:
            text_for_prompt = str(text_trajectories)
    else:
        text_for_prompt = text_trajectories

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text":
                 f"Choose the best trajectory based on this list and the provided image: {text_for_prompt}. "
                 "Give a short 1 sentence reasoning on why the chosen trajectory was the best, "
                 "and return selected trajectory in the JSON format {color: traj_color, trajectory: traj_coords}"},
            ],
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
    except Exception as e:
        print(f"Inference failed for pair {text_path.name} / {image_path.name}: {e}")
