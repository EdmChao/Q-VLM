#!/usr/bin/env python3
"""
Small utility to inspect a subscore pickle and extract a single trajectory as a numpy array.
Usage examples:
  python scripts/extract_single_traj.py --subscore /path/to/subscore.pkl --list
  python scripts/extract_single_traj.py --subscore /path/to/subscore.pkl --token <token> --show
  python scripts/extract_single_traj.py --subscore /path/to/subscore.pkl --token <token> --out out.npy
"""
import argparse
import pickle
from pathlib import Path
import numpy as np
import sys


def to_numpy(x):
    try:
        import torch
    except Exception:
        torch = None
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and hasattr(x, "cpu"):
        try:
            return x.cpu().numpy()
        except Exception:
            pass
    # dataclass with .poses
    if hasattr(x, "poses"):
        return np.asarray(getattr(x, "poses"))
    # try to convert generic sequence
    try:
        return np.asarray(x)
    except Exception:
        return None


def inspect_subscore(path: Path, token: str = None, list_only: bool = False, out: Path = None, show: bool = False):
    if not path.exists():
        print(f"ERROR: subscore file not found: {path}")
        return 2
    with open(path, "rb") as f:
        data = pickle.load(f)

    print("Top-level type:", type(data))
    if isinstance(data, dict):
        print("Number of tokens:", len(data))
        if len(data) == 0:
            print("No entries in pickle.")
            return 0
        if list_only:
            # show first 20 tokens
            keys = list(data.keys())
            for k in keys[:min(200, len(keys))]:
                print(k)
            return 0
        if token is None:
            token = next(iter(data))
            print("No token provided; using first token:", token)
        if token not in data:
            print(f"Token '{token}' not found. Available tokens: (first 20)")
            print(list(data.keys())[:20])
            return 2
        entry = data[token]
        print("Entry type:", type(entry))
        if isinstance(entry, dict):
            print("Keys and types:")
            for k, v in entry.items():
                print(" -", k, "->", type(v))
        else:
            print("Entry is not a dict; type:", type(entry))

        # try common trajectory fields
        cand_fields = [
            "trajectory",
            "proposals",
            "dp_pred",
            "interpolated_proposal",
            "interpolated_proposals",
            "proposals_interpolated",
            "proposals_np",
        ]
        found = None
        if isinstance(entry, dict):
            for f in cand_fields:
                if f in entry:
                    found = (f, entry[f])
                    break
        else:
            # entry itself might be a trajectory-like
            found = ("entry", entry)

        if found is None:
            print("No standard trajectory fields found; attempting to convert whole entry to numpy...")
            arr = to_numpy(entry)
            if arr is None:
                print("Could not convert entry to numpy array.")
                return 0
            print("Converted entry -> numpy shape:", arr.shape)
            if out:
                np.save(out, arr)
                print("Saved to:", out)
            else:
                print(arr)
            return 0

        name, val = found
        print(f"Using field '{name}' of type {type(val)}")
        arr = to_numpy(val)
        if arr is None:
            # if it's a list/dict of proposals, try deeper
            if isinstance(val, dict):
                print("Field is a dict; keys:", list(val.keys())[:20])
                # try to find numeric subfield
                for kk, vv in val.items():
                    aa = to_numpy(vv)
                    if aa is not None:
                        arr = aa
                        print("Found numeric subfield:", kk)
                        break
        if arr is None:
            print("Failed to convert field to numpy array.")
            return 0

        print("Resulting numpy shape:", arr.shape)
        # print a small preview
        np.set_printoptions(precision=3, suppress=True)
        if arr.size <= 200:
            print(arr)
        else:
            # print first 3 and last 1 rows
            try:
                if arr.ndim >= 2:
                    print("preview (first 3 rows):\n", arr[:3])
                else:
                    print("preview (first 100 elems):\n", arr.ravel()[:100])
            except Exception:
                print(arr.ravel()[:100])

        if out:
            np.save(out, arr)
            print("Saved trajectory to:", out)
        return 0

    else:
        print("Top-level object is not a dict. Attempting to convert entire object to numpy...")
        arr = to_numpy(data)
        if arr is None:
            print("Could not convert file contents to numpy.")
            return 1
        print("Converted to numpy shape:", arr.shape)
        if out:
            np.save(out, arr)
            print("Saved to:", out)
        else:
            print(arr)
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscore', '-s', type=str, required=True, help='Path to subscore pickle')
    parser.add_argument('--token', '-t', type=str, help='Token to inspect (default: first token)')
    parser.add_argument('--list', action='store_true', help='List tokens and exit')
    parser.add_argument('--out', '-o', type=str, help='Optional output .npy path to save extracted numpy array')
    parser.add_argument('--show', action='store_true', help='Show detailed entry keys/types')
    args = parser.parse_args()
    rc = inspect_subscore(Path(args.subscore), token=args.token, list_only=args.list, out=Path(args.out) if args.out else None, show=args.show)
    sys.exit(rc)
