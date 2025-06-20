import argparse
import cv2
import numpy as np
from src.preprocessing import preprocess_image, generate_nail_positions

def greedy_string_art(target_img: np.ndarray, nails: np.ndarray, steps: int):
    # Prepare darkness map: higher values where we need more thread
    darkness = 255 - target_img.copy()
    sequence = []

    h, w = darkness.shape
    for s in range(steps):
        best_score = -1
        best_pair = None
        best_mask_idxs = None

        # Try every pair of nails
        for i in range(len(nails)):
            x0, y0 = nails[i]
            for j in range(i+1, len(nails)):
                x1, y1 = nails[j]
                # Create a mask for this line
                mask = np.zeros_like(darkness, dtype=np.uint8)
                cv2.line(mask, (x0, y0), (x1, y1), 1, 1)
                ys, xs = mask.nonzero()
                score = darkness[ys, xs].sum()
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
                    best_mask_idxs = (ys, xs)

        if best_pair is None or best_score <= 0:
            break

        # Record and erase darkness where we’ve placed thread
        sequence.append(best_pair)
        ys, xs = best_mask_idxs
        darkness[ys, xs] = 0

        print(f"Step {s+1}/{steps}: nails {best_pair} score={best_score}")

    return sequence

def save_sequence(seq, out_path):
    with open(out_path, "w") as f:
        for a, b in seq:
            f.write(f"{a},{b}\\n")

def main():
    parser = argparse.ArgumentParser(description="Greedy String-Art Baseline")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--nails", type=int, default=200, help="Number of nails")
    parser.add_argument("--steps", type=int, default=1000, help="String steps")
    parser.add_argument("--output", default="sequence.txt", help="Output sequence file")
    args = parser.parse_args()

    target = preprocess_image(args.input)
    nails = generate_nail_positions(args.nails, target.shape[0])
    seq = greedy_string_art(target, nails, args.steps)
    save_sequence(seq, args.output)

if __name__ == "__main__":
    main()
