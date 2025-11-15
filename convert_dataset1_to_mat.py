import os
import numpy as np
from scipy.io import savemat

BASE = "/home/ialzouby/Learning/Gait-Recognition-Using-Smartphones/data/Dataset1try/Dataset #1"
OUT  = "/home/ialzouby/Learning/Gait-Recognition-Using-Smartphones/data/data118"

def load_split(split):
    """Load 6 IMU axes as arrays of shape (N, 128), stack to (N, 6, 128)."""
    sig_path = os.path.join(BASE, split, "Inertial Signals")
    axes = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

    mats = []
    for axis in axes:
        file_path = os.path.join(sig_path, f"{split}_{axis}.txt")
        arr = np.loadtxt(file_path)      # shape: (N, 128)
        mats.append(arr)

    # Stack into (6, N, 128) → then transpose to (N, 6, 128)
    stacked = np.stack(mats, axis=0)
    stacked = np.transpose(stacked, (1, 0, 2))

    return stacked  # (N, 6, 128)


def process_split(split):
    print(f"Processing {split}...")

    # Load data
    X = load_split(split)
    y = np.loadtxt(os.path.join(BASE, split, f"y_{split}.txt")).astype(int)

    # Output path
    out_dir = os.path.join(OUT, split, "record")
    os.makedirs(out_dir, exist_ok=True)

    # Save each sample as .mat
    for i in range(X.shape[0]):
        sample = X[i]         # shape (6, 128)
        label  = y[i]
        idx    = i + 1

        savemat(os.path.join(out_dir, f"{idx:06d}.mat"), {
            "data": sample,
            "label": label,
            "id": idx
        })

    print(f"✔ {split} complete: {X.shape[0]} samples saved.")


def main():
    process_split("train")
    process_split("test")


if __name__ == "__main__":
    main()
