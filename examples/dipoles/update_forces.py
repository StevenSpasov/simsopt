import os
import json
import numpy as np
from simsopt._core.optimizable import load
from simsopt.field.force import coil_force, regularization_circ

def compute_forces(directory, a=1.0):  # Adjust 'a' as needed
    """
    Loop through all subdirectories of `directory`, compute forces, and update results.json.
    """
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if not os.path.isdir(subdir_path):
            continue  # Skip if not a directory

        bs_opt_path = os.path.join(subdir_path, "bs_opt.json")
        results_path = os.path.join(subdir_path, "results.json")

        if not os.path.exists(bs_opt_path):
            print(f"Skipping {subdir}: bs_opt.json not found.")
            continue

        try:
            bs = load(bs_opt_path)
            coils = bs.coils

            max_forces = [
                np.max(np.linalg.norm(coil_force(c, coils, regularization_circ(a)), axis=1))
                for c in coils
            ]
            min_forces = [
                np.min(np.linalg.norm(coil_force(c, coils, regularization_circ(a)), axis=1))
                for c in coils
            ]
            RMS_forces = [
                np.sqrt(np.mean(np.square(np.linalg.norm(coil_force(c, coils, regularization_circ(a)), axis=1))))
                for c in coils
            ]

            new_data = {
                "max_max_force": max(float(f) for f in max_forces),
                "min_min_force": min(float(f) for f in min_forces),
                "mean_RMS_force": float(np.mean(RMS_forces)),
            }

            # Load existing results.json and update it
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    results = json.load(f)
            else:
                results = {}

            results.update(new_data)

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            print(f"Updated {results_path}")

        except Exception as e:
            print(f"Error processing {subdir}: {e}")

if __name__ == "__main__":
    directory = "/Users/jakehalpern/Projects/C-REX/outputs/20250211_Bt_0.5_unfixed_TFs"  # Change this to the actual directory path
    compute_forces(directory)