import glob
import os
import random

DATA_BASE = "/Documents/vendor_triplet"
TRAIN_RATIO = 0.95

if __name__ == "__main__":
    end_subfolders = []

    for root, dirs, files in os.walk(DATA_BASE):
        # Check if there are no subdirectories
        if not dirs and root != DATA_BASE:  # Exclude the root folder itself
            end_subfolders.append(root)

    png_files = glob.glob(f"{DATA_BASE}/**/*.png", recursive=True)
    assert len(png_files) == 3 * len(end_subfolders)

    num_training = int(len(end_subfolders) * TRAIN_RATIO)
    training_paths = random.sample(end_subfolders, num_training)
    testing_paths = [path for path in end_subfolders if path not in training_paths]

    # Define file paths to save the lists
    training_txt = os.path.join(DATA_BASE, "tri_trainlist.txt")
    testing_txt = os.path.join(DATA_BASE, "tri_testlist.txt")

    # Save random paths to a file
    with open(training_txt, "w") as f:
        for path in training_paths:
            f.write(f"{path}\n")

    # Save remaining paths to a file
    with open(testing_txt, "w") as f:
        for path in testing_paths:
            f.write(f"{path}\n")

    print("Num training:", len(training_paths), "saved to", training_txt)
    print("Num testing:", len(testing_paths), "saved to", testing_txt)
