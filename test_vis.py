import os
import cv2
import glob
import numpy as np

# Define root directory where subfolders exist
root_dir = "checkpoint/resgroups_3_resblocks_6"  # Change this to your base directory

# Find all subfolders
sequence_id = "00005"
frame_id = "0011"
subfolders = glob.glob(
    os.path.join(root_dir, "**", sequence_id, frame_id), recursive=True
)

# Video settings
fps = 5
output_video_path = os.path.join(root_dir, sequence_id + "_" + frame_id + ".mp4")
frame_size = None  # To be determined from first image

# Initialize video writer (will be set later)
video_writer = None
prepared_frames = []

# Add text to image (top-left corner)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0)  # Green color
thickness = 2
position = (10, 30)  # Top-left corner

# Iterate over each subfolder
for subfolder in sorted(subfolders):
    print(f"Processing folder: {subfolder}")

    # Read the corresponding results.txt
    results_txt_path = os.path.join(subfolder, "..", "..", "results.txt")
    if os.path.exists(results_txt_path):
        with open(results_txt_path, "r") as f:
            result_texts = f.readlines()
            epoch = result_texts[0].strip().replace("For ", "")  # Extract "epoch=0"
            metrics = (
                result_texts[1].strip().split(", ")[:2]
            )  # Extract only "PSNR" and "SSIM"
            result_texts = f"{epoch} {metrics[0]} {metrics[1]}"  # Merge into one line
            print(result_texts)
    else:
        raise Exception("Error! Not found: {}".format(results_txt_path))

    # Process images
    image_path = os.path.join(subfolder, "im2.png")
    img = cv2.imread(image_path)

    if img is None:
        raise Exception("Error! Not found: {}".format(image_path))
    img_with_text = img.copy()
    cv2.putText(
        img_with_text,
        result_texts,
        position,
        font,
        font_scale,
        font_color,
        thickness,
        cv2.LINE_AA,
    )
    prepared_frames.append(img_with_text)

# Save to video
for i, img in enumerate(prepared_frames):

    # Set frame size for the video (only once)
    if frame_size is None:
        frame_size = (img.shape[1], img.shape[0])  # (width, height)

        # Initialize video writer when frame size is known
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Write frame to video
    video_writer.write(img)

# Release video writer
if video_writer:
    video_writer.release()
    print(f"Video saved: {output_video_path}")

# Show final video (optional)
cv2.destroyAllWindows()
