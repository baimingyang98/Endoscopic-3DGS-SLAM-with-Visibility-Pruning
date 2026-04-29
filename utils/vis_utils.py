"""Video creation utilities."""
import os
import glob

import cv2


def plot_video(image_folder: str, video_path: str = "", fps: int = 30):
    """
    Create an AVI video from a folder of PNG images.
    
    Args:
        image_folder: path to folder containing numbered .png files
        video_path: output video path (will add .avi if missing)
        fps: frames per second
    """
    if video_path == "":
        video_path = "./plot.avi"
    elif not video_path.endswith(".avi"):
        video_path += ".avi"

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    print(f"Saving video to {video_path}")

    all_image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not all_image_paths:
        print(f"No PNG images found in {image_folder}")
        return

    first_image = cv2.imread(all_image_paths[0])
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image_path in all_image_paths:
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            video.write(img)
        else:
            break

    video.release()
    cv2.destroyAllWindows()
