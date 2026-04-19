import cv2
import os
from pathlib import Path

def process_all_videos(repo_dir, output_folder, frame_skip=10):

    os.makedirs(output_folder, exist_ok=True)

    # rglob recursivelly searches for all .avi files
    video_paths = list(Path(repo_dir).rglob("*.avi")) 

    print(f"There are {len(video_paths)} videos in the repo.")

    total_saved_count = 0

    for video_path in video_paths:
        print(f"Processing: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))

        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_skip == 0:
                frame_filename = os.path.join(output_folder, f"{video_path.stem}_frame_{total_saved_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                total_saved_count += 1

            count +=1

        cap.release()

    print(f"Extracted {total_saved_count} frames to {output_folder}")

process_all_videos(
    repo_dir='.',
    output_folder='./dataset',
    frame_skip=20
)




    

