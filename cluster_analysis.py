import pandas as pd
import cv2
import os
import random
import numpy as np

# Paths
df_path = 'results\\clustering_results_3s_4c.csv'
videos_path = 'splits_3s'

df = pd.read_csv(df_path)

label = int(input("Enter the cluster label (e.g., 0, 1, or 2): "))

filtered_videos = df[df["cluster_label"] == label]["video_name"].tolist()

if len(filtered_videos) < 4:
    print(f"Not enough videos for label {label}. Found only {len(filtered_videos)} videos.")
else:
    # Randomly select 6 videos
    selected_videos = random.sample(filtered_videos, 4)

    caps = [cv2.VideoCapture(os.path.join(videos_path, video)) for video in selected_videos]

    while True:
        frames = []

        for cap in caps:
            success, frame = cap.read()
            if not success:
                print(f"Error reading frame. Resetting video {cap}.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the start
                success, frame = cap.read()
            
            if success:  # Only resize if the frame is read successfully
                frame_resized = cv2.resize(frame, (300, 300))
                frames.append(frame_resized)
            else:
                print(f"Failed to read frame from video, skipping...")

        if len(frames) == 4:  # Ensure there are 6 frames to display
            row1 = np.hstack(frames[:2])  # First 3 frames
            row2 = np.hstack(frames[2:])  # Next 3 frames
            grid = np.vstack([row1, row2])  # Stack rows vertically

            cv2.imshow(f"Cluster {label} - Playing Videos (2x2)", grid)

        # Break loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()

    cv2.destroyAllWindows()

# Print video counts for each cluster
for i in range(6):
    print(f'for {i}th Label : {len(df[df["cluster_label"]==i])}')
