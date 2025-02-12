import cv2
import pandas as pd

csv_path = "..\\results\\clustering_results_3s_4c_time.csv"

video_file_path = "..\\test_videos\\FMWK_4.mp4"

def display_video_with_label(csv_path, video_file_path):
    df = pd.read_csv(csv_path)

    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    thickness = 2
    position = (10, 30)  # Position on the top-left corner

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 3 * fps  # 3-second interval based on fps
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        # Get the current frame number
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        for i, row in df.iterrows():
            start_time_str, end_time_str = row['time_stamp'].split(" to ")
            start_min, start_sec = map(int, start_time_str.split(":"))
            end_min, end_sec = map(int, end_time_str.split(":"))
            
            # Convert the time to seconds
            start_time_sec = start_min * 60 + start_sec
            end_time_sec = end_min * 60 + end_sec
            
            # Convert the time range to frames based on fps
            start_frame = int(start_time_sec * fps)
            end_frame = int(end_time_sec * fps)
            
            # Check if the current frame is within the time period
            if start_frame <= current_frame < end_frame:
                cluster_label = row['cluster_label']
                label_text = f"Cluster: {cluster_label}"
                break  # We found the corresponding cluster label for this frame

        # Overlay the cluster label on the video frame
        cv2.putText(frame, label_text, position, font, font_scale, font_color, thickness)

        # Display the frame with the label
        cv2.imshow('Video with Cluster Label', frame)

        # Wait for the keypress to close the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

# Call the function to display the video with the cluster label
display_video_with_label(csv_path, video_file_path)
