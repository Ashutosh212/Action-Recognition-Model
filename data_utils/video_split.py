import moviepy.editor as mp
import os

def split_video(input_video_path, output_dir, clip_duration=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video = mp.VideoFileClip(input_video_path)

    total_duration = video.duration

    clip_number = 1
    for start_time in range(0, int(total_duration), clip_duration):
        end_time = min(start_time + clip_duration, total_duration)
        
        clip = video.subclip(start_time, end_time)

        output_clip_path = os.path.join(output_dir, f"clip_{clip_number}.mp4")
        clip.write_videofile(output_clip_path, codec="libx264", audio_codec="aac")

        print(f"Saved clip {clip_number} to {output_clip_path}")
        clip_number += 1

    video.close()

input_video_path = "..\\test_videos\\FMWK_4.mp4"  
output_dir = "..\\splits_3s"  

split_video(input_video_path, output_dir)
