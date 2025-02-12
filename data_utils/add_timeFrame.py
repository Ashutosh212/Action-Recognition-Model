# import pandas as pd

# csv_path = "..\\results\\clustering_results_5s_3c.csv"
# output_path = "..\\results\\sorted_clustering_results.csv"  # Specify the output path for the sorted CSV

# def add_tf(time_in_sec, path):
#     # Load the CSV data
#     df = pd.read_csv(path)
    
#     # Sort the DataFrame by the numeric part of the video_name
#     df['video_num'] = df['video_name'].str.extract(r'(\d+)').astype(int)
#     df = df.sort_values(by='video_num')
#     df = df.drop(columns=['video_num'])  # Drop the temporary 'video_num' column
    
#     # Save the sorted DataFrame to a new CSV
#     df.to_csv(output_path, index=False)
    
#     # Reload the sorted CSV file
#     df_sorted = pd.read_csv(output_path)
    
#     # Create a new column 'time_stamp' in the DataFrame
#     time_stamps = []
    
#     # Adjusted time interval to 3 seconds
#     time_interval = 3  # Using 3 seconds per video clip
    
#     for i, row in df_sorted.iterrows():
#         start_time = i * time_interval
#         end_time = (i + 1) * time_interval
#         start_time_formatted = f"{start_time // 60:02}:{start_time % 60:02}"
#         end_time_formatted = f"{end_time // 60:02}:{end_time % 60:02}"
#         time_stamps.append(f"{start_time_formatted} to {end_time_formatted}")
    
#     # Add the new 'time_stamp' column to the dataframe
#     df_sorted['time_stamp'] = time_stamps
    
#     # Print the first few rows to check
#     # print(df_sorted.head(20))
#     df_sorted = pd.read_csv(output_path)
#     return df_sorted

# # Calling the function to add the time stamps
# df_with_time = add_tf(3, csv_path)

# # Optionally, save the final DataFrame with time stamps to a new CSV
# final_output_path = "..\\results\\{csv_path}_time"
# df_with_time.to_csv(final_output_path, index=False)


import os
import pandas as pd

csv_path = "..\\results\\clustering_results_5s_6c.csv"

def add_tf(time_in_sec, path):
    # Load the CSV data
    df = pd.read_csv(path)
    
    # Sort the DataFrame by the numeric part of the video_name
    df['video_num'] = df['video_name'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values(by='video_num')
    df = df.drop(columns=['video_num'])  # Drop the temporary 'video_num' column
    
    # Save the sorted DataFrame to a new CSV
    output_path = path.replace('.csv', '_sorted.csv')
    df.to_csv(output_path, index=False)
    
    # Reload the sorted CSV file
    df_sorted = pd.read_csv(output_path)
    
    # Create a new column 'time_stamp' in the DataFrame
    time_stamps = []
    
    # Adjusted time interval to 3 seconds
    time_interval = time_in_sec  # Using 3 seconds per video clip
    
    for i, row in df_sorted.iterrows():
        start_time = i * time_interval
        end_time = (i + 1) * time_interval
        start_time_formatted = f"{start_time // 60:02}:{start_time % 60:02}"
        end_time_formatted = f"{end_time // 60:02}:{end_time % 60:02}"
        time_stamps.append(f"{start_time_formatted} to {end_time_formatted}")
    
    # Add the new 'time_stamp' column to the dataframe
    df_sorted['time_stamp'] = time_stamps
    
    # Print the first few rows to check
    print(df_sorted.head(20))
    
    # Save the final DataFrame with time stamps using dynamic path
    final_output_path = os.path.join(os.path.dirname(path), f"{os.path.splitext(os.path.basename(path))[0]}_time.csv")
    df_sorted.to_csv(final_output_path, index=False)
    
    return df_sorted

# Calling the function to add the time stamps
df_with_time = add_tf(5, csv_path)

# Optionally, print the final output path to verify
print(f"Final file saved at: final_output_path")
