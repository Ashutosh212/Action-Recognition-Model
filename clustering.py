# folder_path = 'results'

# print(len(os.listdir(folder_path)))


# for files in os.listdir(folder_path):
#     path = os.path.join(folder_path, files)
#     loaded_data = np.load(path, allow_pickle=True).item()
#     video_name = loaded_data["video_name"]
#     pred_score = loaded_data["pred_score"]
#     print(video_name)
#     print(pred_score)


import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Folder containing the saved .npy files
folder_path = 'results\\3s embd'

video_names = []
pred_scores = []

# Load all .npy files
for file in os.listdir(folder_path):
    if file.endswith(".npy"):  # Ensure only .npy files are processed
        path = os.path.join(folder_path, file)
        loaded_data = np.load(path, allow_pickle=True).item()
        video_name = loaded_data["video_name"]
        pred_score = loaded_data["pred_score"]
        
        video_names.append(video_name)
        pred_scores.append(pred_score)

pred_scores_np = np.array(pred_scores)

num_clusters = 4

# Train K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(pred_scores_np)

# Get cluster assignments
cluster_labels = kmeans.labels_

# Save results to a CSV file
results_df = pd.DataFrame({
    "video_name": video_names,
    "cluster_label": cluster_labels
})

# Save to a CSV file
results_csv_path = "clustering_results_3s_4c.csv"
results_df.to_csv(results_csv_path, index=False)

print(f"Clustering results saved to {results_csv_path}")
