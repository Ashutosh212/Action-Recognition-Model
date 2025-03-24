import numpy as np
import json
import os
import pandas as pd
from sklearn.cluster import KMeans


mae_embed_path = "results\\3s_embed_prev"
pose_embed_path = "results\\3s_embed_pose"
output_folder = "results\\concat_embed"


if not os.listdir(output_folder):
    mae_embs = os.listdir(mae_embed_path)
    pose_embs = os.listdir(pose_embed_path)

    for emd1, emd2 in zip(mae_embs, pose_embs):
        filename = os.path.splitext(emd2)[0]

        emb1_path = os.path.join(mae_embed_path, emd1)
        emb2_path = os.path.join(pose_embed_path, emd2)
        emb1_np = np.load(emb1_path)
        
        with open(emb2_path, 'r') as f:
            emb2_js = json.load(f)

        normalized_keypoints = emb2_js[0]['normalized_keypoint']
        keypoint_scores = emb2_js[1]['keypoint_scores']

        x_coords = []
        y_coords = []
        scores = []

        # Extract x, y, and scores for each keypoint
        for keypoint, score in zip(normalized_keypoints, keypoint_scores):
            x_coords.append(keypoint[0])  # x-coordinate
            y_coords.append(keypoint[1])  # y-coordinate
            scores.append(score)          
        combined_features = x_coords + y_coords + scores
        emb2_array = np.array(combined_features)

        emb2_array = emb2_array.reshape(1, 51)
        emb2_array = np.tile(emb2_array, (15, 1))

        # Concatenate along axis 1 (columns)
        result = np.concatenate((emb1_np, emb2_array), axis=1)

        output_file = os.path.join(output_folder, filename)
        np.save(f'{output_file}.npy', result)
        print(f"Processed: {filename}, Result shape: {result.shape}")
else:
    print("Output directory is not empty. Skipping processing.")


csv_output_path = "results\\clustering_results_concat_3s_4c.csv"

npy_files = [f for f in os.listdir(output_folder) if f.endswith('.npy')]

file_names = []
features = []

for npy_file in npy_files:
    npy_file_path = os.path.join(output_folder, npy_file)
    emb = np.load(npy_file_path)  
    emb_flat = emb.flatten()  # Flatten the array to 1D (15 * 385 = 5775)
    features.append(emb_flat) 
    file_names.append(npy_file)  

features_array = np.array(features)
print(features_array.shape)
num_clusters = 4 

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(features_array)

cluster_labels = kmeans.labels_
print(cluster_labels)

df = pd.DataFrame({
    'video_name': file_names,
    'cluster_label': cluster_labels
})

df.to_csv(csv_output_path, index=False)

print(f"K-means clustering completed. Results saved in {csv_output_path}.")
