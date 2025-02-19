# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.patches as mpatches

# def plot_cluster_bar(csv_file):
#     # Load CSV file
#     df = pd.read_csv(csv_file)
#     # print(df.head())
#     # Extract cluster labels and time segments
#     time_segments = df['time_stamp'].values
#     clusters = df['cluster_label'].values

    
#     # Define unique colors for clusters
#     unique_clusters = np.unique(clusters)
#     colors = plt.cm.get_cmap("tab10", len(unique_clusters))
#     cluster_color_map = {cluster: colors(i) for i, cluster in enumerate(unique_clusters)}
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(15, 4))
    
#     # Draw the color bar
#     for i, cluster in enumerate(clusters):
#         ax.barh(y=0, width=1, left=i, color=cluster_color_map[cluster])
    
#     # Set time labels
#     ax.set_xticks(range(len(time_segments)))
#     # ax.set_xticklabels(time_segments, rotation=90)
#     ax.set_yticks([])
    
#     # Add legend
#     patches = [mpatches.Patch(color=cluster_color_map[cl], label=f'Cluster {cl}') for cl in unique_clusters]
#     ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     plt.title("Cluster Visualization Over Time")
#     plt.xlabel("Time Segments")
#     plt.show()

# Example usage
# csv_path = "results\\clustering_results_3s_4c_time.csv"

# plot_cluster_bar(csv_path)


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_3d_segmented_bar():
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111, projection='3d')
    
    num_segments = 6  # Number of segments
    colors = ['yellow', 'green', 'purple', 'blue', 'pink', 'gray']  # Colors for each segment
    labels = [f'Action {i+1}' for i in range(num_segments)]
    
    for i in range(num_segments):
        x = [i, i+1, i+1, i]
        y = [0, 0, 1, 1]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]
        
        face = Poly3DCollection(verts, color=colors[i], alpha=0.7, edgecolor='black')
        ax.add_collection3d(face)
        
        ax.text(i + 0.5, 0.5, 0.05, labels[i], color='black', fontsize=10, ha='center')
    
    ax.set_xlim(0, num_segments)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 0.5)
    ax.axis('off')
    
    plt.show()

# Call the function to draw the 3D segmented bar
draw_3d_segmented_bar()
