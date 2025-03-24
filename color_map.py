import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def plot_cluster_bar(csv_file):
    # Load CSV file
    df = pd.read_csv(csv_file)
    # print(df.head())
    time_segments = df['time_stamp'].values
    clusters = df['cluster_label'].values

    
    unique_clusters = np.unique(clusters)
    # colors = plt.cm.get_cmap("tab10", len(unique_clusters))
    colors = plt.get_cmap("plasma", len(unique_clusters))

    cluster_color_map = {cluster: colors(i) for i, cluster in enumerate(unique_clusters)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Draw the color bar
    for i, cluster in enumerate(clusters):
        ax.barh(y=0, width=1, left=i, height=0.3, color=cluster_color_map[cluster])
    
    # Set time labels
    ax.set_xticks(range(len(time_segments)))
    # ax.set_xticklabels(time_segments, rotation=90)
    ax.set_yticks([])
    
    # Add legend
    patches = [mpatches.Patch(color=cluster_color_map[cl], label=f'Cluster {cl}') for cl in unique_clusters]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 0.5), loc='upper right')
    
    plt.title("Cluster Visualization Over Time")
    plt.xlabel("Time Segments")
    plt.show()

# Example usage
csv_path = "results\\clustering_results_5s_6c_time.csv"

plot_cluster_bar(csv_path)
