# Paper

## Title:
**Class-Agnostic Action Segmentation in Construction Videos Based on Periodicity and Pose Dynamics: A Preliminary Study**

## Abstract:
Monitoring construction activities is essential for ensuring safety and quality on job sites. However, accurately segmenting continuous worker actions from video data is challenging due to the variability in human motion and the scarcity of annotated data. In this work, we propose a prototype framework that first obtains a coarse segmentation of construction activities and then refines these segments by leveraging periodicity detection and pose-based motion analysis. Our approach extracts action cues from video sequences and computes motion features that capture dynamics such as movement speed and joint behavior. These features are fused and grouped into context-aware action segments, enabling the differentiation of similar low-level actions across various tasks (e.g., distinguishing “holding” during masonry from that during plastering). Preliminary experiments using unsupervised clustering (k-means) lead to suboptimal segmentation granularity, prompting us to explore alternative clustering strategies. Further development and experiments are underway to enhance segmentation accuracy.

## Pipeline:

```mermaid
flowchart TD
    A[Input Video]
    B[Coarse Segmentation via Pretrained Action Recognition Model]
    C[Temporal Smoothing and Voting]
    D[Coarse Segments]
    E[Resample Video at Multiple Speeds]
    F[Process Through RepNet for Periodicity Detection]
    G[Cross-Verify Period Boundaries]
    H[Identify Fine Period Segments]
    I[Pose Estimation using Lightweight Model]
    J[Compute Motion Features: Position, Velocity, Acceleration, Joint Angles, Angular Velocities]
    K[Normalize Motion Features]
    L[Select Frames Corresponding to Period Segments]
    M[Uniformly Resample to Fixed Number of Frames]
    N[Extract Visual Features from Action Recognition Backbone]
    O[Fuse Visual Features with Pose-Based Feature Vectors]
    P[Unified Fixed-Size Feature Vectors]
    Q[Group Segments with Similar Coarse Labels]
    R[Unsupervised Clustering/Distance Metrics]
    S[Final Fine-Grained Segments]
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    A --> I
    H --> L
    I --> J
    J --> K
    K --> L
    L --> M
    M --> O
    B --> N
    N --> O
    O --> P
    D --> Q
    P --> R
    Q --> R
    R --> S

```
