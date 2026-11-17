"""
=============================================================================
AeroVision AI - Object Tracking Foundations: Module 02
=============================================================================
Data Association & The Hungarian Matching Algorithm

Learning Objectives:
1. Understand the bipartite matching problem in Multi-Object Tracking (MOT).
2. Construct cost matrices using IoU spatial distance (1 - IoU).
3. Solve linear assignment using the Hungarian (Munkres) Algorithm.
4. Partition assignment results into Matched Pairs, Unmatched Tracks, and Unmatched Detections.
5. Contrast global optimal matching with greedy nearest-neighbor assignment.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict

def compute_iou_cost_matrix(tracks_boxes: np.ndarray, detections_boxes: np.ndarray) -> np.ndarray:
    """
    Constructs an N x M cost matrix where cost(i, j) = 1.0 - IoU(track_i, det_j).
    Lower cost indicates closer spatial proximity and better match.
    """
    n = len(tracks_boxes)
    m = len(detections_boxes)
    cost_matrix = np.ones((n, m), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            tb = tracks_boxes[i]
            db = detections_boxes[j]
            
            x1 = max(tb[0], db[0])
            y1 = max(tb[1], db[1])
            x2 = min(tb[2], db[2])
            y2 = min(tb[3], db[3])
            
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_t = (tb[2] - tb[0]) * (tb[3] - tb[1])
            area_d = (db[2] - db[0]) * (db[3] - db[1])
            union = area_t + area_d - inter
            
            iou = inter / union if union > 0 else 0.0
            cost_matrix[i, j] = 1.0 - iou

    return cost_matrix

def associate_detections_to_tracks(
    tracks_boxes: np.ndarray,
    detections_boxes: np.ndarray,
    max_cost_threshold: float = 0.70  # Min IoU of 0.30 required to match
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Performs optimal bipartite matching using the Hungarian algorithm.
    
    Returns:
        matches: List of (track_idx, det_idx) pairs.
        unmatched_tracks: List of track indices with no valid detection match.
        unmatched_detections: List of detection indices representing newly entering objects.
    """
    if len(tracks_boxes) == 0:
        return [], [], list(range(len(detections_boxes)))
    if len(detections_boxes) == 0:
        return [], list(range(len(tracks_boxes))), []

    cost_matrix = compute_iou_cost_matrix(tracks_boxes, detections_boxes)

    # Solve global minimum cost assignment using Hungarian algorithm: O(N^3)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = set(range(len(tracks_boxes)))
    unmatched_detections = set(range(len(detections_boxes)))

    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost <= max_cost_threshold:
            matches.append((int(r), int(c)))
            unmatched_tracks.discard(r)
            unmatched_detections.discard(c)

    return matches, sorted(list(unmatched_tracks)), sorted(list(unmatched_detections))

def demonstrate_hungarian_association():
    print("=" * 70)
    print("STEP 1: BIPARTITE DATA ASSOCIATION (HUNGARIAN ALGORITHM)")
    print("=" * 70)

    # Scenario: 3 existing vehicle tracks from previous frame
    # Track 1: Moving East at (100, 100)
    # Track 2: Moving East at (300, 100)
    # Track 3: Exiting the frame at (1100, 100)
    tracks_boxes = np.array([
        [100.0, 100.0, 200.0, 160.0],  # Track 0 (ID #101)
        [300.0, 100.0, 400.0, 160.0],  # Track 1 (ID #102)
        [1100.0, 100.0, 1200.0, 160.0] # Track 2 (ID #103, leaves frame)
    ], dtype=np.float64)

    # New detections in current frame:
    # Det 0: Moved position of Track 0 (slightly shifted)
    # Det 1: Moved position of Track 1 (slightly shifted)
    # Det 2: Newly entering vehicle on the left border at (20, 100)
    detections_boxes = np.array([
        [112.0, 100.0, 212.0, 160.0],  # Det 0 (Matches Track 0)
        [315.0, 100.0, 415.0, 160.0],  # Det 1 (Matches Track 1)
        [20.0,  100.0, 120.0, 160.0]   # Det 2 (New vehicle entering)
    ], dtype=np.float64)

    cost_matrix = compute_iou_cost_matrix(tracks_boxes, detections_boxes)
    print("Cost Matrix (1 - IoU):\n")
    for i in range(len(tracks_boxes)):
        row_str = "  ".join([f"{cost_matrix[i, j]:.3f}" for j in range(len(detections_boxes))])
        print(f"  Track {i} Cost with [Det 0, Det 1, Det 2]: [{row_str}]")

    matches, unmatched_t, unmatched_d = associate_detections_to_tracks(
        tracks_boxes, detections_boxes, max_cost_threshold=0.70
    )

    print("\nHungarian Optimal Association Output:")
    print(f"  Matched Track-Detection Pairs : {matches}")
    print(f"  Unmatched Tracks (Lost/Exited): {unmatched_t}")
    print(f"  Unmatched Detections (Newborn): {unmatched_d}")

    # Assert correctness
    assert (0, 0) in matches, "Track 0 should match Det 0!"
    assert (1, 1) in matches, "Track 1 should match Det 1!"
    assert 2 in unmatched_t, "Track 2 should be unmatched (exited frame)!"
    assert 2 in unmatched_d, "Det 2 should be unmatched (new vehicle entering)!"

    print("\nAction Decision in MOT System:")
    print("  1. Matched Pairs    -> Update Kalman Filter with measurement (maintain ID).")
    print("  2. Unmatched Tracks -> Increment missed frames counter (mark 'Lost', then delete if stale).")
    print("  3. Unmatched Dets   -> Initialize new track instance with fresh unique ID.")
    print("\n[SUCCESS] Hungarian data association verified.")

def main():
    demonstrate_hungarian_association()
    print("=" * 70)
    print("PHASE 6 - MODULE 02: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
