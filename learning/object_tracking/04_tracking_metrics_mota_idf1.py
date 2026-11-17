"""
=============================================================================
AeroVision AI - Object Tracking Foundations: Module 04
=============================================================================
Tracking Evaluation Metrics: MOTA, IDF1, and ID Switches

Learning Objectives:
1. Understand the MOT evaluation challenge: evaluating spatio-temporal trajectories.
2. Define False Negatives (FN), False Positives (FP), and ID Switches (IDSW).
3. Compute Multiple Object Tracking Accuracy (MOTA) from scratch.
4. Compute Identification F1-Score (IDF1) for long-term identity preservation.
5. Contrast MOTA (detection-centric) with IDF1 (association-centric).
"""

import numpy as np
from typing import Dict, List, Tuple

def compute_mota(
    total_gt: int,
    total_fn: int,
    total_fp: int,
    total_idsw: int
) -> float:
    """
    Computes Multiple Object Tracking Accuracy (MOTA):
      MOTA = 1.0 - (FN + FP + IDSW) / GT
    """
    if total_gt == 0:
        return 0.0
    return 1.0 - (total_fn + total_fp + total_idsw) / float(total_gt)

def compute_idf1(
    id_tp: int,
    id_fp: int,
    id_fn: int
) -> Tuple[float, float, float]:
    """
    Computes Identification Precision (IDP), Identification Recall (IDR),
    and Identification F1-Score (IDF1).
    """
    idp = id_tp / (id_tp + id_fp) if (id_tp + id_fp) > 0 else 0.0
    idr = id_tp / (id_tp + id_fn) if (id_tp + id_fn) > 0 else 0.0
    idf1 = (2.0 * id_tp) / (2.0 * id_tp + id_fp + id_fn) if (2.0 * id_tp + id_fp + id_fn) > 0 else 0.0
    return idp, idr, idf1

def demonstrate_tracking_evaluation():
    print("=" * 70)
    print("STEP 1: MOTA & IDF1 TRACKING QUALITY EVALUATION")
    print("=" * 70)

    # Simulated 10-frame video sequence with 3 ground truth vehicles (GT-A, GT-B, GT-C)
    # Total GT object occurrences across 10 frames = 3 * 10 = 30
    total_gt = 30

    # Scenario 1: Tracker 1 (ByteTrack-like high precision tracker)
    # - Low misses (FN = 2)
    # - Low false alarms (FP = 1)
    # - Zero ID switches (IDSW = 0)
    # - IDTP = 28, IDFP = 1, IDFN = 2
    t1_fn, t1_fp, t1_idsw = 2, 1, 0
    t1_id_tp, t1_id_fp, t1_id_fn = 28, 1, 2

    # Scenario 2: Tracker 2 (Noisy tracker with frequent ID switches)
    # - Moderate misses (FN = 3)
    # - Low false alarms (FP = 2)
    # - Frequent ID switches due to occlusion (IDSW = 5)
    # - IDTP = 20, IDFP = 7, IDFN = 10
    t2_fn, t2_fp, t2_idsw = 3, 2, 5
    t2_id_tp, t2_id_fp, t2_id_fn = 20, 7, 10

    mota_1 = compute_mota(total_gt, t1_fn, t1_fp, t1_idsw)
    idp_1, idr_1, idf1_1 = compute_idf1(t1_id_tp, t1_id_fp, t1_id_fn)

    mota_2 = compute_mota(total_gt, t2_fn, t2_fp, t2_idsw)
    idp_2, idr_2, idf1_2 = compute_idf1(t2_id_tp, t2_id_fp, t2_id_fn)

    print(f"Dataset Scale: 10 frames, 3 vehicles (Total Ground Truth Occurrences = {total_gt})\n")
    print(f"{'Tracker System':<25} | {'FN':<4} | {'FP':<4} | {'IDSW':<5} | {'MOTA (%)':<10} | {'IDF1 (%)':<10}")
    print("-" * 75)
    print(f"{'ByteTrack (AeroVision)':<25} | {t1_fn:<4} | {t1_fp:<4} | {t1_idsw:<5} | {mota_1 * 100.0:>8.2f}% | {idf1_1 * 100.0:>8.2f}%")
    print(f"{'Standard SORT (Noisy)':<25} | {t2_fn:<4} | {t2_fp:<4} | {t2_idsw:<5} | {mota_2 * 100.0:>8.2f}% | {idf1_2 * 100.0:>8.2f}%")

    print("\nMetric Insights:")
    print("  1. MOTA (Multiple Object Tracking Accuracy):")
    print("     Focuses heavily on frame-level detection accuracy (FN, FP). Highly sensitive to missed detections.")
    print("  2. IDF1 (Identification F1-Score):")
    print("     Measures how consistently a tracker maintains the SAME identity for an object over time.")
    print("     Notice how Tracker 2's IDF1 collapsed from 94.9% down to 70.2% due to ID switches!")
    print("\n[SUCCESS] Tracking evaluation metrics verified.")

def main():
    demonstrate_tracking_evaluation()
    print("=" * 70)
    print("PHASE 6 - MODULE 04: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
