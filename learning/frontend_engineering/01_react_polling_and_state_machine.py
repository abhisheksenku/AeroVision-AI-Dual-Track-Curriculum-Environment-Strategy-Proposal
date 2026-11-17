"""
=============================================================================
AeroVision AI - Frontend Engineering Foundations: Module 01
=============================================================================
React State Machine, Asynchronous Polling, and Exponential Backoff

Learning Objectives:
1. Model the asynchronous client-side job lifecycle as a finite state machine.
2. Formulate polling intervals with exponential backoff: dt = min(dt_max, dt_0 * gamma^k).
3. Handle network jitter and avoid DDoS-ing the API during heavy background video compute.
"""

import time
import math
from typing import Dict, Any, List

class PollingStateMachine:
    """
    Simulates the React client-side polling hook (useJobPolling)
    with adaptive exponential backoff.
    """
    def __init__(self, dt_0: float = 1.0, dt_max: float = 5.0, gamma: float = 1.25):
        self.dt_0 = dt_0       # Initial interval (1 second)
        self.dt_max = dt_max   # Cap interval at 5 seconds
        self.gamma = gamma     # Multiplicative growth factor
        self.poll_count = 0
        self.state = "IDLE"    # 'IDLE', 'UPLOADING', 'PROCESSING', 'COMPLETED', 'FAILED'
        self.poll_history = []

    def get_next_interval(self) -> float:
        """Computes the next sleep interval using exponential backoff."""
        interval = min(self.dt_max, self.dt_0 * (self.gamma ** self.poll_count))
        return round(interval, 2)

    def simulate_poll_loop(self, job_progress_sequence: List[int]):
        """
        Simulates polling a server endpoint until completion.
        """
        self.state = "PROCESSING"
        print(f"Starting Polling Loop from state: {self.state}")
        
        for progress in job_progress_sequence:
            self.poll_count += 1
            interval = self.get_next_interval()
            self.poll_history.append((self.poll_count, progress, interval))
            
            print(f"  Poll #{self.poll_count:02d} | Server Progress: {progress:>3}% | Next Backoff Delay: {interval:.2f}s")
            
            if progress >= 100:
                self.state = "COMPLETED"
                break

        print(f"Final State: {self.state} after {self.poll_count} poll cycles.")

def demonstrate_polling_math():
    print("=" * 70)
    print("STEP 1: EXPONENTIAL BACKOFF POLLING FORMULATION")
    print("=" * 70)

    # Simulated server job taking ~15 seconds to complete
    server_progress = [10, 25, 40, 55, 70, 85, 95, 100]

    sm = PollingStateMachine(dt_0=1.0, dt_max=5.0, gamma=1.2)
    sm.simulate_poll_loop(server_progress)

    assert sm.state == "COMPLETED"
    assert sm.poll_count == len(server_progress)
    assert sm.poll_history[-1][2] <= 5.0 # Must respect dt_max

    print("\nBenefits for Drone Surveillance Web Dashboards:")
    print("  1. Responsive Feedback: Polls frequently (every 1.0s) initially to capture quick starts.")
    print("  2. Server Protection: Backs off smoothly during multi-minute 4K video analysis.")
    print("  3. Predictable Network Traffic: Eliminates socket starvation on edge drone telemetry links.")
    print("\n[SUCCESS] React polling state machine and backoff math verified.")
    print()

def main():
    demonstrate_polling_math()
    print("=" * 70)
    print("PHASE 9 - MODULE 01: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
