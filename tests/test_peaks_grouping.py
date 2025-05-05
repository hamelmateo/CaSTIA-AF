from calcium_activity_characterization.data.peaks import PeakDetector
import numpy as np


def test_peak_grouping_simple_overlap():
    """
    Test peak grouping on a synthetic signal with intentional overlaps.
    """
    # Construct synthetic signal
    trace = np.zeros(100)
    trace[10] = 10     # Peak A
    trace[12] = 6      # Peak B (overlaps with A)
    trace[40] = 9      # Peak C
    trace[60] = 5      # Peak D
    trace[62] = 4.5    # Peak E (overlaps with D)

    # Define detection parameters
    params = {
        "method": "skimage",
        "params": {
            "skimage": {
                "prominence": 1,
                "distance": 1,
                "height": None,
                "threshold": None,
                "width": None,
                "scale_class_quantiles": [0.33, 0.66]
            }
        },
        "peak_grouping": {
            "overlap_margin": 0,
            "verbose": True
        }
    }

    # Run detector
    detector = PeakDetector(params)
    peaks = detector.run(trace.tolist())

    # Print results
    print("\n--- Detected Peaks ---")
    for peak in peaks:
        print(f"{peak}")

    print("\n--- Summary ---")
    group_ids = set(p.group_id for p in peaks if p.group_id is not None)
    print(f"Number of groups: {len(group_ids)}")
    print(f"Parent peaks: {[p.id for p in peaks if p.role == 'parent']}")
    print(f"Member peaks: {[p.id for p in peaks if p.role == 'member']}")
    print(f"Individual peaks: {[p.id for p in peaks if p.role == 'individual']}")


if __name__ == "__main__":
    test_peak_grouping_simple_overlap()