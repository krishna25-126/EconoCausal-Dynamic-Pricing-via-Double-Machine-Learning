from econocausal import data
from econocausal.drift import DriftDetector


def test_stable_and_drift():
    ref = data.generate_observational(3000)
    det = DriftDetector().fit(ref)
    assert det.check(data.campaign_audience(1500, shift=0.0))["status"] == "stable"
    shifted = det.check(data.campaign_audience(1500, shift=1.0))
    assert shifted["status"] == "drift_detected"
    assert "engagement" in shifted["drifted_features"]