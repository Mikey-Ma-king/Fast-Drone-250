import math
import sys
import unittest
from pathlib import Path
import re

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from quaddock_ours.estimator import make_estimator
from quaddock_ours.models import sensor_prediction


def _odom(sensor, stamp, position, yaw=0.0, measurement_id=None):
    return {
        "measurement_id": measurement_id or "%s-%s" % (sensor, stamp),
        "sensor": sensor,
        "sample_time": stamp,
        "arrival_time": stamp,
        "value": {"position": list(position), "velocity": [0.0, 0.0, 0.0], "yaw": yaw},
    }


class CoreTest(unittest.TestCase):
    def test_real_config_matches_frozen_sw3_joint12(self):
        text = (Path(__file__).parents[1] / "config" / "ours_real.yaml").read_text()
        for key, expected in (("window_s", "3.0"), ("process_scale", "4.0"),
                              ("yaw_process_ratio", "0.5"),
                              ("translation_curvature_scale", "0.5")):
            match = re.search(r"^  %s: (.+)$" % re.escape(key), text, re.MULTILINE)
            self.assertIsNotNone(match)
            self.assertEqual(match.group(1).strip(), expected)

    def test_split_uwb_jacobians_are_one_by_four(self):
        uav = {"position": [0.0, 0.0, 0.0], "yaw": 0.0,
               "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
        config = {"uwb_dog_lever_arm_m": [0.1, -0.02, 0.0],
                  "uwb_uav_lever_arm_m": [0.0, 0.0, 0.0]}
        for sensor in ("uwb_range", "uwb_bearing"):
            _, jac, _ = sensor_prediction(sensor, np.array([1.0, 2.0, 0.3]),
                                           0.2, uav, config)
            self.assertEqual(jac.shape, (1, 4))

    def test_ours_initializes_and_publishes_finite_snapshot(self):
        init = {"reference_time": 0.0, "psi": 0.0, "b": [1.0, 0.0, 0.0],
                "rate": [0.0, 0.0, 0.0, 0.0], "q_reference": [0.0, 0.0, 0.0],
                "u_reference": [0.0, 0.0, 0.0]}
        est = make_estimator({"window_s": 2.0}, init)
        est.ingest(_odom("uav_odom", 0.0, [0.0, 0.0, 0.0], measurement_id="u0"))
        est.ingest(_odom("dog_odom", 0.0, [0.0, 0.0, 0.0], measurement_id="d0"))
        vision = _odom("vision", 0.0, [1.0, 0.0, 0.0], measurement_id="v0")
        vision["value"] = {"position": [1.0, 0.0, 0.0], "yaw": 0.0}
        vision["odom"] = {"uav": est.odometry["uav_odom"][1], "dog": est.odometry["dog_odom"][1]}
        self.assertTrue(est.ingest(vision)["accepted"])
        snapshot = est.predict(0.1)
        self.assertIsNotNone(snapshot["position"])
        self.assertTrue(np.isfinite(snapshot["position"]).all())
        self.assertTrue(math.isfinite(snapshot["yaw"]))

    def test_range_and_bearing_are_independent_events(self):
        init = {"reference_time": 0.0, "psi": 0.0, "b": [1.0, 0.0, 0.0],
                "rate": [0.0, 0.0, 0.0, 0.0], "q_reference": [0.0, 0.0, 0.0],
                "u_reference": [0.0, 0.0, 0.0]}
        est = make_estimator({"window_s": 2.0}, init)
        uav = _odom("uav_odom", 0.0, [0.0, 0.0, 0.0], measurement_id="u0")
        dog = _odom("dog_odom", 0.0, [0.0, 0.0, 0.0], measurement_id="d0")
        est.ingest(uav)
        est.ingest(dog)
        matched = {"uav": uav["value"], "dog": dog["value"]}
        for sensor, value, mid in (("uwb_range", {"range": 1.0}, "r0"),
                                   ("uwb_bearing", {"bearing": 0.0}, "b0")):
            event = {"measurement_id": mid, "sensor": sensor, "sample_time": 0.0,
                     "arrival_time": 0.0, "value": value, "odom": matched}
            self.assertTrue(est.ingest(event)["accepted"])
        self.assertEqual(est.diagnostics["updates"], 2)


if __name__ == "__main__":
    unittest.main()
