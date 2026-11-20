#!/usr/bin/env python3
"""Publish a manual annotation only; never changes sensor or vehicle state."""
import argparse
import json
import time

import rospy
from std_msgs.msg import String


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["vision_mask_start", "vision_mask_end",
                        "maneuver_start", "maneuver_end", "note"])
    parser.add_argument("label", nargs="?", default="")
    args = parser.parse_args()
    rospy.init_node("quaddock_manual_event", anonymous=True)
    pub = rospy.Publisher("/quaddock/experiment/events", String, queue_size=10)
    deadline = time.monotonic() + 5.0
    while not pub.get_num_connections() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not pub.get_num_connections():
        parser.exit(1, "No event subscriber: start the recorder first.\n")
    payload = dict(kind=args.kind, label=args.label, source="manual_annotation",
                   ros_time=rospy.Time.now().to_sec(), wall_time=time.time())
    pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
    time.sleep(0.3)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
