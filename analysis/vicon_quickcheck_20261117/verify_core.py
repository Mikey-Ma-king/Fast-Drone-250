#!/usr/bin/env python3
"""Verify that the derivative preserves every selected raw message and timestamp."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import rosbag

out=Path(__file__).resolve().parent
manifest=json.loads((out/'core_filter_manifest.json').read_text())

def digest(path,topics):
    h=hashlib.sha256()
    counts=Counter()
    with rosbag.Bag(path) as bag:
        for topic,raw,t in bag.read_messages(topics=topics,raw=True):
            h.update(topic.encode()+b'\x00')
            h.update(struct.pack('<II',t.secs,t.nsecs))
            h.update(raw[0].encode()+b'\x00')
            h.update(raw[1])
            counts[topic]+=1
    return dict(sha256=h.hexdigest(),counts=dict(counts))

source=digest(manifest['source'],manifest['topics'])
derived=digest(manifest['derivative'],None)
assert source==derived,(source,derived)
result=dict(result='PASS',source=manifest['source'],derivative=manifest['derivative'],
            verification='Ordered topic, datatype, timestamp and serialized payload SHA256 match',**source)
(out/'core_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(result='PASS',messages=sum(source['counts'].values()),sha256=source['sha256'])))
