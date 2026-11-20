"""Regression tests for stream fragmentation, frame validation and firmware upgrade."""
import importlib.util
import math
from pathlib import Path
import struct
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('linktarck',ROOT/'linktarck.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def packet(mark=0x0b, local_id=0xF3F0059B, remote_id=0x671256F8,
           distance_mm=1704, angle=-5032, angle1=250):
    if mark==7:
        body=struct.pack('<B B I I 4s H B',1,7,123,456,b'\0'*4,5000,1)
        node=struct.pack('<BB',2,3)+distance_mm.to_bytes(3,'little',signed=True)
        node+=struct.pack('<hBB2s',angle,166,161,b'\0'*2)
    else:
        body=struct.pack('<B I I I 4s H B',0,local_id,123,456,b'\0'*4,5000,1)
        node=struct.pack('<BI',0,remote_id)+distance_mm.to_bytes(3,'little',signed=True)
        if mark==0x0b:node+=struct.pack('<hh2sBB',angle,angle1,b'\0'*2,166,161)
        else:node+=bytes([166,161])
    value=bytes([0x55,mark])+struct.pack('<H',len(body)+len(node)+5)+body+node
    return value+bytes([sum(value)&255])


class ParserTest(unittest.TestCase):
    def setUp(self):
        m.kalman_state.clear();m.kalman_covariance.clear()

    def test_vendor_layout_and_signed_angles(self):
        data=m.parse_nlink_aoa_nodeframe0(packet())
        self.assertEqual(data['tag_id'],0xF3F0059B)
        self.assertEqual(data['anchors'][0]['anchor_id'],0x671256F8)
        self.assertAlmostEqual(data['anchors'][0]['distance_m'],1.704)
        self.assertAlmostEqual(math.degrees(data['anchors'][0]['angle_rad']),-50.32)
        self.assertAlmostEqual(math.degrees(data['anchors'][0]['angle1_rad']),2.5)

    def test_legacy_protocol(self):
        data=m.parse_nlink_aoa_nodeframe0(packet(7))
        self.assertEqual(data['protocol'],'aoa_nodeframe0')
        self.assertAlmostEqual(math.degrees(data['anchors'][0]['angle_rad']),-50.32)

    def test_tag_range_only_has_no_fabricated_bearing(self):
        frame=packet(8)
        self.assertEqual(m.NLinkStreamParser().feed(frame),[frame])
        data=m.parse_nlink_aoa_nodeframe0(frame)
        self.assertFalse(data['bearing_available'])
        self.assertIsNone(data['anchors'][0]['angle_rad'])
        self.assertFalse(data['anchors'][0]['sending_flag'])

    def test_split_headers_noise_and_multiple_frames(self):
        expected=[packet(),packet(7),packet(8)]
        raw=b'noise\x55\xff'+b''.join(expected)
        for size in [1,2,3,5,17,64,4096]:
            parser=m.NLinkStreamParser();frames=[]
            for i in range(0,len(raw),size):frames+=parser.feed(raw[i:i+size])
            self.assertEqual(frames,expected)

    def test_bad_checksum_length_and_truncation_do_not_update_filters(self):
        good=packet();bad=bytearray(good);bad[-1]^=1
        self.assertIsNone(m.parse_nlink_aoa_nodeframe0(bad))
        self.assertIsNone(m.parse_nlink_aoa_nodeframe0(good[:-1]))
        bad_length=bytearray(good);bad_length[2]=1
        self.assertIsNone(m.parse_nlink_aoa_nodeframe0(bad_length))
        self.assertEqual(m.kalman_state,{})
        parser=m.NLinkStreamParser()
        self.assertEqual(parser.feed(bytes(bad)+bytes(bad_length)+good),[good])

    def test_zero_node_frame(self):
        frame=bytearray(packet()[:24]);frame[2:4]=(25).to_bytes(2,'little');frame[23]=0
        frame.append(sum(frame)&255)
        data=m.parse_nlink_aoa_nodeframe0(bytes(frame))
        self.assertEqual(data['anchors'],[])

    def test_real_serial_capture_survives_every_chunk_size(self):
        raw=(Path(__file__).parent/'20261117_171449.bin').read_bytes()
        frames=m.NLinkStreamParser().feed(raw)
        self.assertEqual(len(frames),154)
        self.assertEqual(sum(len(x)==41 for x in frames),41)
        for size in [1,2,5,17,64,1024]:
            parser=m.NLinkStreamParser();actual=[]
            for i in range(0,len(raw),size):actual+=parser.feed(raw[i:i+size])
            self.assertEqual(actual,frames)
        for frame in frames:
            self.assertIsNotNone(m.parse_nlink_aoa_nodeframe0(frame))


if __name__=='__main__':unittest.main(verbosity=2)
