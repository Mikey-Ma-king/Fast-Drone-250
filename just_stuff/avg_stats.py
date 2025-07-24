import re
import yaml
import numpy as np
import time

def print_mat(m):
    fmt = "   data: [" + "{: 2.5f},"*4
    print(fmt.format(*m[:4]))
    fmt = "          " + "{: 2.5f},"*4
    print(fmt.format(*m[4:8]))
    print(fmt.format(*m[8:12]))
    fmt = "          {: 2.5f},{: 2.5f},{: 2.5f},{: 2.5f}]"
    print(fmt.format(*m[12:]))

data_total = []
num_data = 0
data_last = None
while True:
    with open("/home/ros/vins_output/extrinsic_parameter.txt") as f:
        lists = re.findall(r"\[.+?\]", f.read().replace('\n', ''))
    if data_last == lists:
        continue
    if data_last is None:
        data_total = [np.array(eval(l)) for l in lists]
    else:
        for i, l in enumerate(lists):
            data_total[i] += np.array(eval(l))
    data_last = lists
    num_data += 1
    for i, l in enumerate(data_total):
        print(f"""body_T_cam{i}: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d""")
        print_mat(l / num_data)
    print(num_data)
    time.sleep(1)
