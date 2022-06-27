# coding=utf-8
# author: wangzejun (wangzejunscut@126.com)

import argparse
import time

from lightltp.inference import LTP as LightLTP
from ltp import LTP

def speed_test(args):
    sent_list = []
    with open(args.sent_file, mode='r', encoding='utf-8') as sent_handle:
        for line in sent_handle:
            line = line.strip()
            if line:
                sent_list.append(line)
    
    lightltp = LightLTP(args.onnx_dir, device=args.device, num_threads=args.num_threads)
    ltp = LTP() if args.device == 'gpu' else LTP(device='cpu')   
    
    num_batches = int((len(sent_list) - 1) / args.batch_size) + 1
    print("****************************************************")
    print("processing {} sentences: seg-pos-ner".format(len(sent_list)))

    light_seg, light_pos, light_ner = [], [], []
    t_s = time.time()
    for i in range(num_batches):
        start = i * args.batch_size
        end = min((i + 1) * args.batch_size, len(sent_list))
        batch_sent_list = sent_list[start: end]
        
        seg, pos, hidden = lightltp.seg(batch_sent_list)
        ner = lightltp.ner(hidden)
        light_seg.extend(seg)
        light_pos.extend(pos)
        light_ner.extend(ner)
        
    t_e = time.time()
    print("lightltp time usage: {}s".format(t_e - t_s))
    
    ltp_seg, ltp_pos, ltp_ner = [], [], []
    t_s = time.time()
    for i in range(num_batches):
        start = i * args.batch_size
        end = min((i + 1) * args.batch_size, len(sent_list))
        batch_sent_list = sent_list[start: end]
        
        seg, hidden = ltp.seg(batch_sent_list)
        pos = ltp.pos(hidden)
        ner = ltp.ner(hidden)
        ltp_seg.extend(seg)
        ltp_pos.extend(pos)
        ltp_ner.extend(ner)
    
    t_e = time.time()
    print("ltp time usage: {}s".format(t_e - t_s))
    print("****************************************************")

    assert(light_seg == ltp_seg)
    assert(light_pos == ltp_pos)
    assert(light_ner == ltp_ner)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", type=str, required=True)
    parser.add_argument("--sent_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--device", type=str, choices=['cpu', 'gpu'], default='gpu')
    args = parser.parse_args()

    speed_test(args)
