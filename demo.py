# coding=utf-8

from lightltp.inference import LTP

# Initialization parameters
# path: str, the path of onnx model.
# device: str, cpu/gpu, default is gpu.
# num_threads: int, intra_op_num_threads of inference session, default is None.
# ltp = LTP(path='onnx/', device='gpu', num_threads=4)
ltp = LTP(path='onnx/')

seg, pos, hidden = ltp.seg(['颐和园是北京的著名景点。'])
ner = ltp.ner(hidden)
