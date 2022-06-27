# coding=utf-8
# author: wangzejun (wangzejunscut@126.com)

import argparse
import numpy as np
import os
import onnxruntime
import psutil

from transformers import BertTokenizer

def seg_pos_test(args):
    model_file = "segpos_{}_opt.onnx".format(args.device)
    model_filepath = os.path.join(args.onnx_dir, model_file)
   
    providers = ['CPUExecutionProvider'] if args.device == 'cpu' else ['CUDAExecutionProvider']
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)
    session = onnxruntime.InferenceSession(model_filepath, sess_options, providers=providers)
    
    vocab_path = os.path.join(args.onnx_dir, "vocab.txt")
    tokenizer = BertTokenizer(vocab_path)
    text = ["中文自然语言处理技术平台", "颐和园是北京的著名景点"]
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="np")
    ort_inputs = {k : v for k, v in inputs.items()}
    
    ort_outputs = session.run(None, ort_inputs)
    
    seg_output = np.argmax(ort_outputs[0], axis=-1)
    pos_output = np.argmax(ort_outputs[1], axis=-1)
    assert(seg_output.tolist()[0] == [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0])
    assert(pos_output[1][0] == 9)
    assert(pos_output[1][4] == 9)
    assert(pos_output[1][7] == 5)

def ner_test(args):
    model_file = "ner_{}.onnx".format(args.device)
    model_filepath = os.path.join(args.onnx_dir, model_file)
    
    providers = ['CPUExecutionProvider'] if args.device == 'cpu' else ['CUDAExecutionProvider']
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)
    session = onnxruntime.InferenceSession(model_filepath, sess_options, providers=providers)
    
    try:
        from ltp import LTP
    except ImportError:
        print("Fail to import ltp, install it through: pip install ltp")
        raise
    
    text = ["中文自然语言处理技术平台", "颐和园是北京的著名景点"]
    model = LTP()
    seg, hidden = model.seg(text)
    ner = model.ner(hidden)
    word_input = hidden['word_input'].cpu().numpy()
    word_attention_mask = hidden['word_cls_mask'][:, 1:].cpu().numpy().astype(np.int64)
    ort_inputs = {"word_input": word_input, "word_attention_mask": word_attention_mask}
    
    ort_outputs = session.run(None, ort_inputs)
    ner_output = np.argmax(ort_outputs[0], axis=-1)
    assert(ner_output.tolist()[0] == [0, 0, 0, 0, 0, 0])
    assert(ner_output.tolist()[1] == [1, 0, 1, 0, 0, 0])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", type=str, required=True)
    parser.add_argument("--device", type=str, choices=['cpu', 'gpu'], default='gpu')
    args = parser.parse_args()
    
    if args.device == 'cpu':
        assert('CPUExecutionProvider' in onnxruntime.get_available_providers())
    else:
        assert('CUDAExecutionProvider' in onnxruntime.get_available_providers())

    seg_pos_test(args)
    ner_test(args)
