# coding=utf-8
# author: wangzejun (wangzejunscut@126.com)

from argparse import ArgumentParser

import json
import os
import torch
from transformers import AutoConfig, AutoTokenizer
from lightltp.model import SegPosModel, RelativeTransformerLinearClassifier

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--onnx_dir", type=str, required=True)
    parser.add_argument("--device", type=str, choices=['cpu', 'gpu'], default='gpu')
    args = parser.parse_args()
    os.makedirs(args.onnx_dir, exist_ok=True)        

    ckpt = torch.load(os.path.join(args.model_dir, "ltp.model"))

    if ckpt['seg'][0].startswith('B'):
        ckpt['seg'] = list(reversed(ckpt['seg']))
        seg_classifier_weight = ckpt['model']['seg_classifier.weight']
        ckpt['model']['seg_classifier.weight'] = seg_classifier_weight[[1, 0]]
    
    # save ltp config to config.json
    ltp_config = {}
    ltp_config['version'] = ckpt['version']
    ltp_config['seg'] = ckpt['seg']
    ltp_config['pos'] = ckpt['pos']
    ltp_config['ner'] = ckpt['ner']
    ltp_config['transformer_config'] = ckpt['transformer_config']
    ltp_config['seg_num_labels'] = ckpt['model_config'].seg_num_labels
    ltp_config['pos_num_labels'] = ckpt['model_config'].pos_num_labels
    ltp_config['ner_hidden_size'] = ckpt['model_config'].ner_hidden_size
    ltp_config['ner_num_heads'] = ckpt['model_config'].ner_num_heads
    ltp_config['ner_num_labels'] = ckpt['model_config'].ner_num_labels
    ltp_config['ner_num_layers'] = ckpt['model_config'].ner_num_layers
    
    config_file = os.path.join(args.onnx_dir, 'config.json')
    config_handle = open(config_file, mode='w')
    json.dump(ltp_config, config_handle, indent=2)
   
    # build seg_pos model and ner classifier
    transformer_config = ckpt['transformer_config']
    transformer_config['torchscript'] = True
    config = AutoConfig.for_model(**transformer_config)

    seg_pos_model = SegPosModel(config, seg_num_labels=len(ckpt['seg']), pos_num_labels=len(ckpt['pos']))
    ner_classifier = RelativeTransformerLinearClassifier(
        input_size=seg_pos_model.transformer.config.hidden_size,
        hidden_size=ckpt['model_config'].ner_hidden_size,
        num_layers=ckpt['model_config'].ner_num_layers,
        num_heads=ckpt['model_config'].ner_num_heads,
        num_labels=ckpt['model_config'].ner_num_labels,
        max_length=seg_pos_model.transformer.config.max_position_embeddings,
        dropout=0.1
    )
    
    ner_classifier_weight = {}
    for key, value in ckpt['model'].items():
        if key.startswith('ner_classifier'):
            ner_classifier_weight[key[15:]] = value

    seg_pos_model.load_state_dict(ckpt['model'], strict=False)
    ner_classifier.load_state_dict(ner_classifier_weight, strict=False)
    device = torch.device('cuda:0') if args.device=='gpu' and torch.cuda.is_available() else torch.device('cpu')
    seg_pos_model.to(device)
    ner_classifier.to(device)
    seg_pos_model.eval()
    ner_classifier.eval()
    
    # Get example data to run the model and export it to ONNX
    opset_version = 12
    text = ['中文自然语言处理技术平台']
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, config=seg_pos_model.transformer.config)
    inputs = tokenizer(
        text, padding=True, truncation=True, return_tensors='pt',
        max_length=seg_pos_model.transformer.config.max_position_embeddings
    ).to(device)
    
    # export seg and pos model
    model_file = "segpos_{}.onnx".format(args.device)
    export_model_path = os.path.join(args.onnx_dir, model_file)
    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(
            seg_pos_model,
            args=tuple(inputs.values()),
            f=export_model_path,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=[
                'input_ids',
                'attention_mask',
                'token_type_ids'
            ],
            output_names=['seg_output', 'pos_output', 'char_input'],
            dynamic_axes={
                'input_ids': symbolic_names,
                'attention_mask': symbolic_names,
                'token_type_ids': symbolic_names,
                'seg_output': symbolic_names,
                'pos_output': symbolic_names,
                'char_input': symbolic_names
            }
        )

    print("seg_pos model exported at ", export_model_path)    

    # export ner model
    batch_size = 20
    max_seq_len = 32
    hidden_size = seg_pos_model.transformer.config.hidden_size
    word_input = torch.ones(batch_size, max_seq_len, hidden_size, device=device)
    word_attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.int64, device=device)
    model_file = "ner_{}.onnx".format(args.device)
    export_model_path = os.path.join(args.onnx_dir, model_file)
    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(
            ner_classifier,
            args=(word_input, word_attention_mask),
            f=export_model_path,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=[
                'word_input',
                'word_attention_mask'
            ],
            output_names=['ner_output'],
            dynamic_axes={
                'word_input': symbolic_names,
                'word_attention_mask': symbolic_names,
                'ner_output': symbolic_names
            }
        )

if __name__ == "__main__":
    main()
