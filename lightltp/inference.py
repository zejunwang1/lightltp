# coding=utf-8
# author: wangzejun (wangzejunscut@126.com)

import time
import json
import numpy as np
import os
import onnxruntime
import psutil

from typing import List
from transformers import AutoTokenizer, AutoConfig
from transformers import BatchEncoding

from .trie import Trie
from .utils import get_entities, length_to_mask, pad_sequence, sentence_split


WORD_START = 'B-W'
WORD_MIDDLE = 'I-W'


def convert_idx_to_name(y, array_len, id2label=None):
    if id2label:
        return [[id2label[idx] for idx in row[:row_len]] for row, row_len in zip(y, array_len)]
    else:
        return [[idx for idx in row[:row_len]] for row, row_len in zip(y, array_len)]


class LTP(object):
    seg_vocab: List[str]
    pos_vocab: List[str]
    ner_vocab: List[str]
    
    def __init__(self, path: str, device: str = 'gpu', num_threads: int = None):
        assert device in ['cpu', 'gpu']
        if device == 'gpu':
            assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
            providers = ['CUDAExecutionProvider']
            segpos_opt_onnxfile = os.path.join(path, 'segpos_gpu_opt.onnx')
            ner_onnxfile = os.path.join(path, 'ner_gpu.onnx')
        else:
            assert 'CPUExecutionProvider' in onnxruntime.get_available_providers()
            providers = ['CPUExecutionProvider']
            segpos_opt_onnxfile = os.path.join(path, 'segpos_cpu_opt.onnx')
            ner_onnxfile = os.path.join(path, 'ner_cpu.onnx')
        
        assert os.path.exists(segpos_opt_onnxfile)
        assert os.path.exists(ner_onnxfile)
        
        config_file = os.path.join(path, 'config.json')        
        config_handle = open(config_file, mode='r', encoding='utf-8')
        config = json.load(config_handle)
        self.seg_vocab = config.get('seg')
        self.seg_vocab_dict = {tag: idx for idx, tag in enumerate(self.seg_vocab)}
        self.pos_vocab = config.get('pos')
        self.ner_vocab = config.get('ner')
        self._model_version = config.get('version', None)

        transformer_config = config['transformer_config']
        transformer_config['torchscript'] = True
        self.pretrained_config = AutoConfig.for_model(**transformer_config)
        self.tokenizer = AutoTokenizer.from_pretrained(path, config=self.pretrained_config, use_fast=True)
        self.trie = Trie()
        
        if num_threads is None:
            num_threads = psutil.cpu_count(logical=False)            
        elif not isinstance(num_threads, int) or num_threads < 0:
            raise ValueError("Invalid num_threads!")
        
        # seg and pos session
        segpos_sess_options = onnxruntime.SessionOptions()
        segpos_sess_options.intra_op_num_threads = num_threads
        self.segpos_session = onnxruntime.InferenceSession(
            segpos_opt_onnxfile, segpos_sess_options, providers=providers
        )
        
        # ner session
        ner_sess_options = onnxruntime.SessionOptions()
        ner_sess_options.intra_op_num_threads = num_threads
        self.ner_session = onnxruntime.InferenceSession(
            ner_onnxfile, ner_sess_options, providers=providers
        )

    @property
    def model_version(self):
        return self._model_version or 'unknown'
    
    @property
    def max_length(self):
        return self.pretrained_config.max_position_embeddings
    
    def init_dict(self, path, max_window=None):
        self.trie.init(path, max_window)
    
    def add_words(self, words, max_window=None):
        self.trie.add_words(words)
        self.trie.max_window = max_window
    
    def seg_with_dict(self, inputs: List[str], tokenized: BatchEncoding, batch_prefix):
        matching = []
        for source_text, encoding, preffix in zip(inputs, tokenized.encodings, batch_prefix):
            text = [source_text[start: end] for start, end in encoding.offsets[1:-1] if end != 0]
            matching_pos = self.trie.maximum_forward_matching(text, preffix)
            matching.append(matching_pos)
        return matching
    
    @staticmethod
    def sentence_split(text: str, flag: str = "all", max_len: int = 510, return_loc: bool = False):
        return sentence_split(document=text, flag=flag, limit=max_len, return_loc=return_loc)
    
    def seg(self, inputs: List[str]):
        tokenized = self.tokenizer.batch_encode_plus(
            inputs, padding=True, truncation=True,
            return_tensors='np', max_length=self.max_length
        )
        lengths = np.sum(tokenized['attention_mask'], axis=-1) - 2

        ort_inputs = {
            'input_ids': tokenized['input_ids'], 
            'token_type_ids': tokenized['token_type_ids'],
            'attention_mask': tokenized['attention_mask']
        }
        ort_outputs = self.segpos_session.run(None, ort_inputs)
        
        seg = np.argmax(ort_outputs[0], axis=-1)
        pos = np.argmax(ort_outputs[1], axis=-1)
        char_input = ort_outputs[2]
        
        batch_prefix = [[
            word_idx != encoding.words[idx - 1]
            for idx, word_idx in enumerate(encoding.words) if word_idx is not None
        ] for encoding in tokenized.encodings]

        # merge segments with maximum forward matching
        if self.trie.is_init:
            matches = self.seg_with_dict(inputs, tokenized, batch_prefix)
            for sent_match, sent_seg in zip(matches, seg):
                for start, end in sent_match:
                    sent_seg[start] = self.seg_vocab_dict[WORD_START]
                    sent_seg[start+1: end] = self.seg_vocab_dict[WORD_MIDDLE]
                    if end < len(sent_seg):
                        sent_seg[end] = self.seg_vocab_dict[WORD_START]
        
        segment_output = convert_idx_to_name(seg, lengths, self.seg_vocab)
        sentences = []
        word_idx = []
        word_length = []
        
        for source_text, length, encoding, seg_tag, preffix in \
                zip(inputs, lengths, tokenized.encodings, segment_output, batch_prefix):
            offsets = encoding.offsets[1: length + 1]
            text = []
            last_offset = None
            for start, end in offsets:
                text.append('' if last_offset == (start, end) else source_text[start: end])
                last_offset = (start, end)
            
            for idx in range(1, length):
                current_beg = offsets[idx][0]
                forward_end = offsets[idx - 1][-1]
                if forward_end < current_beg:
                    text[idx] = source_text[forward_end: current_beg] + text[idx]
                if not preffix[idx]:
                    seg_tag[idx] = WORD_MIDDLE
            
            entities = get_entities(seg_tag)
            word_length.append(len(entities))
            sentences.append([''.join(text[entity[1]: entity[2] + 1]).strip() for entity in entities])
            word_idx.append([entity[1] for entity in entities])
        
        first_idx = np.arange(len(word_idx)).reshape(len(word_idx), -1)
        word_idx = pad_sequence(word_idx)
        word_input = char_input[first_idx, word_idx]
        word_mask = length_to_mask(word_length, dtype=np.int64)

        postagger_output = convert_idx_to_name(pos[first_idx, word_idx], word_length, self.pos_vocab)
        return sentences, postagger_output, {
            'word_input': word_input,
            'word_mask': word_mask,
            'word_length': word_length
        }        
    
    def ner(self, hidden: dict, as_entities: bool = True):
        """
        Named entity recognition.
        Args:
            hidden: ner input obtained from seg.
            as_entities: whether to return list of (Type, Start, End).
        
        Returns: list
        """
        ort_inputs = {
            'word_input': hidden['word_input'],
            'word_attention_mask': hidden['word_mask']
        }
        ort_outputs = self.ner_session.run(None, ort_inputs)
        ner_output = np.argmax(ort_outputs[0], axis=-1)
        ner_output = convert_idx_to_name(ner_output, hidden['word_length'], self.ner_vocab)
        return [get_entities(ner) for ner in ner_output] if as_entities else ner_output
