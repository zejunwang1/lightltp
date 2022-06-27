# coding=utf-8
# author: wangzejun (wangzejunscut@126.com)
# ref: https://github.com/HIT-SCIR/ltp

import numpy as np
import re
import warnings
from typing import List


def sentence_split(document: str, flag: str = "all", limit: int = 510, return_loc: bool = False) -> List[str]:
    """
    Args:
        document: str, 待分句文本
        flag: str, "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
        limit: int, 默认单句最大长度为 510 个字符
        return_loc: bool, 是否返回每个句子在原文本中的位置下标

    Returns: list

    """
    pos = 0
    sent_list = []
    try:
        if flag == "zh":
            document = re.sub('(?P<quotation_mark>([。？！…](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>([。？！]|…{1,2})[”’"\'])', r'\g<quotation_mark>\n', document)  # 特殊引号
        elif flag == "en":
            document = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 英文单字符断句符
            document = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', document)  # 特殊引号
        else:
            document = re.sub('(?P<quotation_mark>([。？！…?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>(([。？！!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n',
                              document)  # 特殊引号

        sent_list_ori = document.splitlines()
        for sent in sent_list_ori:
            length = len(sent)
            # strip
            for i in range(length):
                if sent[i] in [' ', '\t', '\r', '\n']:
                    i += 1
                else:
                    break
            start = i
            for i in range(length-1, -1, -1):
                if sent[i] in [' ', '\t', '\r', '\n']:
                    i -= 1
                else:
                    break
            end = i + 1
            sent = sent[start: end]
                 
            if not sent:
                pos += length
                continue
            if len(sent) <= limit:
                if return_loc:
                    sent_list.append((pos + start, sent))
                else:
                    sent_list.append(sent)
            else:
                i = 0
                n = 0
                cur = 0
                while i < len(sent):
                    i += 1
                    n += 1
                    if n == limit:
                        move = 1
                        while move < limit / 2:
                            c = sent[i - move]
                            if c in [',', '，', ';', '；', '.']:
                                i = i - move + 1
                                break
                            move += 1
                        if return_loc:
                            sent_list.append((pos + start + cur, sent[cur: i]))
                        else:
                            sent_list.append(sent[cur: i])
                        cur = i
                        n = 0
                if cur < len(sent):
                    if return_loc:
                        sent_list.append((pos + start + cur, sent[cur:]))
                    else:
                        sent_list.append(sent[cur:])
            pos += length

    except:
        sent_list.clear()
        if return_loc:
            sent_list.append((0, document.strip()))
        else:
            sent_list.append(document.strip())
    return sent_list        


def get_entities(seq, suffix=False):
    """
    Get entities from labeled sequence.
    Args:
        seq: list, sequence of labels.
    
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    
    Example:
        >>> from utils import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    
    def _validate_chunk(chunk, suffix):
        if chunk in ['O', 'B', 'I', 'E', 'S']:
            return

        if suffix:
            if not chunk.endswith(('-B', '-I', '-E', '-S')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))
        else:
            if not chunk.startswith(('B-', 'I-', 'E-', 'S-')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))
    
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """
    Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """
    Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def pad_sequence(seq, val: int = 0):
    """
    Pad sequence to same length.
    Args:
        seq: list of list, input sequence.
        val: int, pad value.
    
    Returns: list of list.
    
    Example:
        >>> from utils import pad_sequence
        >>> seq = [[1,2,3], [4,5,6,7,8], [5,7]]
        >>> pad_sequence(seq)
        [[1,2,3,0,0], 
         [4,5,6,7,8],
         [5,7,0,0,0]]
    """
    max_len = max(len(_) for _ in seq)
    for i in range(0, len(seq)):
        pad_len = max_len - len(seq[i])
        seq[i] += [val] * pad_len
    return seq


def length_to_mask(length, max_len: int = None, dtype=None):
    """
    将 Sequence length 转换成 Mask
    
    Example:
        >>> from utils import length_to_mask
        >>> length = [3,5,4]
        >>> length_to_mask(length)
        [[True, True, True, False, False],
         [True, True, True, True, True],
         [True, True, True, True, False]]
    """
    if max_len is None:
        max_len = max(length)
    mask = np.array([list(range(max_len))] * len(length)) < np.expand_dims(length, axis=-1)
    if dtype is not None:
        mask = mask.astype(dtype)
    return mask
