# lightltp: 基于 onnxruntime 推理引擎的中文 ltp 词法分析

ltp (https://github.com/HIT-SCIR/ltp) 是哈工大社会计算和信息检索研究中心（HIT-SCIR）开源的中文自然语言处理工具集，用户可以使用 ltp 对中文文本进行分词、词性标注、命名实体识别、语义角色标注、依存句法分析、语义依存分析等等工作。

lightltp 是一个基于 onnxruntime-gpu 推理引擎的中文 ltp 词法分析工具，支持对中文文本进行分词、词性标注和命名实体识别三项任务。lightltp 将 ltp 中默认的 small 模型的分词、词性标注和命名实体识别部分导出成 ONNX 格式，然后基于 onnxruntime-gpu 进行快速推理。

## 依赖环境

- onnxruntime-gpu >= 1.6.0
  
- onnx >= 1.5.0
  
- torch >= 1.2.0
  
- transformers >= 4.0.0, < 5.0
  
- pygtrie >= 2.3.0, < 2.5
  

## 快速使用

将 lightltp 克隆到本地：

```shell
git clone https://github.com/zejunwang1/lightltp
cd lightltp
```

运行 demo.py：

```python
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
```

seg:

```
[['颐和园', '是', '北京', '的', '著名', '景点', '。']]
```

pos:

```
[['ns', 'v', 'ns', 'u', 'a', 'n', 'wp']]
```

ner:

```
[[('Ns', 0, 0), ('Ns', 2, 2)]]
```

## 模型转换

可以使用 convert_ltp_to_onnx.py 将哈工大训练的 ltp pytorch 模型转化为 ONNX 格式：

```
usage: convert_ltp_to_onnx.py [-h] --model_dir MODEL_DIR --onnx_dir ONNX_DIR
                              [--device {cpu,gpu}]
```

其中 --model_dir 表示原始的 ltp pytorch 模型，--onnx_dir 表示导出的 ONNX 格式存储路径。进一步可使用 transformer_optimizer 文件夹中的 optimizer.py 对 transformer 结构进行优化：

```shell
python optimizer.py --input INPUT --output OUTPUT --model_type bert --num_heads NUM_HEADS --hidden_size HIDDEN_SIZE
```

其中 --input 为原始的 ONNX 模型文件，--output 为经过优化后的 ONNX 模型文件。

## 速度测试

lightltp 支持在 cpu 或 gpu 上对中文文本进行处理。sents.txt 为从中文维基百科中抽取的 1000 条句子（平均长度在 128 个字符以上），运行 test_speed.py 进行速度测试：

```
usage: test_speed.py [-h] --onnx_dir ONNX_DIR --sent_file SENT_FILE
                     [--batch_size BATCH_SIZE] [--num_threads NUM_THREADS]
                     [--device {cpu,gpu}]
```

```shell
python test_speed.py --onnx_dir onnx/ --sent_file sents.txt --batch_size 1 --device gpu
```

在 cpu 和 gpu 上分别实验了 batch_size=1, 2, 4, 8, 32, 64，lightltp 和 ltp 的处理速度比较如下表所示：

| batch_size | 1   | 2   | 4   | 8   | 32  | 64  |
| --- | --- | --- | --- | --- | --- | --- |
| ltp-cpu 耗时 (s) | 34.377 | 23.577 | 16.247 | 12.022 | 12.063 | 15.255 |
| lightltp-cpu 耗时 (s) | 15.905 | 13.834 | 12.455 | 11.327 | 12.918 | 12.736 |
| ltp-gpu 耗时 (s) | 18.545 | 12.318 | 9.171 | 5.686 | 3.190 | 2.871 |
| lightltp-gpu 耗时 (s) | 4.913 | 4.136 | 3.915 | 3.609 | 2.774 | 2.593 |

可以看出，当 batch_size=1 时，gpu 模式下的 lightltp 相比于 ltp 约有 3~4 倍的推理加速；cpu 模式下的 lightltp 相比于 ltp 约有 2 倍的推理加速。随着 batch_size 的增加，由于 onnxruntime 不适合处理大批量输入数据，lightltp 和 ltp 的处理速度逐渐持平。当 batch_size=64 时，lightltp 仅略快于 ltp。

## Contact

邮箱： [wangzejunscut@126.com](mailto:wangzejunscut@126.com)

微信：autonlp
