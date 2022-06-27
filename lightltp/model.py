# coding=utf-8
# author: wangzejun (wangzejunscut@126.com)

from torch import nn
from transformers import AutoModel
from .relative_transformer import RelativeTransformer

class SegPosModel(nn.Module):
    def __init__(self, config, seg_num_labels: int = 2, pos_num_labels: int = 27):
        super().__init__()
        # base electra model
        self.transformer = AutoModel.from_config(config)
        transformer_hidden_size = self.transformer.config.hidden_size
        
        # linear classifier of segmentor
        self.seg_classifier = nn.Linear(transformer_hidden_size, seg_num_labels)
        
        # linear classifier of postagger
        self.pos_classifier = nn.Linear(transformer_hidden_size, pos_num_labels)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None
    ):
        sequence_output, *_ = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False
        )
        
        # remove [CLS] [SEP]
        char_input = sequence_output[:, 1:-1]
        
        seg_logits = self.seg_classifier(char_input)
        pos_logits = self.pos_classifier(char_input)
        return seg_logits, pos_logits, char_input

class RelativeTransformerLinearClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, num_labels, max_length, dropout):
        super().__init__()
        self.relative_transformer = RelativeTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_length=max_length * 2
        )
        self.classifier = nn.Linear(input_size, num_labels)
        
    def forward(self, input, word_attention_mask=None):
        sequence_output = self.relative_transformer(input, word_attention_mask)
        logits = self.classifier(sequence_output)
        return logits
