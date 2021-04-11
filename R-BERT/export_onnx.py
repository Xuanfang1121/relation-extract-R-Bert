# -*- coding: utf-8 -*-
# @Time    : 2021/3/29 23:08
# @Author  : zxf
import os
import torch
import numpy as np
from torch.autograd import Variable

from model import RBERT
from utils import load_tokenizer


def get_args():
    return torch.load(os.path.join("./model/", "training_args.bin"))


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_label(label_file):
    label = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            label.append(line.strip())
    return label


def convert_input_file_to_tensor_dataset(
    input_file,
    args,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    tokenizer = load_tokenizer(args)

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_e1_mask = []
    all_e2_mask = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            tokens = tokenizer.tokenize(line)

            e11_p = tokens.index("<e1>")  # the start position of entity1
            e12_p = tokens.index("</e1>")  # the end position of entity1
            e21_p = tokens.index("<e2>")  # the start position of entity2
            e22_p = tokens.index("</e2>")  # the end position of entity2

            # Replace the token
            tokens[e11_p] = "$"
            tokens[e12_p] = "$"
            tokens[e21_p] = "#"
            tokens[e22_p] = "#"

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            if args.add_sep_token:
                special_tokens_count = 2
            else:
                special_tokens_count = 1
            if len(tokens) > args.max_seq_len - special_tokens_count:
                tokens = tokens[: (args.max_seq_len - special_tokens_count)]

            # Add [SEP] token
            if args.add_sep_token:
                tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            # Add [CLS] token
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = args.max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            # e1 mask, e2 mask
            e1_mask = [0] * len(attention_mask)
            e2_mask = [0] * len(attention_mask)

            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)
            all_e1_mask.append(e1_mask)
            all_e2_mask.append(e2_mask)
    return all_input_ids, all_attention_mask, all_token_type_ids, all_e1_mask, all_e2_mask


args = get_args()
model = RBERT.from_pretrained("./model/", args=args)
device = get_device(args)
model.eval()
all_input_ids, all_attention_mask, all_token_type_ids, \
all_e1_mask, all_e2_mask = convert_input_file_to_tensor_dataset("./sample_pred_in_ch_onnx.txt", args)
print(np.array(all_input_ids).shape)

input_ids = Variable(torch.from_numpy(np.array(all_input_ids))).type(torch.LongTensor)
attention_mask = Variable(torch.from_numpy(np.array(all_attention_mask))).type(torch.LongTensor)
token_type_ids = Variable(torch.from_numpy(np.array(all_token_type_ids))).type(torch.LongTensor)
labels = Variable(torch.ones(1)).type(torch.LongTensor).flatten()
e1_mask = Variable(torch.from_numpy(np.array(all_e1_mask))).type(torch.LongTensor)
e2_mask = Variable(torch.from_numpy(np.array(all_e2_mask))).type(torch.LongTensor)

# input_ids = torch.LongTensor(1, 128).to(torch.device("cpu"))
# attention_mask = torch.LongTensor(1, 128).to(torch.device("cpu"))
# token_type_ids = torch.LongTensor(1, 128).to(torch.device("cpu"))
# labels = torch.empty(1, 8)
# e1_mask = torch.LongTensor(1, 128).to(torch.device("cpu"))
# e2_mask = torch.LongTensor(1, 128).to(torch.device("cpu"))

# Export to ONNX format
torch.onnx.export(model, (input_ids, attention_mask, token_type_ids, labels,
                          e1_mask, e2_mask), './model/model_simple.onnx',
                  input_names=['input_ids',
                               'attention_mask',
                               'token_type_ids',
                               "e1_mask",
                               "e2_mask"],
                  output_names=['output'],
                  opset_version=10,
                  verbose=True,
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)