# -*- coding: utf-8 -*-
# @Time    : 2021/3/27 15:58
# @Author  : zxf
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--task", default="semeval", type=str, help="The name of the task to train")
parser.add_argument(
    "--data_dir",
    default="./data",
    type=str,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
)
parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
parser.add_argument(
    "--eval_dir",
    default="./eval",
    type=str,
    help="Evaluation script, result directory",
)
parser.add_argument("--train_file", default="train_ch_demo.tsv", type=str,
                    help="Train file")
parser.add_argument("--test_file", default="test_ch_demo.tsv", type=str,
                    help="Test file")
parser.add_argument("--label_file", default="relation_label.txt", type=str,
                    help="Label file")

parser.add_argument(
    "--model_name_or_path",
    type=str,
    # default="/work/zhangxf/torch_pretraining_model/bert-base-uncased",
    # default="/work/zhangxf/torch_pretraining_model/bert-base-chinese",
    # default='D:/Spyder/pretrain_model/transformers_torch_tf/bert_base_uncased',
    default='D:/Spyder/pretrain_model/transformers_torch_tf/bert_base_chinese',
    help="Model Name or Path",
)

parser.add_argument("--seed", type=int, default=77,
                    help="random seed for initialization")
parser.add_argument("--train_batch_size", default=16, type=int,
                    help="Batch size for training.")
parser.add_argument("--eval_batch_size", default=16, type=int,
                    help="Batch size for evaluation.")
parser.add_argument(
    "--max_seq_len",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization.",
)
parser.add_argument(
    "--learning_rate",
    default=2e-5,
    type=float,
    help="The initial learning rate for Adam.",
)
parser.add_argument(
    "--num_train_epochs",
    default=2.0,
    type=float,
    help="Total number of training epochs to perform.",
)
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument(
    "--dropout_rate",
    default=0.1,
    type=float,
    help="Dropout for fully-connected layers",
)

parser.add_argument("--logging_steps", type=int, default=10,   # 250
                    help="Log every X updates steps.")
parser.add_argument(
    "--save_steps",
    type=int,
    default=10,
    help="Save checkpoint every X updates steps.",
)

# parser.add_argument("--do_train", action="store_true",
#                     help="Whether to run training.")
# parser.add_argument("--do_eval", action="store_true",
#                     help="Whether to run eval on the test set.")

parser.add_argument("--do_train", type=bool, default=True,
                    help="Whether to run training.")
parser.add_argument("--do_eval", type=bool, default=True,
                    help="Whether to run eval on the test set.")

parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
parser.add_argument(
    "--add_sep_token",
    action="store_true",
    help="Add [SEP] token at the end of the sentence",
)

args = parser.parse_args()

# # "The name of the task to train"
# task = "semeval"
# # "The input data dir. Should contain the .tsv files (or other data files) for the task."
# data_dir = "./data"
# # Path to model
# model_dir = "./model"
# # Evaluation script, result directory
# eval_dir = "./eval"
# # Train file
# train_file = "train_ch_demo.tsv"
# # Test file
# test_file = "test_ch_demo.tsv"
# # Label file
# label_file = "relation_label.txt"
# # Model Name or Path
# # model_name_or_path = "/work/zhangxf/torch_pretraining_model/bert-base-uncased"
# # model_name_or_path = "/work/zhangxf/torch_pretraining_model/bert-base-chinese"
# model_name_or_path = 'D:/Spyder/pretrain_model/transformers_torch_tf/bert_base_chinese'
# # random seed for initialization
# seed = 77
# # Batch size for training.
# train_batch_size = 32
# # Batch size for evaluation.
# eval_batch_size = 32
# # The maximum total input sequence length after tokenization.
# max_seq_len = 128
# # The initial learning rate for Adam.
# learning_rate = 2e-5
# # Total number of training epochs to perform.
# num_train_epochs = 5.0
# # Weight decay if we apply some.
# weight_decay = 0.0
# # Number of updates steps to accumulate before performing a backward/update pass.
# gradient_accumulation_steps = 1
# # Epsilon for Adam optimizer.
# adam_epsilon = 1e-8
# # Max gradient norm.
# max_grad_norm = 1.0
# # If > 0: set total number of training steps to perform. Override num_train_epochs.
# max_steps = -1
# # Linear warmup over warmup_steps
# warmup_steps = 0
# # Dropout for fully-connected layers
# dropout_rate = 0.1
# # Log every X updates steps.
# logging_steps = 10
# # Save checkpoint every X updates steps.
# save_steps = 10
# # Whether to run training.
# do_train = True
# # Whether to run eval on the test set.
# do_eval = True
# # Avoid using CUDA when available
# no_cuda = False
# # Add [SEP] token at the end of the sentence
# add_sep_token = False
