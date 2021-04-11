# -*- coding: utf-8 -*-
import argparse

from rbert_config import args
from trainer_new import Trainer  # 取消学习率衰减
from data_loader import load_and_cache_examples
from utils import init_logger, load_tokenizer, set_seed


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset=train_dataset, test_dataset=test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("--task", default="semeval", type=str, help="The name of the task to train")
    # parser.add_argument(
    #     "--data_dir",
    #     default="./data",
    #     type=str,
    #     help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    # )
    # parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    # parser.add_argument(
    #     "--eval_dir",
    #     default="./eval",
    #     type=str,
    #     help="Evaluation script, result directory",
    # )
    # parser.add_argument("--train_file", default="train_ch_demo.tsv", type=str,
    #                     help="Train file")
    # parser.add_argument("--test_file", default="test_ch_demo.tsv", type=str,
    #                     help="Test file")
    # parser.add_argument("--label_file", default="relation_label.txt", type=str,
    #                     help="Label file")
    #
    # parser.add_argument(
    #     "--model_name_or_path",
    #     type=str,
    #     # default="/work/zhangxf/torch_pretraining_model/bert-base-uncased",
    #     # default="/work/zhangxf/torch_pretraining_model/bert-base-chinese",
    #     # default='D:/Spyder/pretrain_model/transformers_torch_tf/bert_base_uncased',
    #     default='D:/Spyder/pretrain_model/transformers_torch_tf/bert_base_chinese',
    #     help="Model Name or Path",
    # )
    #
    # parser.add_argument("--seed", type=int, default=77,
    #                     help="random seed for initialization")
    # parser.add_argument("--train_batch_size", default=32, type=int,
    #                     help="Batch size for training.")
    # parser.add_argument("--eval_batch_size", default=32, type=int,
    #                     help="Batch size for evaluation.")
    # parser.add_argument(
    #     "--max_seq_len",
    #     default=128,
    #     type=int,
    #     help="The maximum total input sequence length after tokenization.",
    # )
    # parser.add_argument(
    #     "--learning_rate",
    #     default=2e-5,
    #     type=float,
    #     help="The initial learning rate for Adam.",
    # )
    # parser.add_argument(
    #     "--num_train_epochs",
    #     default=5.0,
    #     type=float,
    #     help="Total number of training epochs to perform.",
    # )
    # parser.add_argument("--weight_decay", default=0.0, type=float,
    #                     help="Weight decay if we apply some.")
    # parser.add_argument(
    #     "--gradient_accumulation_steps",
    #     type=int,
    #     default=1,
    #     help="Number of updates steps to accumulate before performing a backward/update pass.",
    # )
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float,
    #                     help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    # parser.add_argument(
    #     "--max_steps",
    #     default=-1,
    #     type=int,
    #     help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    # )
    # parser.add_argument("--warmup_steps", default=0, type=int,
    #                     help="Linear warmup over warmup_steps.")
    # parser.add_argument(
    #     "--dropout_rate",
    #     default=0.1,
    #     type=float,
    #     help="Dropout for fully-connected layers",
    # )
    #
    # parser.add_argument("--logging_steps", type=int, default=10,   # 250
    #                     help="Log every X updates steps.")
    # parser.add_argument(
    #     "--save_steps",
    #     type=int,
    #     default=10,
    #     help="Save checkpoint every X updates steps.",
    # )
    #
    # # parser.add_argument("--do_train", action="store_true",
    # #                     help="Whether to run training.")
    # # parser.add_argument("--do_eval", action="store_true",
    # #                     help="Whether to run eval on the test set.")
    #
    # parser.add_argument("--do_train", type=bool, default=True,
    #                     help="Whether to run training.")
    # parser.add_argument("--do_eval", type=bool, default=True,
    #                     help="Whether to run eval on the test set.")
    #
    # parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    # parser.add_argument(
    #     "--add_sep_token",
    #     action="store_true",
    #     help="Add [SEP] token at the end of the sentence",
    # )
    #
    # args = parser.parse_args()

    main(args)
