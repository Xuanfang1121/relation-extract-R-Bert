# -*- coding: utf-8 -*-
# @Time    : 2021/3/27 17:42
# @Author  : zxf
import os

"""
    将数据集准备成r-bert模型输入的格式 label\tsent，其中sent中在实体1和实体2前后分别插入<e1></e1><e2></e2>
"""


def prepare_model_data(file_path, output_file):
    data = []
    i = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            entity1, entity2, label, sent = line.strip().split('\t')
            if entity1 in sent and entity2 in sent:
                entity1_index = sent.index(entity1)
                entity2_index = sent.index(entity2)
                sent = list(sent)
                if entity2_index >= entity1_index:
                    sent.insert(entity1_index, '<e1>')
                    sent.insert(entity1_index + len(entity1) + 1, '</e1>')
                    sent.insert(entity2_index + 2, '<e2>')
                    sent.insert(entity2_index + len(entity2) + 3, '</e2>')
                else:
                    sent.insert(entity2_index, '<e2>')
                    sent.insert(entity2_index + len(entity2) + 1, '</e2>')
                    sent.insert(entity1_index + 2, '<e1>')
                    sent.insert(entity1_index + len(entity1) + 3, '</e1>')
                sent = ''.join(sent)
                data.append(label + "\t" + sent)
                i += 1
                if i % 1000 == 0:
                    print("processing line: {}".format(i))

    with open(output_file, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line + "\n")

    print("data save finish")


if __name__ == "__main__":
    # file_path = "./data/relation_train.data"
    # output_file = "./data/train_ch.tsv"
    file_path = "./data/relation_val.data"
    output_file = "./data/test_ch.tsv"
    prepare_model_data(file_path, output_file)
