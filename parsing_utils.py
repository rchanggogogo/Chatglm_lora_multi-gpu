# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/4/10 4:20 PM
==================================="""
import json
import os

from tqdm import tqdm


# from unstructured.partition.text import partition_text
#
# elements = partition_text('/8t/workspace/lchang/models/data/only_jojo_with_instruction.txt')
# for element in elements:
#     print(element.text)
# print(elements)
# import sentence_transformers
# import torch
# from sklearn.preprocessing import normalize
# from sklearn.metrics.pairwise import cosine_similarity
# model = sentence_transformers.SentenceTransformer('/8t/workspace/lchang/models/paraphrase-multilingual-mpnet-base-v2')
#
# sentence1 = "小鸡叫叫IP的历程包括哪些重要的事件和成就?"
# sentence2 = "叫叫的理念是什么"
# sentence_embeding, sentence2_em = model.encode([sentence1, sentence2])
# print(sentence_embeding.shape)
# print(torch.cosine_similarity(torch.from_numpy(sentence_embeding.reshape(1,-1)), torch.from_numpy(sentence2_em.reshape(1,-1)), dim=1))
#
# sentence_embeding = normalize([sentence_embeding])
# sentence2_em = normalize([sentence2_em])
# print(cosine_similarity(sentence_embeding, sentence2_em))
#
#
# # coding=utf8
# sentence_embeding, sentence2_em = model.encode([sentence1, sentence2], normalize_embeddings=True)
# print(torch.cosine_similarity(torch.from_numpy(sentence_embeding.reshape(1,-1)), torch.from_numpy(sentence2_em.reshape(1,-1)), dim=1))
#

def txt_file_process(file_path):
    with open(file_path, 'r') as f1,\
            open('/8t/workspace/lchang/models/data/only_jojo_with_instruction.txt', 'w') as f3:
        final_result = []

        jojo_data = f1.readlines()
        for i, jojo in enumerate(jojo_data):
            jojo = eval(jojo)
            output = json.dumps(jojo['output'], ensure_ascii=False)
            f3.write(jojo['instruction']+'<i_am_split>' +output + '\n')


def load_text_from_file(file_path, split_by='<i_am_split>'):
    texts = []
    metadatas = []
    if isinstance(file_path, str):

        f = open(file_path, 'r', encoding='utf-8').readlines()
        source = file_path.split('/')[-1]
    else:
        f = open(file_path.name, 'r', encoding='utf-8').readlines()
        source = file_path.name.split('/')[-1]
    for line in tqdm(f, desc='load text from file'):
        if split_by in line:
            texts.append(line.split(split_by)[0].strip())

            metadatas.append({'source': source,
                              'content_': line.strip().replace(split_by, '')})
        else:
            texts.append(line.strip())
            metadatas.append({'source': file_path.split('/')[-1], 'content_': line.strip()})

    return texts, metadatas
