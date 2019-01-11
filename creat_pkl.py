#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 CloudBrain <byzhang@>
#
# Distributed under terms of the CloudBrain license.

"""
create pkl
"""
import pickle
import random
import utils
def word_to_index():
    word = "我是一个中国人，这是个测试的词典"
    word_index_dict = {}
    for i, char in enumerate(word):
        word_index_dict[char] = i + 2
    return word_index_dict

def embedding_list():
    embedding = []
    for i in range(200):
        rand_list = [random.random() for x in range(300)]
        print(len(rand_list))
        embedding.append(rand_list)
    print(len(embedding))
    return embedding

def save_pkl(obj, out):
    with open(out, 'wb+') as f:
        pickle.dump(obj, f)

if __name__ == "__main__":
    path = './data/'
    word_index = word_to_index()
    embedding = embedding_list()
    save_pkl(word_index, path+'word_to_index.pkl')
    save_pkl(embedding, path+'embedding.pkl')




