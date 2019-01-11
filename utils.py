import logging
import os
import re
import jpype
import random
import pickle
import tensorflow as tf
import numpy as np
import sys
import glob
def init_logger(logfile):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(logfile)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


class TFRecordUtils(object):
    """
    Only save file as tfrecord.
    """
    def __init__(self):
        self.word2index_path = word2index_path
        self.corpus_path = corpus_path
        self.tfrecord_out_path
        self.num_class = 33
        self.seq_length = 35
        self.pad_id = 0
        self.unk_id = 1
        self.spllit_char = "\x01"
        self.num_fold = 10
    def read_word2index(self):
        with open(self.word2index_path,"rb") as f:
            self.word2index = pickle.load(f)

    def word_to_index(self,line_split,title_position,word2index):
        line_split = line_split[title_position].split()
        word_index=[]
        for word in line_split:
        #not long enough and unk
            if word not in word2index.keys():
                word_index.append(unk_id)
            else:
                word_index.append(word2index[word])
        if len(word_index) > seq_length:
            word_index = word_index[0:seq_length]
        else:
            word_index = word_index+[pad_id]*(seq_length-len(word_index))
        return word_index

    def write_tfrecord(self, word2index, lines, tfrecord_path):
        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        for line in lines:
            line = line.replace("\n","")
            if split_char not in line:
                print("please examine split char in: \n{}".format(line))
            line_split = line.split(split_char)
            label = num_class*[0]
            try:
                label[int(line_split[1])] = 1
            except IndexError:
                print(line_split)
            word_index = self.word_to_index(line_split,0,self.word2index)

            record = tf.train.Example(features = tf.train.Features(
                feature = {
                "input": tf.train.Feature(int64_list=tf.train.Int64List(value=word_index)),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label))
                }))

            serialized = record.SerializeToString()
            writer.write(serialized)
            word_index=[]
        writer.close()

    def corpus2tfrecord(self, word2index, corpus_path, tfrecord_path, fold, num_class):
        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print("word_and_char_data:",len(lines))
            seed = 100
            random.seed(seed)
            random.shuffle(lines)
            total_num = len(lines)
            print("number of total train data:", total_num)
            test_num = int(total_num/fold)
            print("number of test data:", test_num)
            train_num = test_num*(fold-1)
            print("number of train data:", train_num)
            print("write train record")
            write_tfrecord(word2index, lines[:train_num], tfrecord_path+"train")
            print("write test record")
            write_tfrecord(word2index, lines[-test_num:], tfrecord_path+"test")
        print("finish save")

    def get_batch(self, record_file_path, epoch, batch_size, shuffle=False):
        dataset = tf.data.TFRecordDataset(record_file_path)
        dataset = dataset.map(_parse_function)
        if shuffle:
            dataset = dataset.shuffle(10)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(epoch)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def _parse_function(self, example_proto, seq_length=50, num_class=33):
        features = {
            "input": tf.FixedLenFeature([seq_length], tf.int64),
            "label": tf.FixedLenFeature([num_class], tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features['input'], parsed_features['label']

class PredictDataUtils(object):
    def __init__(self):
        self.word2index_path = word2index_path
        self.predict_data_path= predict_data_path
        self.seq_length = 35
        self.pad_id = 0
        self.unk_id = 1
        self.spllit_char = "\x01"
        self.title_position = 0
        self.min_title_length = 3

    def data_preprocess(self,data_path):
        with open(data_path,"r",encoding="utf-8") as f:
            lines = f.readlines()
        jvmPath = jpype.getDefaultJVMPath()
        jpype.startJVM(jvmPath, '-Djava.class.path=C:\ProgramData\Anaconda3\Lib\site-packages\pyhanlp\static\hanlp-1.6.8.jar;C:\ProgramData\Anaconda3\Lib\site-packages\pyhanlp\static')
        HanLP = jpype.JClass('com.hankcs.hanlp.HanLP')
        for i,org_line in enumerate(lines):
            line_split = org_line.split(self.split_char)
            title_line = line_split[self.title_position]
            if len(title_line) < self.min_title_length:
                continue
            title_line = remove_punctuation(title_line)
            try:
                title_split = split_line(title_line)
                word_char_line = unk_word_to_char(title_split,word2index)
                word_index = word_to_index(word_char_line, self.word2index)
                batch_word_index = []
                if i % self.batch_size == 0:
                    yield batch_word_index
                    batch_word_index = []

            except IndexError:
                print(org_line)
        if len(batch_word_index) > 0:
            yield batch_word_index


    def read_word2index(self):
        with open(self.word2index_path,"rb") as f:
            self.word2index = pickle.load(f)

    def word_to_index(self, line_split, word2index):
        word_index = []
        for word in line_split:
        #not long enough and unk
            if word not in word2index.keys():
                word_index.append(self.unk_id)
            else:
                word_index.append(word2index[word])
        if len(word_index) > seq_length:
            word_index = word_index[0:self.seq_length]
        else:
            word_index = word_index+[self.pad_id]*(self.seq_length-len(word_index))
        return word_index


    def remove_punctuation(self, line): #-\"
        try:
            rule = re.compile(u"[^·\x01\t \b\uFF09\uFF08\u300A\u300B\u2013\.\t\{\}a-zA-Z0-9\u4e00-\u9fa5]")
            line = rule.sub('',line)
        except:
            print(line)
        return line

    def unk_word_to_char(self,line,word2index):
        word_char_line = list()
        for word in line:
            if word not in word2index.keys():
                for char in word:
                    word_char_line.append(char)
            else:
                word_char_line.append(word)
        return word_char_line



# def transform_word2index(input_file, seq_length, word2ind_path):
    # pad_id = 0
    # unk_id = 1
    # with open(word2ind_path,"rb") as f:
        # word2ind_dict = pickle.load(f)
    # word_index = list()
    # with open(input_file, "r", encoding="utf-8") as f:
        # lines = f.readlines()
        # for line in lines:
            # line_split = line.split("\x01")
            # line_list = line_to_word_and_char(line_split[1],word2ind_dict)
            # index = [word2ind_dict.get(word,unk_id) for word in line_list]
            # if len(index) > seq_length:
                # index = index[0:seq_length]
            # else:
                # index = index+[pad_id]*(seq_length-len(index))
            # word_index.append(index)
    # return word_index,lines
# def line_to_word_and_char(line, word2index):
    # line_list = []
    # for word in line.split():
        # if word in word2index.keys():
            # line_list.append(word)
        # elif word not in word2index.keys():
            # for char in word:
                # line_list.append(char)
    # return line_list



def batch_yield(data,batch_size):
    data_num = len(data)
    iterator_num = int((data_num-1)/batch_size) + 1
    for batch_num in range(iterator_num):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_num)
        yield data[start_index:end_index]


def read_embedding(path):
    with open(path, "rb") as f:
        embedding = pickle.load(f)
        print(len(embedding))
    return embedding, len(embedding)


def predict_data_transform(path, word2index_path, path_out,delimter=","):
    handler_line = ""
    with open(word2index_path,"rb") as f:
        word2index = pickle.load(f)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for org_line in lines:
            line = remove_punctuation(org_line)
            line_split = line.split(delimter)
            try:
                word_list = line_split[0].split()
                handler_line += line_split[1]+delimter
                for word in word_list:
                    if word not in word2index.keys():
                        for char in word:
                            handler_line += " "+char
                    else:
                        handler_line += " "+word
                handler_line += "\n"
            except IndexError:
                print(org_line)
    with open(path_out,"w+",encoding="utf-8") as f:
        f.write(handler_line)
def splited_data_to_predict(path,word2ind_path,path_out):
    with open(word2ind_path, "rb") as f:
        word2index = pickle.load(f)
    with open(path,"r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split()
            for word in line_split:
                if word not in word2index.keys():
                    for char in word:
                        handler_line += " " + char
                else:
                    handler_line += " " + word
            handler_line += "\n"
    with open(path_out,"w+", encoding="utf-8") as f:
        f.write(handler_line)

def copus_to_input():
    corpus_path = r"D:\\project\\categorey textcnn\\copus\\nocut_copy1\\"
    merged_corpus_path = r'./data/corpus_word_merge_v2.txt'
    label2ind_path = r"./data/label2ind.pkl"
    word2index_path = r"./data/word2ind_dct.pkl"
    char_word_corpus = r"./data/trans_copus_out_v2.txt"
    tfrecord_path = './data/word_tfrecord_v2.record'
    #merge 1 file
    if os.path.exists(merged_corpus_path):
        os.remove(merged_corpus_path)
    if os.path.exists(label2ind_path):
        os.remove(label2ind_path)
    data_merge(corpus_path, merged_corpus_path , label2ind_path)
    #split unk
    if os.path.exists(char_word_corpus):
        os.remove(char_word_corpus)
    trans_to_char_and_word(word2index_path,merged_corpus_path,char_word_corpus)
    if os.path.exists(tfrecord_path+"train") :
        os.remove(tfrecord_path+"train")
    if os.path.exists(tfrecord_path+"test"):
        os.remove(tfrecord_path+"test")
    corpus2tfrecord(word2index_path, char_word_corpus, tfrecord_path, fold=8, num_class=33)




if __name__ == "__main__":

    #get_batch('./char_test.record', 1, 2, shuffle=False)
    corpus_path = r"D:\\project\\categorey textcnn\\copus\\nocut\\"
    merged_corpus_path = r'./corpus_word_merge.txt'
    char2ind_path = "./char2index.pickle"
    word2ind_path = "./data/word2ind_dct.pkl"
    embedding_path = "./data/char_embedding.pkl"
    #random embedding
    # with open(char2ind_path,"rb") as f:
    # 	obj = pickle.load(f)
    # 	char_num = len(obj.keys())
    # 	print("char_num",char_num)
    # 	print("obj keys",list(obj.items()))
    # 	random_char_embedding(300, char_num, embedding_path)


    #char_data_merge(corpus_path, merged_corpus_path, char2ind_path)
    #split
    jvmPath = jpype.getDefaultJVMPath()
    jpype.startJVM(jvmPath, '-Djava.class.path=C:\ProgramData\Anaconda3\Lib\site-packages\pyhanlp\static\hanlp-1.6.8.jar;C:\ProgramData\Anaconda3\Lib\site-packages\pyhanlp\static')
    HanLP = jpype.JClass('com.hankcs.hanlp.HanLP')

    #copus_to_input()
    # Config = jpype.JClass('com.hankcs.hanlp.HanLP$Config')
    # Config.ShowTermNature=False
    #data_merge(corpus_path, merged_corpus_path, "./label2ind.pkl")
    #corpus2tfrecord(r'word2ind_dct.pkl',r'./corpus_merge2.txt','./tfrecord_texst.record')

    # char_data_merge(corpus_path, merged_corpus_path, char2ind_path)
    #corpus2tfrecord(word2ind_path,merged_corpus_path,'./word_tfrecord_test.record')
    #corpus2tfrecord(char2ind_path,merged_corpus_path,'./char_tfrecord.record')
    #生成word训练集和验证集
    #word2ind_path = "./data/word2ind_dct.pkl"
    #merged_corpus_path = r'./corpus_word_merge.txt'
    #corpus2tfrecord(word2ind_path, merged_corpus_path, './word_tfrecord.record', fold=8, num_class=33)
    #corpus2tfrecord(word2ind_path, merged_corpus_path, './word_test3000_tfrecord.record', fold=8, num_class=33)
    char2ind_path = 'char2index.pickle'
    merged_corpus_path = 'corpus_char_merge.txt'
    #corpus2tfrecord(char2ind_path, merged_corpus_path, './char_tfrecord.record', fold=9, num_class=33)
    #char_tfrecord.record
    #测试读取
    # iterator = get_batch('word_test3000_tfrecord.recordtest',1,100,)
    # nextelement = iterator.get_next()
    # with tf.Session() as sess:
    # 	while True:
    # 		try:
    # 			input_,label = sess.run(nextelement)
    # 			print("input",input_[0])
    # 			print("label",label)
    # 		except tf.errors.OutOfRangeError:
    # 			break
    # 	print("finish")
    corpus_path = './data/corpus_word_merge_out.txt'
    tfrecord_path  = './data/word_tfrecord_merge.record'
    #corpus2tfrecord(word2ind_path, corpus_path, tfrecord_path, fold=8, num_class=33)
    pred_path = './data/std_val_dataset_1022.csvcp'
    #predict_data_transform(pred_path, word2ind_path)
    with open("label2ind.pkl","rb") as f:
        obj = pickle.load(f)
        print(obj)




