import os
import random
import numpy as np
import pickle
import tensorflow as tf
class DataSet():
    """
    load dict->init_iterator->yield batch
    """
    def __init__(self,
                 text_path,
                 model_dir,
                 batch_size=10,
                 num_class=2,
                 seq_length=20,
                 shuffle=False,
                 shuff_buffer_size=10):
        """
        Init class, create dataset, load word to index.

        :param:word_to_index_path: store word to index dict, dict
        :param:text_path, text after split word, string
        :param:seq_length, max input length, int
        :param: shuffle, if shuffle data or not, bool
        :param: shuffle_buffer_size
        """
        self.word_to_index_path = model_dir + 'word_to_ind.pkl'
        self.seq_length = seq_length
        self.negative_path = model_dir + "negative.txt"
        self.navigation_path = model_dir + "navigation.txt"
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_class = num_class
        self.shuff_buffer_size = shuff_buffer_size
        self.unk_word = []
        with open(self.word_to_index_path, 'rb') as f:
            self.word_to_index = pickle.load(f)
        self.dataset = tf.data.Dataset.from_generator(
            self.__iter__,
            output_types=(tf.int32, tf.int32),
            output_shapes=(
                (self.seq_length),
                (self.num_class))).batch(self.batch_size)
        if self.shuffle:
            dataset = self.dataset.shuffle(self.shuff_buffer_size)
        # self.dataset = dataset

    def sample_data(self):
        with open(self.negative_path, 'r', encoding='utf-8') as f:
            negative_text = [l.strip('\n') + "\x010" for l in f.readlines()]
        with open(self.navigation_path, 'r', encoding='utf-8') as f:
            navigation_text = [l.strip('\n') + "\x011" for l in f.readlines()]
        end = max(len(navigation_text), 5 * len(negative_text))
        random.shuffle(negative_text)
        self.text = negative_text + navigation_text[:end]
        random.shuffle(self.text)

    def read_text(self):
        with open(self.text_path, 'r', encoding='utf-8') as f:
            self.text = f.readlines()

    def __iter__(self):
        """
        Create iter, yeild data input.

        return: word_index_list represent word,word_freq_matrix count word freq
        """
        for line in self.text:
            yield self.handle_line(line)

        unk_path = 'unk_char.txt'
        print('unk save in:{}'.format(unk_path))
        with open(unk_path, 'w+', encoding='utf-8') as f:
            f.writelines(self.unk_word)

    def init_iterator(self):
        """
        Return iterator though dataset, dataset init.

        :param:dataset: tensorflow dataset, initialize iterator
        """
        iterator = tf.data.Iterator.from_structure(
            output_types=self.dataset.output_types,
            output_shapes=self.dataset.output_shapes)
        return iterator

    def handle_line(self, text):
        """Parse single line."""
        unk_id = 0
        pad_id = 1
        word_id_list = [pad_id] * self.seq_length
        if len(text.split('\x01')) != 2:
            print("line len not 2")
            return False
        line, label = text.split('\x01')
        label = int(label)

        for i, char in enumerate(line.split()):
            if i >= self.seq_length:
                break
            try:
                if char not in self.word_to_index.keys():
                    self.unk_word.append(char + '\n')
                word_id_list[i] = self.word_to_index.get(char, unk_id)
            except IndexError:
                print('i is ', i)
                break
        one_hot_label = [0] * self.num_class
        one_hot_label[label] = 1

        return word_id_list, one_hot_label

def parse_lines(text, flags):
    with open(flags.model_dir + 'word_to_ind.pkl', 'rb') as f:
        word_to_index = pickle.load(f)
    unk_id = 0
    pad_id = 1

    batch_data = []
    for line in text:
        word_id_list = [pad_id] * flags.seq_length
        for i, char in enumerate(line.split()):
            # print(word_id_list)
            if i == flags.seq_length:
                return np.array(word_id_list)
            else:
                word_id_list[i] = word_to_index.get(char, unk_id)
        batch_data.append(word_id_list)
    if len(batch_data) == flags.batch_size:
        yield np.array(batch_data).reshape([-1, flags.seq_length])
        batch_data = []
    yield np.array(batch_data).reshape([-1, flags.seq_length])

def main():
    model_dir = 'D:\\project\\nav\model\\navigation\\'
    text_path = 'D:\\project\\nav\model\\navigation\\test.txt'

    dataset = DataSet('test.txt', model_dir)
    dataset.load_dict()
    iterator_init = dataset.init_iterator()
    t = dataset.yield_batch()
    with tf.Session() as sess:
        sess.run(iterator_init)
        while True:
            try:
                a, b = sess.run(t)
                # print(a, b)
            except tf.errors.OutOfRangeError:
                print("stop")
                break

if __name__ == '__main__':
    main()
