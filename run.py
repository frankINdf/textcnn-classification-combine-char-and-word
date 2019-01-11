from classification import ModelHandler
import tensorflow as tf
import numpy as np
from data_io_balance import parse_lines
import jpype
import time

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('embedding_num', 232800, 'word number in embedding ')
tf.flags.DEFINE_integer("num_class", 2, "num topic")
tf.flags.DEFINE_integer("linear_hidden_size", 100, "num topic")
tf.flags.DEFINE_integer("num_scope", 3, "num topic")
tf.flags.DEFINE_integer("seq_length", 20, "num topic")
tf.flags.DEFINE_integer("num_filter", 500, "num topic")
tf.flags.DEFINE_integer("batch_size", 256, "num topic")
tf.flags.DEFINE_integer("embedding_size", 300, "num topic")
tf.flags.DEFINE_float("dropout", 0.5, "num topic")
tf.flags.DEFINE_float("lr", 0.001, "num topic")
tf.flags.DEFINE_integer("clip", 1, "num topic")
tf.flags.DEFINE_integer("epoch", 1, "num topic")
tf.flags.DEFINE_integer("patient_passes", 3, "num topic")
tf.flags.DEFINE_integer("num_checkpoints", 1, "num topic")
tf.flags.DEFINE_string("lr_method", "adam", "num topic")
tf.flags.DEFINE_string("embedding_path", "embedding.pkl", "num topic")
tf.flags.DEFINE_string("train_file", "train.txt", "num topic")
tf.flags.DEFINE_string("dev_file", "test.txt", "num topic")
tf.flags.DEFINE_string("model_dir", "./model/navigation/", "num topic")

def train(flags):
    """Train."""
    model = ModelHandler(flags)
    model.add_tensor()
    model.build_graph()
    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer().run()
        all_v =tf.global_variables()
        for t in all_v:
            print(t)
        writer = tf.summary.FileWriter("./model/navigation/logs/", sess.graph)
        model.train(sess, saver)


def test(flags, path):
    """Predict line。"""
    jvmPath = jpype.getDefaultJVMPath()
    jpype.startJVM(jvmPath, '-Djava.class.path=C:\ProgramData\Anaconda3\Lib\site-packages\pyhanlp\static\hanlp-1.6.8.jar;C:\ProgramData\Anaconda3\Lib\site-packages\pyhanlp\static')
    HanLP = jpype.JClass('com.hankcs.hanlp.HanLP')

    model_output = flags.model_dir + "model/"
    handler = ModelHandler(flags)
    # handler.add_tensor()
    # handler.build_graph()
    model, word_ids = handler.predict()
    content = []
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer().run()
        # print('*'* 100)
        # print(tf.train.latest_checkpoint(flags.model_dir))
        saver.restore(sess, tf.train.latest_checkpoint("./model/navigation/model/"))
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        start = time.time()
        for lines in lines:
            text = line.split(',')[0]
            cut = HanLP.segment(text)
            split_line = ' '.join([w.word for w in cut])
            word_id = parse_line(split_line, FLAGS)
            word_id = np.expand_dims(word_id, axis=0)
            pred = sess.run(model.pred, {word_ids: word_id})
            content.append("{},{}\n".format(line.strip('\n'),pred[0]))
            if len(content)%500 == 0:
                end = time.time()
                print("start:{},end:{},per{}".format(start, end, float((end-start)/500)))
                start = end
                with open(path[:-4] + 'out.csv', "a+", encoding="utf-8") as f:
                    f.writelines(content)
                content = []

        with open(path[:-4] + 'out.csv', "a+", encoding="utf-8") as f:
            f.writelines(content)

def main(flags):

    train(flags)
    # train_restore(flags)
    path = "./测试数据/test_data.csv"
    # test(flags, path)

if __name__ == '__main__':

    main(FLAGS)
    # r = parse_line("试 一", FLAGS)
    # print(r)
