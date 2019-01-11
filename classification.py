import tensorflow as tf
from data_io_balance import DataSet
from utils import read_embedding
from textcnn import TextCNN
import os

class ModelHandler():
    """Build train process."""
    def __init__(self, flags):
        """Init class."""
        self.flags = flags
        self.embedding, self.embedding_size = read_embedding(
            self.flags.model_dir + self.flags.embedding_path)

    def add_tensor(self):
        """Add data and embeding."""
        self.train_dat = DataSet(self.flags.train_file,
                            self.flags.model_dir,
                            self.flags.batch_size,
                            self.flags.num_class,
                            self.flags.seq_length)

        iterator = self.train_dat.init_iterator()
        self.word_ids, self.word_label = iterator.get_next()
        self.dev_dat = DataSet(self.flags.dev_file,
                          self.flags.model_dir,
                          self.flags.batch_size,
                          self.flags.num_class,
                          self.flags.seq_length)
        self.train_data_init = iterator.make_initializer(self.train_dat.dataset)
        self.dev_data_init = iterator.make_initializer(self.dev_dat.dataset)

        print('add_dev_tensor')

    def train(self, sess, saver):
        """Train process."""
        self.step = 0
        best_accuracy = 0
        patient_passes = 0


        # sess.run(self.train_graph.embedding_init)
        for epoch in range(self.flags.epoch):

            self.train_dat.sample_data()
            sess.run(self.train_data_init)
            tf.local_variables_initializer().run()
            self.current_epoch = epoch
            print("epoch is :", epoch+1)
            self.train_epoch(sess, self.tf_graph)

            self.dev_dat.read_text(self)
            sess.run(self.dev_data_init)
            accuracy, losses = self.evaluate(sess, self.tf_graph)
            if accuracy < best_accuracy:
                patient_passes += 1
                if patient_passes == self.flags.patient_passes:
                    print("without improvement, break")
                    break
                else:
                    print("without improvement")
            else:
                print("new best acc {}".format(accuracy))
                best_accuracy = accuracy
                patient_passes = 0
                saver.save(sess, os.path.join(self.flags.model_dir, "model"),
                           global_step=self.step)

    def build_graph(self):
        """Build graph."""

        self.tf_graph = TextCNN(self.flags, self.embedding)
        self.tf_graph.build(self.word_ids, self.word_label)

    def train_epoch(self, sess, graph):
        """Operation in one epoch."""
        while True:
            self.step += 1
            try:
                _, loss, pred, ids, labels = sess.run(
                    [graph.train_op, graph.loss, graph.pred, graph.word_ids, graph.labels])

                if self.step % 10 == 0:
                    print("training epoch:{}, step:{}, loss:{}"
                          .format(self.current_epoch + 1, self.step, loss))
            except tf.errors.OutOfRangeError:
                print('finish')
                break

    def evaluate(self, sess, graph):
        """Evaluate process."""
        correct_preds = 0
        total_preds = 0
        accuracy = 0
        losses = 0
        while True:
            try:
                batch_correct_pred, pred, batch_loss = sess.run(
                    [graph.correct_pred, graph.pred, graph.loss])
                correct_preds += batch_correct_pred
                total_preds += pred.shape[0]
                losses += batch_loss * pred.shape[0]
            except tf.errors.OutOfRangeError:
                break
        accuracy = float(correct_preds / (total_preds+0.1))
        losses = float(losses / (total_preds+0.1))
        return accuracy, losses

    def predict(self):
        """Predict line."""
        word_id_list = tf.placeholder(tf.int32, shape=[None, None])
        model = TextCNN(self.flags, self.embedding)
        model.build_predictor(word_id_list)
        return model, word_id_list

