import tensorflow as tf
import os
from datetime import datetime
from utils import init_logger
from utils import get_batch,read_embedding
class ModelHandler(object):
    def __init__(self, flags):
        self.flags = flags
        self.model_dir = '{}/{:%Y%m%d_%H%M%S}/'.format(self.flags.model_output,datetime.now())
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.logger = init_logger(os.path.join(self.model_dir,"model_train.log"))
        self.logger.info("output path:{}".format(self.model_dir))
    def add_summary(self, graph):

        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.model_dir, graph)

    def attach_session(self, sess):
        self.sess = sess

    def train_epoch(self, epcoh, train, dev):
        raise NotImplementedError("Method train_epcoh not implemented")

    def train(self, sess, saver):
        self.logger.info("training model")
        best_accuracy = 0
        patient_passes = 0
        self.logger.info("init summary")
        self.add_summary(sess.graph)
        self.logger.info("init session")
        self.attach_session(sess)
        embedding,voc_size = read_embedding(self.flags.embedding_data)
        _ = sess.run(self.embedding_init,{self.embedding_placeholder:embedding})

        for epoch in range(self.flags.epoch):
            self.logger.info("Running epoch {} of {}".format(epoch+1, self.flags.epoch))
            accuracy, loss = self.train_epoch(epoch, embedding, saver)
            self.logger.info('accuracy accuracy on dev {} loss on dev {}'.format(accuracy, loss))
            if accuracy <= best_accuracy:
                patient_passes +=1
                if patient_passes == self.flags.patient_passes:
                    self.logger.info(' - {} epochs without improvement, training stopped'.format(patient_passes))
                    break
            else:
                self.logger.info('- New best accuracy {}'.format(accuracy))
                best_accuracy = accuracy
                patient_passes = 0
                saver.save(sess, os.path.join(self.model_dir, "model"), global_step = self.step)

    # def restore(self):
    #       self.logger.info("loading model")
    #   self.attach_session(sess)
    #   saver.restore(sess, self.flags.model_output)
if __name__ == "__main__":
    eb,i = read_embedding('./data/embedding.pkl')
    print(eb[4])
