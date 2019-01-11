import tensorflow as tf
import logging
import numpy as np

class Model():
    def __init__(self, flags, embedding=None, train_process=False):
        """Init param."""
        self.flags = flags
        self.embedding = embedding
        self.embedding_init =  tf.constant(self.embedding)


    def build(self, word_ids, word_lable):
       # with tf.na
        self.word_ids = word_ids
        self.labels = word_lable
        """Build tf graph."""
        with tf.variable_scope("net_variable", reuse=tf.AUTO_REUSE):
            self.add_embeddings()
            print('add_embedding')
            self.build_net()
            print('build_net')
            self.add_preds()
            print('add_preds')
            self.add_loss()
            self.add_train_op()
    # def build_acc_and_loss(self):
    #     """Build graph get loss and accuray."""
            self.add_accuarcy()
    def build_predictor(self, word_ids):
        with tf.variable_scope("net_variable", reuse=tf.AUTO_REUSE):
            self.word_ids = word_ids
            self.add_embeddings()
            self.build_net()
            self.add_preds()

    def add_embeddings(self):
        """Add embedding in graph."""
        # self.embedding_init = self.W.assign(self.embedding)
        # self.embedding_init = tf.get_variable(
        #     name="embedding",
        #     shape=[self.flags.embedding_num, self.flags.embedding_size],
        #     initializer=tf.constant_initializer(self.embedding))
        self.embedded_chars = tf.nn.embedding_lookup(self.embedding_init,
                                                     self.word_ids)

    def build_net(self):
        """Define tensor caculate process."""
        raise NotImplementedError("method build_network not implemented")

    def add_accuarcy(self):
        """Caculate accuarcy."""
        with tf.name_scope("accuracy"):
            correct_predict = tf.equal(tf.argmax(self.logits, 1),
                                       tf.argmax(self.labels, 1))
            self.correct_pred = tf.reduce_sum(tf.cast(correct_predict,
                                              tf.float32))
            self.acc = tf.reduce_mean(self.correct_pred, name="accuracy")

    def add_preds(self):
        """Get predict result."""
        with tf.name_scope("predict"):
            self.pred = tf.cast(tf.argmax(self.logits, axis=1),
                                tf.int32, name="label_pred")

    def add_loss(self):
        """Caculate loss."""
        with tf.name_scope("loss"):
            loss = tf.losses.softmax_cross_entropy(logits=self.logits,
                                                   onehot_labels=self.labels)
            self.loss = loss
            tf.summary.scalar('loss', self.loss)

    def add_train_op(self):
        """Define train optimizer and learning rate."""
        lr_method = self.flags.lr_method
        lr = self.flags.lr
        clip = self.flags.clip
        loss = self.loss
        _m = lr_method.lower()
        with tf.variable_scope('optimizer'):
            if _m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _m == 'adam':
                # Adadelta算法
                optimizer = tf.train.AdamOptimizer(lr)
            elif _m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError(
                    'Optimization method not supported:{}'.format(lr_method))
            if clip < 0:
                self.train_op = optimizer.minimize(loss)
            else:
                grads, variables = zip(*optimizer.compute_gradients(loss))
                grads_clipped, norm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads_clipped,
                                                              variables))
