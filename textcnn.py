import tensorflow as tf
from model import Model

class TextCNN(Model):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    param:num_filter, linear_hidden_size, dropout, num_class, seq_length
    """
    def __init__(self, flags, embedding=None):
        Model.__init__(self, flags, embedding)

        print('finish graph init')

    def build_net(self):
        self.build_textcnn(self.embedded_chars)

    def build_textcnn(self, seq_embedding):
        """Build textcnn graph."""
        with tf.name_scope("text_cnn"):
            textcnn_cell_layer = self.textcnn_unit(
                input_x=seq_embedding,
                num_filter=self.flags.num_filter,
                scope_num=1)
            num_filters_total = self.flags.num_scope * self.flags.num_filter
            pooling_layer_flatten = tf.reshape(tensor=textcnn_cell_layer,
                                               shape=[-1, num_filters_total])
        with tf.name_scope("textcnn_to_fc"):
            linear_hidden = tf.layers.dense(
                inputs=pooling_layer_flatten,
                units=self.flags.linear_hidden_size,
                name="fc_output")
        with tf.name_scope("droupout"):
            linear_hidden_dropout = tf.layers.dropout(linear_hidden,
                                                      self.flags.dropout)
        with tf.name_scope("result"):
            self.logits = tf.layers.dense(linear_hidden_dropout,
                                          self.flags.num_class,
                                          name="scores")

    def textcnn_unit(self, input_x, num_filter, scope_num):
        with tf.name_scope("TextCnn-{}".format(scope_num)):
            branch1 = self.branch(input_x,
                                  num_filter,
                                  conv_size=3,
                                  scope_num=3)
            branch2 = self.branch(input_x,
                                  num_filter,
                                  conv_size=5,
                                  scope_num=5)
            branch3 = self.branch(input_x,
                                  num_filter,
                                  conv_size=7,
                                  scope_num=7)
            cell = tf.concat([branch1, branch2, branch3], axis=-1)
            # norm = tf.layers.batch_normalization(cell)
            result = tf.nn.relu(cell)
            return result

    #out channel  filter size for branch
    def branch(self, input_x, num_filter, conv_size, scope_num):
        with tf.name_scope("conv_branch_{}".format(scope_num)):
            #branch1 with 1 conv layer
            conv1 = tf.layers.conv1d(
                input_x,
                filters=num_filter,
                kernel_size=conv_size,
                padding="VALID"
            )
            # norm1 = tf.layers.batch_normalization(conv1)
            relu1 = tf.nn.relu(conv1)
            pool1 = tf.layers.max_pooling1d(
                relu1,
                pool_size=self.flags.seq_length - conv_size + 1,
                strides=1,
                padding="VALID"
            )
            return pool1

