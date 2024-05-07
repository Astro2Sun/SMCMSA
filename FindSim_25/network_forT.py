import tensorflow as tf
import numpy as np
from Settings import Config
from module import ff, multihead_attention, ln, mask, SigmoidAtt
import sys
from tensorflow.python.keras.utils import losses_utils


class MM:
    def __init__(self, is_training):
        self.temp = 'null'
        self.config = Config()
        self.att_dim = self.config.att_dim      # 300
        self.text = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, self.config.max_text_len, 768],name='text')     # 32 * 25 * 768
        self.label = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size], name='label')       # （32，）
        text = tf.layers.dense(self.text, self.config.att_dim, use_bias=False)      # 32 * 25 * 300

        with tf.variable_scope('tt', reuse=tf.AUTO_REUSE):
            enc_tt = multihead_attention(queries=text,      # 32 * 25 * 300
                                         keys=text,
                                         values=text,
                                         num_heads=4,
                                         dropout_rate=0.2,
                                         training=True,
                                         causality=False)
            enc_tt = ff(enc_tt, num_units=[4 * self.config.att_dim, self.config.att_dim])       # # 32 * 25 * 300

        with tf.variable_scope('all_weights', reuse=tf.AUTO_REUSE):
            Wr_wq = tf.get_variable('Wr_wq', [300, 1])
            Wm_wq = tf.get_variable('Wm_wq', [300, 300])
            Wu_wq = tf.get_variable('Wu_wq', [300, 300])
            W_l = tf.get_variable('W_l', [self.att_dim, self.config.class_num])
            b_l = tf.get_variable('b_l', [1, self.config.class_num])

        enc_new = tf.convert_to_tensor(enc_tt)      # 32 * 25 * 300

        with tf.variable_scope('de', reuse=tf.AUTO_REUSE):
            enc_de = multihead_attention(queries=enc_new,       # 32 * 25 * 300
                                         keys=enc_new,
                                         values=enc_new,
                                         num_heads=4,
                                         dropout_rate=0.2,
                                         training=True,
                                         causality=False)
            enc_de = ff(enc_de, num_units=[4 * 300, 300])       # 32 * 25 * 300

        # encode kl loss
        kl = tf.keras.losses.KLDivergence(reduction=losses_utils.ReductionV2.NONE, name='kl')
        de_loss1 = kl(tf.nn.softmax(enc_de, -1), tf.nn.softmax(text, -1))
        de_loss2 = kl(tf.nn.softmax(text, -1), tf.nn.softmax(enc_de, -1))
        self.de_loss = tf.reduce_sum(tf.reduce_mean(de_loss1, -1), -1) + tf.reduce_sum(tf.reduce_mean(de_loss2, -1), -1)

        outputs_en = SigmoidAtt(enc_tt, Wr_wq, Wm_wq, Wu_wq)        # 32 * 300

        self.output = outputs_en

        temp_new = outputs_en
        temp_new = tf.layers.dense(temp_new, self.config.att_dim, use_bias=False)       # 32 * 300
        self.temp_new_todowndim = temp_new

        output_res = tf.add(tf.matmul(temp_new, W_l), b_l)
        ouput_label = tf.one_hot(self.label, self.config.class_num)
        self.prob = tf.nn.softmax(output_res)

        with tf.name_scope('loss'):
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=output_res, labels=ouput_label))
            self.loss = loss
            self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                                  weights_list=[W_l, b_l])
            self.total_loss = self.loss + self.l2_loss + 0.1 * self.de_loss

