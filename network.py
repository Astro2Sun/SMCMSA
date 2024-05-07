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
        self.att_dim = self.config.att_dim
        self.visual = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, self.config.max_visual_len, 709], name='visual')
        self.audio = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, self.config.max_audio_len, 33], name='audio')
        self.text = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, self.config.max_text_len, 768], name='text')
        self.label = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size], name='label')
        self.flag = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size], name='flag')
        self.pretrained_output = tf.placeholder(dtype = tf.float32, shape = [self.config.batch_size, 275, 300], name = 'pre')

        visual = tf.layers.dense(self.visual, self.config.att_dim, use_bias=False)
        audio = tf.layers.dense(self.audio, self.config.att_dim, use_bias=False)
        text = tf.layers.dense(self.text, self.config.att_dim, use_bias=False)

        with tf.variable_scope('vv', reuse=tf.AUTO_REUSE):
            enc_vv = multihead_attention(queries=visual,
                                         keys=visual,
                                         values=visual,
                                         num_heads=4,
                                         dropout_rate=0.2,
                                         training=True,
                                         causality=False)
            enc_vv = ff(enc_vv, num_units=[4 * self.config.att_dim, self.config.att_dim])

        with tf.variable_scope('aa', reuse=tf.AUTO_REUSE):
            enc_aa = multihead_attention(queries=audio,
                                         keys=audio,
                                         values=audio,
                                         num_heads=4,
                                         dropout_rate=0.2,
                                         training=True,
                                         causality=False)
            enc_aa = ff(enc_aa, num_units=[4 * self.config.att_dim, self.config.att_dim])

        with tf.variable_scope('tt', reuse=tf.AUTO_REUSE):
            enc_tt = multihead_attention(queries=text,
                                         keys=text,
                                         values=text,
                                         num_heads=4,
                                         dropout_rate=0.2,
                                         training=True,
                                         causality=False)
            enc_tt = ff(enc_tt, num_units=[4 * self.config.att_dim, self.config.att_dim])

        with tf.variable_scope('all_weights', reuse=tf.AUTO_REUSE):
            Wr_wq = tf.get_variable('Wr_wq', [304, 1])
            Wm_wq = tf.get_variable('Wm_wq', [304, 304])
            Wu_wq = tf.get_variable('Wu_wq', [304, 304])

            Wr_wqv = tf.get_variable('Wr_wqv', [300, 1])
            Wm_wqv = tf.get_variable('Wm_wqv', [300, 300])
            Wu_wqv = tf.get_variable('Wu_wqv', [300, 300])
            Wr_wqa = tf.get_variable('Wr_wqa', [300, 1])
            Wm_wqa = tf.get_variable('Wm_wqa', [300, 300])
            Wu_wqa = tf.get_variable('Wu_wqa', [300, 300])
            Wr_wqt = tf.get_variable('Wr_wqt', [300, 1])
            Wm_wqt= tf.get_variable('Wm_wqt', [300, 300])
            Wu_wqt = tf.get_variable('Wu_wqt', [300, 300])

            Wr_wa = tf.get_variable('Wr_wa', [self.att_dim, 1])
            Wm_wa = tf.get_variable('Wm_wa', [self.att_dim, self.att_dim])
            Wu_wa = tf.get_variable('Wu_wa', [self.att_dim, self.att_dim])

            Wr_va = tf.get_variable('Wr_va', [self.att_dim, 1])
            Wm_va = tf.get_variable('Wm_va', [self.att_dim, self.att_dim])
            Wu_va = tf.get_variable('Wu_va', [self.att_dim, self.att_dim])

            wei_va = tf.get_variable('wei_va', [self.att_dim, 150])
            wei_vt = tf.get_variable('wei_vt', [self.att_dim, 150])
            wei_ta = tf.get_variable('wei_ta', [self.att_dim, 150])

            dis_va = tf.get_variable('wei_va', [self.att_dim, 150])
            dis_vt = tf.get_variable('wei_va', [self.att_dim, 150])
            dis_ta = tf.get_variable('wei_va', [self.att_dim, 150])

            W_l = tf.get_variable('W_l', [self.att_dim, self.config.class_num])
            b_l = tf.get_variable('b_l', [1, self.config.class_num])

            wei_v = tf.get_variable('wei_v', [self.config.batch_size, self.config.class_num])
            wei_a = tf.get_variable('wei_a', [self.config.batch_size, self.config.class_num])
            wei_t = tf.get_variable('wei_t', [self.config.batch_size, self.config.class_num])

        common_v = enc_vv
        common_a = enc_aa
        common_t = enc_tt

        v = []
        a = []
        t = []
        v2 = []
        a2 = []
        t2 = []
        for i in range(self.config.batch_size):
            tmp_v = common_v[i]
            tmp_a = common_a[i]
            tmp_t = common_t[i]
            v.append(tmp_v)
            a.append(tmp_a)
            t.append(tmp_t)

        infuluence_v = []
        infuluence_a = []
        infuluence_t = []
        infuluence_v2 = []
        infuluence_a2 = []
        infuluence_t2 = []
        with tf.variable_scope('influence', reuse=tf.AUTO_REUSE):
            Wei_v = tf.get_variable('Wei_v',[self.att_dim, self.config.max_visual_len])  
            Wei_a = tf.get_variable('Wei_a',[self.att_dim, self.config.max_audio_len])  
            Wei_t = tf.get_variable('Wei_t',[self.att_dim, self.config.max_text_len])  

            Wei_vback = tf.get_variable('Wei_vback', [self.config.max_visual_len,self.att_dim])  
            Wei_aback = tf.get_variable('Wei_aback', [self.config.max_audio_len,self.att_dim]) 
            Wei_tback = tf.get_variable('Wei_tback', [self.config.max_text_len,self.att_dim, ])  

        for i in range(self.config.batch_size): 
            v[i] = tf.matmul(Wei_v, v[i])
            a[i] = tf.matmul(Wei_a, a[i])
            t[i] = tf.matmul(Wei_t, t[i])

        for i in range(self.config.batch_size):
            tempv = tf.reshape(v[i], [1, self.att_dim, self.att_dim])
            tempa = tf.reshape(a[i], [1, self.att_dim, self.att_dim])
            tempt = tf.reshape(t[i], [1, self.att_dim, self.att_dim])

            infuluence_v.append(tempv)
            infuluence_a.append(tempa)
            infuluence_t.append(tempt)

        tempv = tf.convert_to_tensor(infuluence_v)
        tempa = tf.convert_to_tensor(infuluence_a)
        tempt = tf.convert_to_tensor(infuluence_t)

        tempv = tf.squeeze(tempv,1)
        tempa = tf.squeeze(tempa,1)
        tempt = tf.squeeze(tempt,1)

        with tf.variable_scope('enc_TinfuluenceV', reuse=tf.AUTO_REUSE):
            enc_TinfuluenceV = multihead_attention(queries=tempv,  # 32 * 300 * 300
                                                       keys=tempt,
                                                       values=tempt,
                                                       num_heads=4,
                                                       dropout_rate=0.2,
                                                       training=True,
                                                       causality=False)
            enc_TinfuluenceV = ff(enc_TinfuluenceV, num_units=[4 * self.att_dim, self.att_dim])  
        with tf.variable_scope('enc_TinfuluenceA', reuse=tf.AUTO_REUSE):
            enc_TinfuluenceA = multihead_attention(queries=tempa,  
                                                       keys=tempt,
                                                       values=tempt,
                                                       num_heads=4,
                                                       dropout_rate=0.2,
                                                       training=True,
                                                       causality=False)
            enc_TinfuluenceA = ff(enc_TinfuluenceA, num_units=[4 * self.att_dim, self.att_dim])  

        for i in range(self.config.batch_size):
            tempv2 = enc_TinfuluenceV[i]
            v2.append(tempv2)
            tempa2 = enc_TinfuluenceA[i]
            a2.append(tempa2)
        for i in range(self.config.batch_size):  
            v2[i] = tf.matmul(Wei_vback, v2[i])
            a2[i] = tf.matmul(Wei_aback, a2[i])
        for i in range(self.config.batch_size):
            tempv_2 = tf.reshape(v2[i], [1, 100, self.att_dim])
            tempa_2 = tf.reshape(a2[i], [1, 150, self.att_dim])
            infuluence_v2.append(tempv_2)
            infuluence_a2.append(tempa_2)

        enc_vv_new = tf.convert_to_tensor(infuluence_v2)
        enc_aa_new = tf.convert_to_tensor(infuluence_a2)
        enc_vv_new = tf.squeeze(enc_vv_new,1)
        enc_aa_new = tf.squeeze(enc_aa_new,1)
        enc_tt_new = enc_tt

        enc_en = enc_all
        enc_en = tf.multiply(enc_en, 1, name='encode_outputs')

        # encode kl loss
        kl = tf.keras.losses.KLDivergence(reduction=losses_utils.ReductionV2.NONE, name='kl')
        de_loss1_pre = kl(tf.nn.softmax(self.pretrained_output, -1), tf.nn.softmax(enc_en, -1))
        de_loss2_pre = kl(tf.nn.softmax(enc_en, -1), tf.nn.softmax(self.pretrained_output, -1))
        self.de_loss_pre = tf.reduce_sum(tf.reduce_mean(de_loss1_pre, -1), -1) + tf.reduce_sum(tf.reduce_mean(de_loss2_pre, -1), -1)

        with tf.variable_scope('de_vv', reuse=tf.AUTO_REUSE):
            enc_tt_de = multihead_attention(queries=enc_tt_new,
                                            keys=enc_tt_new,
                                            values=enc_tt_new,
                                            num_heads=4,
                                            dropout_rate=0.2,
                                            training=True,
                                            causality=False)
            enc_tt_de = ff(enc_tt_de, num_units=[4 * 300, 300])

        de_loss1_t = kl(tf.nn.softmax(enc_tt_de, -1), tf.nn.softmax(enc_tt, -1))
        de_loss2_t = kl(tf.nn.softmax(enc_tt, -1), tf.nn.softmax(enc_tt_de, -1))
        self.de_loss_t = tf.reduce_sum(tf.reduce_mean(de_loss1_t, -1), -1) + tf.reduce_sum(tf.reduce_mean(de_loss2_t, -1), -1)

        outputs_en_vv = SigmoidAtt(enc_vv_new, Wr_wqv, Wm_wqv, Wu_wqv)
        outputs_en_aa = SigmoidAtt(enc_aa_new, Wr_wqa, Wm_wqa, Wu_wqa)
        outputs_en_tt = SigmoidAtt(enc_tt, Wr_wqt, Wm_wqt, Wu_wqt)

        ouput_label = tf.one_hot(self.label, self.config.class_num)

        temp_new_vv = outputs_en_vv
        temp_new_vv = tf.layers.dense(temp_new_vv, self.config.att_dim, use_bias=False)
        output_res_vv = tf.add(tf.matmul(temp_new_vv, W_l), b_l)
        temp_new_aa = outputs_en_aa
        temp_new_aa = tf.layers.dense(temp_new_aa, self.config.att_dim, use_bias=False)
        output_res_aa = tf.add(tf.matmul(temp_new_aa, W_l), b_l)

        temp_new_tt = outputs_en_tt
        temp_new_tt = tf.layers.dense(temp_new_tt, self.config.att_dim, use_bias=False)
        output_res_tt = tf.add(tf.matmul(temp_new_tt, W_l), b_l)

        with tf.variable_scope('vatWei', reuse=tf.AUTO_REUSE):
            Wv = tf.get_variable('Wv', shape=[1])
            Wa = tf.get_variable('Wa', shape=[1])
            Wt = tf.get_variable('Wt', shape=[1])
        output_res_vv = tf.multiply(output_res_vv, Wv)
        output_res_aa = tf.multiply(output_res_aa, Wa)
        output_res_tt = tf.multiply(output_res_tt, Wt)
        output_res = tf.add(output_res_vv, tf.add(output_res_aa, output_res_tt))

        self.prob = tf.nn.softmax(output_res)

        with tf.name_scope('loss'):
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=output_res, labels=ouput_label))
            self.loss = loss
            self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001), weights_list=[W_l, b_l])
            # self.total_loss = self.loss + self.l2_loss + self.de_loss_t
            self.total_loss = self.loss + self.l2_loss + self.de_loss_t + self.de_loss_pre
