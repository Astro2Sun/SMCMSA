import pickle
import tensorflow as tf
import numpy as np
from Settings import Config
from Dataset import Dataset
import random

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

bestvmodel = './saved-modelV/MT_ATT_model-140'
bestamodel = './saved-modelA/MT_ATT_model-860'
besttmodel = './saved-modelT/MT_ATT_model-220'

mtraindata_fortrain = pickle.load(open('F:\data\mosi\m25train.pkl', 'rb'))
mtestdata = pickle.load(open('F:\data\mosi\m25test.pkl', 'rb'))

miss_type = 'm25'

traindata_name = './' + miss_type + 'VATcomplete_train.pkl'
testdata_name = './' + miss_type + 'VATcomplete_test.pkl'

def seperate_data():
    data = Dataset()
    traindata_Vcomplete, testdata_Vcomplete, \
    traindata_Acomplete, testdata_Acomplete, \
    traindata_Tcomplete, testdata_Tcomplete, \
    traindata_VATcomplete, testdata_VATcomplete = data.setdata_for_complete(mtraindata_fortrain, mtestdata)
    Vtrain_file = './' + miss_type + 'Vcomplete_train.pkl'
    Vtest_file = './' + miss_type + 'Vcomplete_test.pkl'
    Atrain_file = './' + miss_type + 'Acomplete_train.pkl'
    Atest_file = './' + miss_type + 'Acomplete_test.pkl'
    Ttrain_file = './' + miss_type + 'Tcomplete_train.pkl'
    Ttest_file = './' + miss_type + 'Tcomplete_test.pkl'
    VATtrain_file = './' + miss_type + 'VATcomplete_train.pkl'
    VATtest_file = './' + miss_type + 'VATcomplete_test.pkl'
    pickle.dump(traindata_Vcomplete, open(Vtrain_file, 'wb'))
    pickle.dump(testdata_Vcomplete, open(Vtest_file, 'wb'))
    pickle.dump(traindata_Acomplete, open(Atrain_file, 'wb'))
    pickle.dump(testdata_Acomplete, open(Atest_file, 'wb'))
    pickle.dump(traindata_Tcomplete, open(Ttrain_file, 'wb'))
    pickle.dump(testdata_Tcomplete, open(Ttest_file, 'wb'))
    pickle.dump(traindata_VATcomplete, open(VATtrain_file, 'wb'))
    pickle.dump(testdata_VATcomplete, open(VATtest_file, 'wb'))

def downdim_forAtrain_fromVAT(sess,setting, modelname = './saved-modelA/MT_ATT_model-590'):
    traindata = pickle.load(open(traindata_name, 'rb'))
    testdata  = pickle.load(open(testdata_name, 'rb'))
    from network_forA import MM
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=False)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, modelname)
        except Exception:
            print('Error')
        result = np.zeros((1, 300))
        total_pred = []
        for i in range(int(len(traindata['L']) / setting.batch_size)): 
            cur_batch = dataset.nextBatch_forA(traindata, testdata, is_training = True)
            feed_dict = {}
            feed_dict[mtest.audio] = cur_batch['A']
            feed_dict[mtest.label] = cur_batch['L']
            temp_new = sess.run([mtest.temp_new_todowndim], feed_dict)
            prob = sess.run([mtest.prob], feed_dict)
            result = np.append(result, temp_new[0], axis=0)
            for j in range(len(prob[0])):  
                total_pred.append(np.argmax(prob[0][j], -1))
        result = result[1:, :]
        print(len(result))
        lowdim_topkltrain = dataset.setdata_for_Adowndim(result, traindata,total_pred)
        AtrainFilename = './' + miss_type + 'AComplete_lowdim_train.pkl'
        pickle.dump(lowdim_topkltrain, open(AtrainFilename, 'wb'))

def downdim_forAtest_fromVAT(sess, setting, modelname = './saved-modelA/MT_ATT_model-590'):
    traindata = pickle.load(open(traindata_name, 'rb'))
    testdata = pickle.load(open(testdata_name, 'rb'))
    from network_forA import MM
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=False)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, modelname)
        except Exception:
            print('Error')
        result = np.zeros((1, 300))
        total_pred = []
        for i in range(int(len(testdata['L']) / setting.batch_size)): 
            cur_batch = dataset.nextBatch_forA(traindata, testdata, is_training= False)
            feed_dict = {}
            feed_dict[mtest.audio] = cur_batch['A']
            feed_dict[mtest.label] = cur_batch['L']
            temp_new = sess.run([mtest.temp_new_todowndim], feed_dict)
            prob = sess.run([mtest.prob], feed_dict)
            result = np.append(result, temp_new[0], axis=0)
            for j in range(len(prob[0])):  
                total_pred.append(np.argmax(prob[0][j], -1))
        result = result[1:, :]
        print(len(result))
        lowdim_topkltest = dataset.setdata_for_Adowndim(result, testdata,total_pred)
        AtestFilename = './' + miss_type + 'AComplete_lowdim_test.pkl'
        pickle.dump(lowdim_topkltest, open(AtestFilename, 'wb'))

def downdim_forVtrain_fromVAT(sess,setting, modelname = './saved-modelV/MT_ATT_model-590'):
    traindata = pickle.load(open(traindata_name, 'rb'))
    testdata = pickle.load(open(testdata_name, 'rb'))
    from network_forV import MM
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=False)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, modelname)
        except Exception:
            print('Error')
        result = np.zeros((1, 300))
        total_pred = []
        for i in range(int(len(traindata['L']) / setting.batch_size)):  
            cur_batch = dataset.nextBatch_forV(traindata, testdata,is_training = True)
            feed_dict = {}
            feed_dict[mtest.visual] = cur_batch['V']
            feed_dict[mtest.label] = cur_batch['L']
            temp_new = sess.run([mtest.temp_new_todowndim], feed_dict)
            prob = sess.run([mtest.prob], feed_dict)
            result = np.append(result, temp_new[0], axis=0)
            for j in range(len(prob[0])):  
                total_pred.append(np.argmax(prob[0][j], -1))
        result = result[1:, :]
        print(len(result))
        lowdim_topkltrain = dataset.setdata_for_Vdowndim(result, traindata,total_pred)
        VtrainFilename = './' + miss_type + 'VComplete_lowdim_train.pkl'
        pickle.dump(lowdim_topkltrain, open(VtrainFilename, 'wb'))

def downdim_forVtest_fromVAT(sess, setting, modelname = './saved-modelV/MT_ATT_model-590'):
    traindata = pickle.load(open(traindata_name, 'rb'))
    testdata = pickle.load(open(testdata_name, 'rb'))
    from network_forV import MM
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=False)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, modelname)
        except Exception:
            print('Error')
        result = np.zeros((1, 300))
        total_pred = []
        for i in range(int(len(testdata['L']) / setting.batch_size)): 
            cur_batch = dataset.nextBatch_forV(traindata, testdata, is_training= False)
            feed_dict = {}
            feed_dict[mtest.visual] = cur_batch['V']
            feed_dict[mtest.label] = cur_batch['L']
            temp_new = sess.run([mtest.temp_new_todowndim], feed_dict)
            prob = sess.run([mtest.prob], feed_dict)
            result = np.append(result, temp_new[0], axis=0)
            for j in range(len(prob[0])):
                total_pred.append(np.argmax(prob[0][j], -1))
        result = result[1:, :]
        print(len(result))
        lowdim_topkltest = dataset.setdata_for_Vdowndim(result, testdata,total_pred)
        VtestFilename = './' + miss_type + 'VComplete_lowdim_test.pkl'
        pickle.dump(lowdim_topkltest, open(VtestFilename, 'wb'))

def downdim_forTtrain_fromVAT(sess,setting, modelname = './saved-modelT/MT_ATT_model-590'):
    traindata = pickle.load(open(traindata_name, 'rb'))
    testdata = pickle.load(open(testdata_name, 'rb'))
    from network_forT import MM
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=False)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, modelname)
        except Exception:
            print('Error')
        result = np.zeros((1, 300))
        total_pred = []
        for i in range(int(len(traindata['L']) / setting.batch_size)): 
            cur_batch = dataset.nextBatch_forT(traindata, testdata, is_training = True)
            feed_dict = {}
            feed_dict[mtest.text] = cur_batch['T']
            feed_dict[mtest.label] = cur_batch['L']
            temp_new = sess.run([mtest.temp_new_todowndim], feed_dict)
            prob = sess.run([mtest.prob], feed_dict)
            result = np.append(result, temp_new[0], axis=0)
            for j in range(len(prob[0])):  
                total_pred.append(np.argmax(prob[0][j], -1))
        result = result[1:, :]
        print(len(result))
        lowdim_topkltrain = dataset.setdata_for_Tdowndim(result, traindata,total_pred)
        TtrainFilename = './' + miss_type + 'TComplete_lowdim_train.pkl'
        pickle.dump(lowdim_topkltrain, open(TtrainFilename, 'wb'))

def downdim_forTtest_fromVAT(sess, setting, modelname='./saved-modelT/MT_ATT_model-590'):
    traindata = pickle.load(open(traindata_name, 'rb'))
    testdata = pickle.load(open(testdata_name, 'rb'))
    from network_forT import MM
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=False)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, modelname)
        except Exception:
            print('Error')
        result = np.zeros((1, 300))
        total_pred = []
        for i in range(int(len(testdata['L']) / setting.batch_size)):  
            cur_batch = dataset.nextBatch_forT(traindata, testdata, is_training= False)
            feed_dict = {}
            feed_dict[mtest.text] = cur_batch['T']
            feed_dict[mtest.label] = cur_batch['L']
            temp_new = sess.run([mtest.temp_new_todowndim], feed_dict)
            prob = sess.run([mtest.prob], feed_dict)
            result = np.append(result, temp_new[0], axis=0)
            for j in range(len(prob[0])): 
                total_pred.append(np.argmax(prob[0][j], -1))
        result = result[1:, :]
        print(len(result))
        lowdim_topkltest = dataset.setdata_for_Tdowndim(result, testdata,total_pred)
        TtestFilename = './' + miss_type + 'TComplete_lowdim_test.pkl'
        pickle.dump(lowdim_topkltest, open(TtestFilename, 'wb'))

uncom_traindata_namea = './' + miss_type + 'Acomplete_train.pkl'
uncom_testdata_namea = './' + miss_type + 'Acomplete_test.pkl'
uncom_traindata_namev = './' + miss_type + 'Vcomplete_train.pkl'
uncom_testdata_namev = './' + miss_type + 'Vcomplete_test.pkl'
uncom_traindata_namet = './' + miss_type + 'Tcomplete_train.pkl'
uncom_testdata_namet = './' + miss_type + 'Tcomplete_test.pkl'

def downdim_forAtrain_fromuncomplete(sess,setting, modelname = './saved-modelA/MT_ATT_model-590'):
    traindata = pickle.load(open(uncom_traindata_namea, 'rb'))
    testdata  = pickle.load(open(uncom_testdata_namea, 'rb'))
    from network_forA import MM
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=False)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, modelname)
        except Exception:
            print('Error')
        result = np.zeros((1, 300))
        total_pred = []
        for i in range(int(len(traindata['L']) / setting.batch_size)):  
            cur_batch = dataset.nextBatch_forA(traindata, testdata, is_training = True)
            feed_dict = {}
            feed_dict[mtest.audio] = cur_batch['A']
            feed_dict[mtest.label] = cur_batch['L']
            temp_new = sess.run([mtest.temp_new_todowndim], feed_dict)
            prob = sess.run([mtest.prob], feed_dict)
            result = np.append(result, temp_new[0], axis=0)
            for j in range(len(prob[0])):  
                total_pred.append(np.argmax(prob[0][j], -1))
        result = result[1:, :]
        print(len(result))
        lowdim_topkltrain = dataset.setdata_for_Adowndim(result, traindata,total_pred)
        AtrainFilename = './' + miss_type + 'Alowdim_train.pkl'
        pickle.dump(lowdim_topkltrain, open(AtrainFilename, 'wb'))

def downdim_forAtest_fromuncomplete(sess, setting, modelname = './saved-modelA/MT_ATT_model-590'):
    traindata = pickle.load(open(uncom_traindata_namea, 'rb'))
    testdata = pickle.load(open(uncom_testdata_namea, 'rb'))
    from network_forA import MM
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=False)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, modelname)
        except Exception:
            print('Error')
        result = np.zeros((1, 300))
        total_pred = []
        for i in range(int(len(testdata['L']) / setting.batch_size)): 
            cur_batch = dataset.nextBatch_forA(traindata, testdata, is_training= False)
            feed_dict = {}
            feed_dict[mtest.audio] = cur_batch['A']
            feed_dict[mtest.label] = cur_batch['L']
            temp_new = sess.run([mtest.temp_new_todowndim], feed_dict)
            prob = sess.run([mtest.prob], feed_dict)
            result = np.append(result, temp_new[0], axis=0)
            for j in range(len(prob[0])):  
                total_pred.append(np.argmax(prob[0][j], -1))
        result = result[1:, :]
        print(len(result))
        lowdim_topkltest = dataset.setdata_for_Adowndim(result, testdata,total_pred)
        AtestFilename = './' + miss_type + 'Alowdim_test.pkl'
        pickle.dump(lowdim_topkltest, open(AtestFilename, 'wb'))

def downdim_forVtrain_fromuncomplete(sess,setting, modelname = './saved-modelV/MT_ATT_model-590'):
    traindata = pickle.load(open(uncom_traindata_namev, 'rb'))
    testdata = pickle.load(open(uncom_testdata_namev, 'rb'))
    from network_forV import MM
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=False)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, modelname)
        except Exception:
            print('Error')
        result = np.zeros((1, 300))
        total_pred = []
        for i in range(int(len(traindata['L']) / setting.batch_size)): 
            cur_batch = dataset.nextBatch_forV(traindata, testdata,is_training = True)
            feed_dict = {}
            feed_dict[mtest.visual] = cur_batch['V']
            feed_dict[mtest.label] = cur_batch['L']
            temp_new = sess.run([mtest.temp_new_todowndim], feed_dict)
            prob = sess.run([mtest.prob], feed_dict)
            result = np.append(result, temp_new[0], axis=0)
            for j in range(len(prob[0])):  
                total_pred.append(np.argmax(prob[0][j], -1))
        result = result[1:, :]
        print(len(result))
        lowdim_topkltrain = dataset.setdata_for_Vdowndim(result, traindata,total_pred)
        VtrainFilename = './' + miss_type + 'Vlowdim_train.pkl'
        pickle.dump(lowdim_topkltrain, open(VtrainFilename, 'wb'))

def downdim_forVtest_fromuncomplete(sess, setting, modelname = './saved-modelV/MT_ATT_model-590'):
    traindata = pickle.load(open(uncom_traindata_namev, 'rb'))
    testdata = pickle.load(open(uncom_testdata_namev, 'rb'))
    from network_forV import MM
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=False)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, modelname)
        except Exception:
            print('Error')
        result = np.zeros((1, 300))
        total_pred = []
        for i in range(int(len(testdata['L']) / setting.batch_size)):  
            cur_batch = dataset.nextBatch_forV(traindata, testdata, is_training= False)
            feed_dict = {}
            feed_dict[mtest.visual] = cur_batch['V']
            feed_dict[mtest.label] = cur_batch['L']
            temp_new = sess.run([mtest.temp_new_todowndim], feed_dict)
            prob = sess.run([mtest.prob], feed_dict)
            result = np.append(result, temp_new[0], axis=0)
            for j in range(len(prob[0])):  
                total_pred.append(np.argmax(prob[0][j], -1))
        result = result[1:, :]
        print(len(result))
        lowdim_topkltest = dataset.setdata_for_Vdowndim(result, testdata,total_pred)
        VtestFilename = './' + miss_type + 'Vlowdim_test.pkl'
        pickle.dump(lowdim_topkltest, open(VtestFilename, 'wb'))

def downdim_forTtrain_fromuncomplete(sess,setting, modelname = './saved-modelT/MT_ATT_model-590'):
    traindata = pickle.load(open(uncom_traindata_namet, 'rb'))
    testdata = pickle.load(open(uncom_testdata_namet, 'rb'))
    from network_forT import MM
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=False)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, modelname)
        except Exception:
            print('Error')
        result = np.zeros((1, 300))
        total_pred = []
        for i in range(int(len(traindata['L']) / setting.batch_size)): 
            cur_batch = dataset.nextBatch_forT(traindata, testdata, is_training = True)
            feed_dict = {}
            feed_dict[mtest.text] = cur_batch['T']
            feed_dict[mtest.label] = cur_batch['L']
            temp_new = sess.run([mtest.temp_new_todowndim], feed_dict)
            prob = sess.run([mtest.prob], feed_dict)
            result = np.append(result, temp_new[0], axis=0)
            for j in range(len(prob[0])):  
                total_pred.append(np.argmax(prob[0][j], -1))
        result = result[1:, :]
        print(len(result))
        lowdim_topkltrain = dataset.setdata_for_Tdowndim(result, traindata,total_pred)
        TtrainFilename = './' + miss_type + 'Tlowdim_train.pkl'
        pickle.dump(lowdim_topkltrain, open(TtrainFilename, 'wb'))

def downdim_forTtest_fromuncomplete(sess, setting, modelname='./saved-modelT/MT_ATT_model-590'):
    traindata = pickle.load(open(uncom_traindata_namet, 'rb'))
    testdata = pickle.load(open(uncom_testdata_namet, 'rb'))
    from network_forT import MM
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=False)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, modelname)
        except Exception:
            print('Error')
        result = np.zeros((1, 300))
        total_pred = []
        for i in range(int(len(testdata['L']) / setting.batch_size)): 
            cur_batch = dataset.nextBatch_forT(traindata, testdata, is_training= False)
            feed_dict = {}
            feed_dict[mtest.text] = cur_batch['T']
            feed_dict[mtest.label] = cur_batch['L']
            temp_new = sess.run([mtest.temp_new_todowndim], feed_dict)
            prob = sess.run([mtest.prob], feed_dict)
            result = np.append(result, temp_new[0], axis=0)
            for j in range(len(prob[0])):  
                total_pred.append(np.argmax(prob[0][j], -1))
        result = result[1:, :]
        print(len(result))
        lowdim_topkltest = dataset.setdata_for_Tdowndim(result, testdata,total_pred)
        TtestFilename = './' + miss_type + 'Tlowdim_test.pkl'
        pickle.dump(lowdim_topkltest, open(TtestFilename, 'wb'))

def downdim_fortestDataset(sess, setting, modelname='./saved-modelT/MT_ATT_model-590'):
    # mtraindata_fortrain = pickle.load(open('../../data/m11train.pkl', 'rb'))
    # mtestdata = pickle.load(open('../../data/m11test.pkl', 'rb'))
    traindata = mtraindata_fortrain
    testdata = mtestdata
    from network_forT import MM
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=False)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, modelname)
        except Exception:
            print('Error')
        result = np.zeros((1, 300))
        total_pred = []
        for i in range(int(len(testdata['L']) / setting.batch_size)):  
            cur_batch = dataset.nextBatch_forT(traindata, testdata, is_training= False)
            feed_dict = {}
            feed_dict[mtest.text] = cur_batch['T']
            feed_dict[mtest.label] = cur_batch['L']
            temp_new = sess.run([mtest.temp_new_todowndim], feed_dict)
            prob = sess.run([mtest.prob], feed_dict)
            result = np.append(result, temp_new[0], axis=0)
            for j in range(len(prob[0])): 
                total_pred.append(np.argmax(prob[0][j], -1))
        result = result[1:, :]
        print(len(result))
        lowdim_topkltest = dataset.setdata_for_Tdowndim(result, testdata,total_pred)
        TtestFilename = './' + miss_type + 'Tlowdim_test.pkl'
        pickle.dump(lowdim_topkltest, open(TtestFilename, 'wb'))

def fill_m2():
    traindata = {'ID': [], 'V': [], 'A': [], 'T': [], 'L': [], 'F': [], 'PreL': []}
    testdata = {'ID': [], 'V': [], 'A': [], 'T': [], 'L': [], 'F': [], 'PreL': []}

    # traindata complete
    uncomplete_data = mtraindata_fortrain

    Vlowdim = pickle.load(open(miss_type + 'Vlowdim_train.pkl', 'rb'))
    Alowdim = pickle.load(open(miss_type + 'Alowdim_train.pkl', 'rb'))
    Tlowdim = pickle.load(open(miss_type + 'Tlowdim_train.pkl', 'rb'))

    vtraindata = pickle.load(open(miss_type + 'VComplete_lowdim_train.pkl', 'rb'))
    vdata = np.array(vtraindata['V'])
    vlabels = np.array(vtraindata['L'])
    vids = np.array(vtraindata['ID'])

    atraindata = pickle.load(open(miss_type + 'AComplete_lowdim_train.pkl', 'rb'))
    adata = np.array(atraindata['A'])
    alabels = np.array(atraindata['L'])
    aids = np.array(atraindata['ID'])

    ttraindata = pickle.load(open(miss_type + 'TComplete_lowdim_train.pkl', 'rb'))
    tdata = np.array(ttraindata['T'])
    tlabels = np.array(ttraindata['L'])
    tids = np.array(ttraindata['ID'])

    for i in range(len(uncomplete_data['ID'])):
        if uncomplete_data['F'][i] == 3:
            vlowdmi_index = uncomplete_data['ID'][i]
            if vlowdmi_index in Vlowdim['ID']:
                vlowdmi_index = Vlowdim['ID'].index(vlowdmi_index)
                input_data = np.array(Vlowdim['V'][vlowdmi_index]).reshape(1, 300)

                similarity = np.dot(vdata, input_data.T) / (np.linalg.norm(vdata, axis=1) * np.linalg.norm(input_data))
                num_top_similar_points = 3
                top_similar_points_indices = np.argsort(similarity[:, 0])[::-1][:num_top_similar_points]
                for j in top_similar_points_indices:
                    id = vids[j]
                    index = uncomplete_data['ID'].index(id)
                    preL_index = vtraindata['ID'].index(id)
                    if Vlowdim['PreL'][vlowdmi_index] == vtraindata['PreL'][preL_index]:
                        print('ID: {}, Label: {}'.format(vids[j], vlabels[j]))
                        traindata['ID'].append(uncomplete_data['ID'][i])
                        traindata['V'].append(uncomplete_data['V'][i])
                        traindata['A'].append(uncomplete_data['A'][index])
                        traindata['T'].append(uncomplete_data['T'][index])
                        traindata['L'].append(uncomplete_data['L'][i])
                        traindata['F'].append(uncomplete_data['F'][i])
                        break
                    elif j == top_similar_points_indices[2]:
                        traindata['ID'].append(uncomplete_data['ID'][i])
                        traindata['V'].append(uncomplete_data['V'][i])
                        traindata['A'].append(uncomplete_data['A'][i])
                        traindata['T'].append(uncomplete_data['T'][i])
                        traindata['L'].append(uncomplete_data['L'][i])
                        traindata['F'].append(uncomplete_data['F'][i])
            else:
                traindata['ID'].append(uncomplete_data['ID'][i])
                traindata['V'].append(uncomplete_data['V'][i])
                traindata['A'].append(uncomplete_data['A'][i])
                traindata['T'].append(uncomplete_data['T'][i])
                traindata['L'].append(uncomplete_data['L'][i])
                traindata['F'].append(uncomplete_data['F'][i])

        elif uncomplete_data['F'][i] == 4:
            alowdmi_index = uncomplete_data['ID'][i]
            if alowdmi_index in Alowdim['ID']:
                alowdmi_index = Alowdim['ID'].index(alowdmi_index)
                input_data = np.array(Alowdim['A'][alowdmi_index]).reshape(1, 300)

                similarity = np.dot(adata, input_data.T) / (np.linalg.norm(adata, axis=1) * np.linalg.norm(input_data))
                num_top_similar_points = 3
                top_similar_points_indices = np.argsort(similarity[:, 0])[::-1][:num_top_similar_points]
                for j in top_similar_points_indices:
                    id = aids[j]
                    index = uncomplete_data['ID'].index(id)
                    preL_index = atraindata['ID'].index(id)
                    if Alowdim['PreL'][alowdmi_index] == atraindata['PreL'][preL_index]:
                        print('ID: {}, Label: {}'.format(aids[j], alabels[j]))
                        traindata['ID'].append(uncomplete_data['ID'][i])
                        traindata['V'].append(uncomplete_data['V'][index])
                        traindata['A'].append(uncomplete_data['A'][i])
                        traindata['T'].append(uncomplete_data['T'][index])
                        traindata['L'].append(uncomplete_data['L'][i])
                        traindata['F'].append(uncomplete_data['F'][i])
                        break
                    elif j == top_similar_points_indices[2]:
                        traindata['ID'].append(uncomplete_data['ID'][i])
                        traindata['V'].append(uncomplete_data['V'][i])
                        traindata['A'].append(uncomplete_data['A'][i])
                        traindata['T'].append(uncomplete_data['T'][i])
                        traindata['L'].append(uncomplete_data['L'][i])
                        traindata['F'].append(uncomplete_data['F'][i])
            else:
                traindata['ID'].append(uncomplete_data['ID'][i])
                traindata['V'].append(uncomplete_data['V'][i])
                traindata['A'].append(uncomplete_data['A'][i])
                traindata['T'].append(uncomplete_data['T'][i])
                traindata['L'].append(uncomplete_data['L'][i])
                traindata['F'].append(uncomplete_data['F'][i])

        elif uncomplete_data['F'][i] == 5:  
            tlowdmi_index = uncomplete_data['ID'][i]
            if tlowdmi_index in Tlowdim['ID']:
                tlowdmi_index = Tlowdim['ID'].index(tlowdmi_index)
                input_data = np.array(Tlowdim['T'][tlowdmi_index]).reshape(1, 300)
                similarity = np.dot(tdata, input_data.T) / (np.linalg.norm(tdata, axis=1) * np.linalg.norm(input_data))
                num_top_similar_points = 3
                top_similar_points_indices = np.argsort(similarity[:, 0])[::-1][:num_top_similar_points]
                for j in top_similar_points_indices:
                    id = tids[j]
                    index = uncomplete_data['ID'].index(id)
                    if Tlowdim['PreL'][tlowdmi_index] == uncomplete_data['L'][index]:
                        print('ID: {}, Label: {}'.format(tids[j], tlabels[j]))
                        traindata['ID'].append(uncomplete_data['ID'][i])
                        traindata['V'].append(uncomplete_data['V'][index])
                        traindata['A'].append(uncomplete_data['A'][index])
                        traindata['T'].append(uncomplete_data['T'][i])
                        traindata['L'].append(uncomplete_data['L'][i])
                        traindata['F'].append(uncomplete_data['F'][i])
                        break
                    elif j == top_similar_points_indices[2]:
                        traindata['ID'].append(uncomplete_data['ID'][i])
                        traindata['V'].append(uncomplete_data['V'][i])
                        traindata['A'].append(uncomplete_data['A'][i])
                        traindata['T'].append(uncomplete_data['T'][i])
                        traindata['L'].append(uncomplete_data['L'][i])
                        traindata['F'].append(uncomplete_data['F'][i])
            else:
                traindata['ID'].append(uncomplete_data['ID'][i])
                traindata['V'].append(uncomplete_data['V'][i])
                traindata['A'].append(uncomplete_data['A'][i])
                traindata['T'].append(uncomplete_data['T'][i])
                traindata['L'].append(uncomplete_data['L'][i])
                traindata['F'].append(uncomplete_data['F'][i])
        else:
            traindata['ID'].append(uncomplete_data['ID'][i])
            traindata['V'].append(uncomplete_data['V'][i])
            traindata['A'].append(uncomplete_data['A'][i])
            traindata['T'].append(uncomplete_data['T'][i])
            traindata['L'].append(uncomplete_data['L'][i])
            traindata['F'].append(uncomplete_data['F'][i])

    pickle.dump(traindata, open('./'+ miss_type + 'train_complete.pkl', 'wb'))

    # testdata complete
    uncomplete_data_test = mtestdata
    mtraindata = pickle.load(open('./'+ miss_type + 'VATcomplete_train.pkl', 'rb'))
    complete_data_train = mtraindata

    vtraindata = pickle.load(open(miss_type + 'VComplete_lowdim_train.pkl', 'rb'))
    vdata_train = np.array(vtraindata['V'])
    vlabels_train = np.array(vtraindata['L'])
    vids_train = np.array(vtraindata['ID'])

    atraindata = pickle.load(open(miss_type + 'AComplete_lowdim_train.pkl', 'rb'))
    adata_train = np.array(atraindata['A'])
    alabels_train = np.array(atraindata['L'])
    aids_train = np.array(atraindata['ID'])

    ttraindata = pickle.load(open(miss_type + 'TComplete_lowdim_train.pkl', 'rb'))
    tdata_train = np.array(ttraindata['T'])
    tlabels_train = np.array(ttraindata['L'])
    tids_train = np.array(ttraindata['ID'])

    Vlowdim = pickle.load(open(miss_type + 'Vlowdim_test.pkl', 'rb'))
    Alowdim = pickle.load(open(miss_type + 'Alowdim_test.pkl', 'rb'))
    Tlowdim = pickle.load(open(miss_type + 'Tlowdim_test.pkl', 'rb'))

    for i in range(len(uncomplete_data_test['ID'])):
        if uncomplete_data_test['F'][i] == 3:  
            vlowdmi_index = uncomplete_data_test['ID'][i]
            if vlowdmi_index in Vlowdim['ID']:
                vlowdmi_index = Vlowdim['ID'].index(vlowdmi_index)
                input_data = np.array(Vlowdim['V'][vlowdmi_index]).reshape(1, 300)
                similarity = np.dot(vdata_train, input_data.T) / (np.linalg.norm(vdata_train, axis=1) * np.linalg.norm(input_data))
                num_top_similar_points = 3
                top_similar_points_indices = np.argsort(similarity[:, 0])[::-1][:num_top_similar_points]
                for j in top_similar_points_indices:
                    id = vids_train[j]
                    index = complete_data_train['ID'].index(id)
                    preL_index = vtraindata['ID'].index(id)
                    if Vlowdim['PreL'][vlowdmi_index] == vtraindata['PreL'][preL_index]:
                        print('ID: {}, Label: {}'.format(vids_train[j], vlabels_train[j]))
                        testdata['ID'].append(uncomplete_data_test['ID'][i])
                        testdata['V'].append(uncomplete_data_test['V'][i])
                        testdata['A'].append(complete_data_train['A'][index])
                        testdata['T'].append(complete_data_train['T'][index])
                        testdata['L'].append(uncomplete_data_test['L'][i])
                        testdata['F'].append(uncomplete_data_test['F'][i])
                        break
                    elif j == top_similar_points_indices[2]:      
                        testdata['ID'].append(uncomplete_data_test['ID'][i])
                        testdata['V'].append(uncomplete_data_test['V'][i])
                        testdata['A'].append(uncomplete_data_test['A'][i])
                        testdata['T'].append(uncomplete_data_test['T'][i])
                        testdata['L'].append(uncomplete_data_test['L'][i])
                        testdata['F'].append(uncomplete_data_test['F'][i])
            else:           
                testdata['ID'].append(uncomplete_data_test['ID'][i])
                testdata['V'].append(uncomplete_data_test['V'][i])
                testdata['A'].append(uncomplete_data_test['A'][i])
                testdata['T'].append(uncomplete_data_test['T'][i])
                testdata['L'].append(uncomplete_data_test['L'][i])
                testdata['F'].append(uncomplete_data_test['F'][i])

        elif uncomplete_data_test['F'][i] == 4:  
            alowdmi_index = uncomplete_data_test['ID'][i]
            if alowdmi_index in Alowdim['ID']:
                alowdmi_index = Alowdim['ID'].index(alowdmi_index)
                input_data = np.array(Alowdim['A'][alowdmi_index]).reshape(1, 300)
                similarity = np.dot(adata_train, input_data.T) / (np.linalg.norm(adata_train, axis=1) * np.linalg.norm(input_data))
                num_top_similar_points = 3
                top_similar_points_indices = np.argsort(similarity[:, 0])[::-1][:num_top_similar_points]
                for j in top_similar_points_indices:
                    id = aids_train[j]
                    index = complete_data_train['ID'].index(id)
                    preL_index = atraindata['ID'].index(id)
                    if Alowdim['PreL'][alowdmi_index] == atraindata['PreL'][preL_index]:
                        print('ID: {}, Label: {}'.format(aids_train[j], alabels_train[j]))
                        testdata['ID'].append(uncomplete_data_test['ID'][i])
                        testdata['V'].append(complete_data_train['V'][index])
                        testdata['A'].append(uncomplete_data_test['A'][i])
                        testdata['T'].append(complete_data_train['T'][index])
                        testdata['L'].append(uncomplete_data_test['L'][i])
                        testdata['F'].append(uncomplete_data_test['F'][i])
                        break
                    elif j == top_similar_points_indices[2]:      
                        testdata['ID'].append(uncomplete_data_test['ID'][i])
                        testdata['V'].append(uncomplete_data_test['V'][i])
                        testdata['A'].append(uncomplete_data_test['A'][i])
                        testdata['T'].append(uncomplete_data_test['T'][i])
                        testdata['L'].append(uncomplete_data_test['L'][i])
                        testdata['F'].append(uncomplete_data_test['F'][i])
            else:          
                testdata['ID'].append(uncomplete_data_test['ID'][i])
                testdata['V'].append(uncomplete_data_test['V'][i])
                testdata['A'].append(uncomplete_data_test['A'][i])
                testdata['T'].append(uncomplete_data_test['T'][i])
                testdata['L'].append(uncomplete_data_test['L'][i])
                testdata['F'].append(uncomplete_data_test['F'][i])

        elif uncomplete_data_test['F'][i] == 5:  
            tlowdmi_index = uncomplete_data_test['ID'][i]
            if tlowdmi_index in Tlowdim['ID']:
                tlowdmi_index = Tlowdim['ID'].index(tlowdmi_index)
                input_data = np.array(Tlowdim['T'][tlowdmi_index]).reshape(1, 300)
                similarity = np.dot(tdata_train, input_data.T) / (np.linalg.norm(tdata_train, axis=1) * np.linalg.norm(input_data))
                num_top_similar_points = 3
                top_similar_points_indices = np.argsort(similarity[:, 0])[::-1][:num_top_similar_points]
                for j in top_similar_points_indices:
                    id = tids_train[j]
                    index = complete_data_train['ID'].index(id)
                    if Tlowdim['PreL'][tlowdmi_index] == complete_data_train['L'][index]:
                        print('ID: {}, Label: {}'.format(tids_train[j], tlabels_train[j]))
                        testdata['ID'].append(uncomplete_data_test['ID'][i])
                        testdata['V'].append(complete_data_train['V'][index])
                        testdata['A'].append(complete_data_train['A'][index])
                        testdata['T'].append(uncomplete_data_test['T'][i])
                        testdata['L'].append(uncomplete_data_test['L'][i])
                        testdata['F'].append(uncomplete_data_test['F'][i])
                        break
                    elif j == top_similar_points_indices[2]:      
                        testdata['ID'].append(uncomplete_data_test['ID'][i])
                        testdata['V'].append(uncomplete_data_test['V'][i])
                        testdata['A'].append(uncomplete_data_test['A'][i])
                        testdata['T'].append(uncomplete_data_test['T'][i])
                        testdata['L'].append(uncomplete_data_test['L'][i])
                        testdata['F'].append(uncomplete_data_test['F'][i])
            else:           
                testdata['ID'].append(uncomplete_data_test['ID'][i])
                testdata['V'].append(uncomplete_data_test['V'][i])
                testdata['A'].append(uncomplete_data_test['A'][i])
                testdata['T'].append(uncomplete_data_test['T'][i])
                testdata['L'].append(uncomplete_data_test['L'][i])
                testdata['F'].append(uncomplete_data_test['F'][i])
        else:
            testdata['ID'].append(uncomplete_data_test['ID'][i])
            testdata['V'].append(uncomplete_data_test['V'][i])
            testdata['A'].append(uncomplete_data_test['A'][i])
            testdata['T'].append(uncomplete_data_test['T'][i])
            testdata['L'].append(uncomplete_data_test['L'][i])
            testdata['F'].append(uncomplete_data_test['F'][i])

    pickle.dump(testdata, open('./' + miss_type + 'test_complete.pkl', 'wb'))


if __name__ == '__main__':
    miss = 1
    # seperate_data()     
    if miss == 1:
        # setting = Config()
        # with tf.Graph().as_default():
        #     config = tf.ConfigProto(allow_soft_placement=True)
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config)
        #     downdim_forAtrain_fromVAT(sess, setting, bestamodel)
        #
        # with tf.Graph().as_default():
        #     config = tf.ConfigProto(allow_soft_placement=True)
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config)
        #     downdim_forAtest_fromVAT(sess, setting, bestamodel)
        #
        # with tf.Graph().as_default():
        #     config = tf.ConfigProto(allow_soft_placement=True)
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config)
        #     downdim_forVtrain_fromVAT(sess, setting, bestvmodel)
        #
        # with tf.Graph().as_default():
        #     config = tf.ConfigProto(allow_soft_placement=True)
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config)
        #     downdim_forVtest_fromVAT(sess, setting, bestvmodel)
        #
        # with tf.Graph().as_default():
        #     config = tf.ConfigProto(allow_soft_placement=True)
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config)
        #     downdim_forTtrain_fromVAT(sess, setting, besttmodel)
        #
        # with tf.Graph().as_default():
        #     config = tf.ConfigProto(allow_soft_placement=True)
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config)
        #     downdim_forTtest_fromVAT(sess, setting, besttmodel)
        #
        # with tf.Graph().as_default():
        #     config = tf.ConfigProto(allow_soft_placement=True)
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config)
        #     downdim_forAtrain_fromuncomplete(sess, setting, bestamodel)
        #
        # with tf.Graph().as_default():
        #     config = tf.ConfigProto(allow_soft_placement=True)
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config)
        #     downdim_forAtest_fromuncomplete(sess, setting, bestamodel)
        #
        # with tf.Graph().as_default():
        #     config = tf.ConfigProto(allow_soft_placement=True)
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config)
        #     downdim_forVtrain_fromuncomplete(sess, setting, bestvmodel)
        #
        # with tf.Graph().as_default():
        #     config = tf.ConfigProto(allow_soft_placement=True)
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config)
        #     downdim_forVtest_fromuncomplete(sess, setting, bestvmodel)
        #
        # with tf.Graph().as_default():
        #     config = tf.ConfigProto(allow_soft_placement=True)
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config)
        #     downdim_forTtrain_fromuncomplete(sess, setting, besttmodel)
        #
        # with tf.Graph().as_default():
        #     config = tf.ConfigProto(allow_soft_placement=True)
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config)
        #     downdim_forTtest_fromuncomplete(sess, setting, besttmodel)

            fill_m2()
    else:
        pass
