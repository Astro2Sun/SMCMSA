
# -*- coding: UTF-8 -*-
from Settings import Config
import re
import os
import sys
import numpy as np
import pickle
import random



missing_type = 'm21'
missing_type_single_multi = 5   
missing_rate = 0.1
fenmu = 10 


traindata = pickle.load(open('F:\data\\iemocap\\train7.pkl', 'rb'))
testdata = pickle.load(open('F:\data\\iemocap\\test7.pkl', 'rb'))

missing_num_train = missing_rate * len(traindata['ID'])
missing_num_test = missing_rate * len(testdata['ID'])

miss_visual = list(np.zeros([100, 709]))
miss_audio = list(np.zeros([150, 33]))
miss_text = list(np.zeros([25, 768]))
#set miss value

traindata_missing = {'ID': [], 'V': [], 'A': [], 'T': [], 'L': [], 'F': []}
testdata_missing = {'ID': [], 'V': [], 'A': [], 'T': [], 'L': [], 'F': []}

missing_num_train_true = 0
missing_num_test_true = 0

for i in range(len(traindata['ID'])):
    # print(i)
    rnd_is_missing = random.randint(1, fenmu)  # 0: visual  1:audio 2:text
    if rnd_is_missing == 1 and missing_num_train_true <= missing_num_train:
        # print('missingmodality')
        missing_num_train_true = missing_num_train_true + 1
        rnd = random.randint(0, missing_type_single_multi)  # 0: visual  1:audio 2:text
        if rnd == 0:
            flag = 0
            traindata_missing['ID'].append(traindata['ID'][i])
            traindata_missing['V'].append(miss_visual)
            traindata_missing['A'].append(traindata['A'][i])
            traindata_missing['T'].append(traindata['T'][i])
            traindata_missing['L'].append(traindata['L'][i])
            traindata_missing['F'].append(flag)
        elif rnd == 1:
            flag = 1
            traindata_missing['ID'].append(traindata['ID'][i])
            traindata_missing['V'].append(traindata['V'][i])
            traindata_missing['A'].append(miss_audio)
            traindata_missing['T'].append(traindata['T'][i])
            traindata_missing['L'].append(traindata['L'][i])
            traindata_missing['F'].append(flag)
        elif rnd == 2:
            flag = 2
            traindata_missing['ID'].append(traindata['ID'][i])
            traindata_missing['V'].append(traindata['V'][i])
            traindata_missing['A'].append(traindata['A'][i])
            traindata_missing['T'].append(miss_text)
            traindata_missing['L'].append(traindata['L'][i])
            traindata_missing['F'].append(flag)
        elif rnd == 3:
            flag = 3
            traindata_missing['ID'].append(traindata['ID'][i])
            traindata_missing['V'].append(traindata['V'][i])
            traindata_missing['A'].append(miss_audio)
            traindata_missing['T'].append(miss_text)
            traindata_missing['L'].append(traindata['L'][i])
            # traindata_missing['Emotion'].append(traindata['Emotion'][i])
            traindata_missing['F'].append(flag)
        elif rnd == 4:
            flag = 4
            traindata_missing['ID'].append(traindata['ID'][i])
            traindata_missing['V'].append(miss_visual)
            traindata_missing['A'].append(traindata['A'][i])
            traindata_missing['T'].append(miss_text)
            traindata_missing['L'].append(traindata['L'][i])
            traindata_missing['F'].append(flag)
        else:
            flag = 5
            traindata_missing['ID'].append(traindata['ID'][i])
            traindata_missing['V'].append(miss_visual)
            traindata_missing['A'].append(miss_audio)
            traindata_missing['T'].append(traindata['T'][i])
            traindata_missing['L'].append(traindata['L'][i])
            traindata_missing['F'].append(flag)
    else:
        # print('nomissingmodality')
        flag = -1
        traindata_missing['ID'].append(traindata['ID'][i])
        traindata_missing['V'].append(traindata['V'][i])
        traindata_missing['A'].append(traindata['A'][i])
        traindata_missing['T'].append(traindata['T'][i])
        traindata_missing['L'].append(traindata['L'][i])
        traindata_missing['F'].append(flag)

for i in range(len(testdata['ID'])):
    rnd_is_missing = random.randint(1, fenmu)  # 0: visual  1:audio 2:text
    if rnd_is_missing == 1 and missing_num_test_true <= missing_num_test :
        missing_num_test_true = missing_num_test_true + 1
        rnd = random.randint(0, missing_type_single_multi)  # 0: visual  1:audio 2:text
        if rnd == 0:
            flag = 0
            testdata_missing['ID'].append(testdata['ID'][i])
            testdata_missing['V'].append(miss_visual)
            testdata_missing['A'].append(testdata['A'][i])
            testdata_missing['T'].append(testdata['T'][i])
            testdata_missing['L'].append(testdata['L'][i])
            testdata_missing['F'].append(flag)
        elif rnd == 1:
            flag = 1
            testdata_missing['ID'].append(testdata['ID'][i])
            testdata_missing['V'].append(testdata['V'][i])
            testdata_missing['A'].append(miss_audio)
            testdata_missing['T'].append(testdata['T'][i])
            testdata_missing['L'].append(testdata['L'][i])
            testdata_missing['F'].append(flag)
        elif rnd == 2:
            flag = 2
            testdata_missing['ID'].append(testdata['ID'][i])
            testdata_missing['V'].append(testdata['V'][i])
            testdata_missing['A'].append(testdata['A'][i])
            testdata_missing['T'].append(miss_text)
            testdata_missing['L'].append(testdata['L'][i])
            testdata_missing['F'].append(flag)
        elif rnd == 3:
            flag = 3
            testdata_missing['ID'].append(testdata['ID'][i])
            testdata_missing['V'].append(testdata['V'][i])
            testdata_missing['A'].append(miss_audio)
            testdata_missing['T'].append(miss_text)
            testdata_missing['L'].append(testdata['L'][i])
            testdata_missing['F'].append(flag)
        elif rnd == 4:
            flag = 4
            testdata_missing['ID'].append(testdata['ID'][i])
            testdata_missing['V'].append(miss_visual)
            testdata_missing['A'].append(testdata['A'][i])
            testdata_missing['T'].append(miss_text)
            testdata_missing['L'].append(testdata['L'][i])
            testdata_missing['F'].append(flag)
        else:
            flag = 5
            testdata_missing['ID'].append(testdata['ID'][i])
            testdata_missing['V'].append(miss_visual)
            testdata_missing['A'].append(miss_audio)
            testdata_missing['T'].append(testdata['T'][i])
            testdata_missing['L'].append(testdata['L'][i])
            testdata_missing['F'].append(flag)
    else:
        flag = -1
        testdata_missing['ID'].append(testdata['ID'][i])
        testdata_missing['V'].append(testdata['V'][i])
        testdata_missing['A'].append(testdata['A'][i])
        testdata_missing['T'].append(testdata['T'][i])
        testdata_missing['L'].append(testdata['L'][i])
        testdata_missing['F'].append(flag)

pickle.dump(traindata_missing, open('F:\data\iemocap\\' + missing_type + 'train7.pkl', 'wb'))
pickle.dump(testdata_missing, open('F:\data\iemocap\\' + missing_type + 'test7.pkl', 'wb'))


