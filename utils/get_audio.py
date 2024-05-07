import librosa
import numpy as np
import os
import pickle

audiofilepath = "../AudioCapture/"
audiofile = os.listdir(audiofilepath)
# print (len(files))
# audiofeature = []
audiofeature = ''
audio_dict = {}
a= 0
def adjust_array_forA(arr):
    if arr.shape[0] > 150:
        return arr[:150, :]
    elif arr.shape[0] < 150:
        padding = np.zeros((150 - arr.shape[0], 33))
        return np.concatenate((arr, padding), axis=0)
    else:
        return arr

for file in audiofile:
    tmp_name = '../AudioCapture/' + file
    y, sr = librosa.load(tmp_name, sr=None)
    hops = 512

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    f0 = librosa.feature.zero_crossing_rate(y, hops)

    mfcc = librosa.feature.mfcc(y, sr)

    cqt = librosa.feature.chroma_cqt(y_harmonic, sr)

    audiofeature = np.transpose(np.concatenate([f0,mfcc,cqt], 0))

    # audiolist.append(features)
    # audio_dict[file[:-4]] = features
    a+=1
    if a %10 == 0:
        print (a)
# pickle.dump(audio_dict, open('./audio_dict.pkl', 'wb'))
audiofeature = adjust_array_forA(audiofeature)
print (audiofeature)