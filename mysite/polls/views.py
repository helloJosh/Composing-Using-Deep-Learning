# -*- coding: utf-8 -*-

from django.shortcuts import render, get_object_or_404
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from ipykernel import kernelapp as app
import music21
import os
from tensorflow import keras
from django.http import HttpResponse
from django.urls import reverse
from tensorflow.keras.layers import LSTM
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.layers.recurrent import RNN
import numpy as np
from keras.models import load_model, Model
import random
from keras.utils import np_utils
from .forms import UserForm
from wsgiref.util import FileWrapper
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse

np.random.seed(5)

# 손실 이력 클래스 정의
class LossHistory(keras.callbacks.Callback):
    def init(self):
        self.epoch = 0
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        print("epoch: {0} - loss: {1:8.6f}".format(self.epoch, logs.get('loss')))
        self.epoch += 1

def index(request):
    return render(request, 'polls/index.html')

def down(request):
    fs = FileSystemStorage("")
    response = FileResponse(fs.open('new_music.mid', 'rb'), content_type='application/force-download')
    response['Content-Disposition'] = 'attachment; filename="music.mid"'
    return response

def create_happy(request):
    s = sampling(1)
    note_seq = ""
    for note in s:
        note_seq += note + " "

    conv_midi = music21.converter.subConverters.ConverterMidi()

    m = music21.converter.parse(s, format='abc')

    # midi file create
    m.write("midi", fp="./new_music.mid")

    return render(request, 'polls/music.html')

def create_calm(request):
    s = sampling(2)
    note_seq = ""
    for note in s:
        note_seq += note + " "

    conv_midi = music21.converter.subConverters.ConverterMidi()

    m = music21.converter.parse(s, format='abc')

    # midi file create
    m.write("midi", fp="./new_music.mid")

    return render(request, 'polls/music.html')

def create_urgency(request):
    s = sampling(3)
    note_seq = ""
    for note in s:
        note_seq += note + " "
    print(s)
    conv_midi = music21.converter.subConverters.ConverterMidi()

    m = music21.converter.parse(s, format='abc')

    # midi file create
    m.write("midi", fp="./new_music.mid")
    return render(request, 'polls/music.html')

def open_seq(code_list):
    data = []
    learn_data = []
    data_ls = []
    data_num = []
    for j in range(len(code_list)):
        for k in range(len(code_list[j])):
            data.append(code_list[j][k])
    for l in range(len(data)):
        if data[l] is ',':
            if (data[l - 1] is 'G') or (data[l - 1] is 'A') or (data[l - 1] is 'B'):
                data[l - 1] = data[l - 1] + data[l]
                learn_data.pop()
                learn_data.append(data[l-1])
        elif data[l] is "'":
            if data[l - 1] is 'c' or data[l - 1] is 'd':
                data[l - 1] = data[l- 1] + data[l]
                learn_data.pop()
                learn_data.append(data[l - 1])
        elif data[l] is '|' and data[l - 1] is ':':
            data[l - 1] = data[l - 1] + data[l]
            learn_data.pop()
            learn_data.append(data[l - 1])
        elif data[l] is ':' and data[l - 1] is '|':
            data[l - 1] = data[l- 1] + data[l]
            learn_data.pop()
            learn_data.append(data[l - 1])
        elif data[l] is ':' and data[l - 1] is ':':
            data[l - 1] = data[l - 1] + data[l]
            learn_data.pop()
            learn_data.append(data[l - 1])
        else:
            learn_data.append(data[l])

    for g in learn_data:
        if not g in data_ls:
            data_ls.append(g)

    for z in range(0, len(data_ls)):
        data_num.append(z)
    data2num = dict(zip(data_ls, data_num))
    num2data = dict(zip(data_num, data_ls))

    return learn_data, data2num, num2data

def open_file(filename):
    f = open(filename, 'r', encoding='utf-16')
    M = []
    L = []
    K = []
    Q = []
    X = []
    tmp = ''
    count = 0
    while True:
        line = f.readline()
        if not line: break
        if line[0] is 'X':
            count = 0
            X.append(tmp)
            tmp = ''
        if line[0] is 'M':
            M.append(line[2:])
        if line[0] is 'L':
            L.append(line[2:])
        if line[0] is 'K':
            K.append(line[2:])
            count = count + 1
            continue
        if line[0] is 'Q':
            Q.append(line[2:])
        if len(Q) < len(K):
            Q.append('no')
        if len(L) < len(K):
            L.append('no')
        if len(M) < len(K):
            M.append('no')
        if count is 1:
            tmp = tmp + line
    f.close()
    X.append(tmp)
    del X[0]
    return M, L, K, Q, X

def sampling(mood):
    n_steps = 6  # step
    if mood is 2:
        n_steps = 5
    n_inputs = 1  # 특성수
    path = ""

    # model file select(mood)

    if mood is 1:
        target_txt = "happy.txt"
        target_model = "model_h.h5"
    elif mood is 2:
        target_txt = "calm.txt"
        target_model = "model_c.h5"
    elif mood is 3:
        target_txt = "thrill.txt"
        target_model = "model_t.h5"

    model = keras.models.load_model(path + target_model)

    rhythm, code_len, chords, quick, X_code = open_file(path + target_txt)

    seq, code2idx, idx2code = open_seq(X_code)
    dataset = []
    window_size = n_steps
    for i in range(len(seq) - window_size):
        subset = seq[i:(i + window_size + 1)]
        dataset.append([code2idx[item] for item in subset])
    dataset = np.array(dataset)

    x_train = dataset[:, 0:n_steps]
    y_train = dataset[:, n_steps]

    max_idx_value = len(code2idx) - 1
    # 입력값 정규화 시키기
    x_train = x_train / float(max_idx_value)

    # 입력을 (샘플 수, 타입스텝, 특성 수)로 형태 변환
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], n_inputs))

    # 라벨값에 대한 one-hot 인코딩 수행
    y_train = np_utils.to_categorical(y_train)

    one_hot_vec_size = y_train.shape[1]

    pred_count = 200  # 최대 예측 개수 정의

    # 곡 전체 예측

    in_list = ['G,', 'A,', 'B,', 'C', 'D', 'E', 'F', 'G', 'A', 'B', 'c', 'd', 'e', 'f', 'g', 'a', 'b', "c'", "d'", 1, 2, 3, 4]
    seq_in = []
    state = 0  # 0: 알파벳 , 1: 숫자
    random_num = 0  # 랜덤으로 뽑은 횟수

    while random_num < n_steps:
        select_ch = random.choice(in_list)
        if not select_ch in seq:
            continue
        if random_num == 0:  # 문자만 가능
            if select_ch.isdigit() is False:
                seq_in.append(select_ch)
                random_num = random_num + 1
        else:
            if state == 0:
                if select_ch.isdigit() is False:
                    seq_in.append(select_ch)
                    random_num = random_num + 1
                    state = 0
                elif select_ch.isdigit() is True:
                    seq_in.append(select_ch)
                    random_num = random_num + 1
                    state = 1
            elif state == 1:
                if select_ch.isdigit() is False:
                    seq_in.append(select_ch)
                    random_num = random_num + 1
                    state = 0

    seq_out = seq_in
    seq_in = [code2idx[it] / float(max_idx_value) for it in seq_in]  # 코드를 인덱스값으로 변환

    for p in range(pred_count):
        sample_in = np.array(seq_in)
        sample_in = np.reshape(sample_in, (1, n_steps, n_inputs))  # 샘플 수, 타입스텝 수, 속성 수
        pred_out = model.predict(sample_in)
        idx = np.argmax(pred_out)
        seq_out.append(idx2code[idx])
        seq_in.append(idx / float(max_idx_value))
        seq_in.pop(0)

    model.reset_states()

    m_result = ''.join(random.sample(rhythm, 1))
    l_result = ''.join(random.sample(code_len, 1))
    k_result = ''.join(random.sample(chords, 1))
    q_result = ''.join(random.sample(quick, 1))
    print(seq_out)
    count = 0  # 도돌이표 수
    for o in range(len(seq_out)):
        if seq_out[o] == '|:':
            if count == 0:
                count = count + 1
            elif count == 1:
                seq_out[o] = ':||:'
        elif seq_out[o] == ':|':
            if count == 1:
                count = 0
        elif seq_out[o] == '::':
            if count != 1:
                count = 1
    if count == 1:
        seq_out.append(':|')
    for i in range(len(seq_out)):
        if seq_out[i] == '/':
            count_s = count_s+1
            if count_s == 6:
                seq_out[i] = ''
                seq_out[i-1] = ''
                seq_out[i-2] = ''
                seq_out[i-3] = ''
                seq_out[i-4] = ''
                seq_out[i-5] = ''
        else:
            count_s = 0
    print(seq_out)
    code = ''.join(seq_out)

    composition = ["X:1\n", "T:sample\n", "M:"+m_result, "L:"+l_result, "Q:"+q_result, "K:"+k_result, code]
    if m_result is 'no':
    	del composition[2]
    if l_result is 'no':
    	for i in range(len(composition)):
    		if "L:" in composition[i]:
    			del composition[i]
    			break
    if q_result is 'no':
    	for i in range(len(composition)):
    		if "Q:" in composition[i]:
    			del composition[i]
    			break
    end_code = ''
    print("full song prediction : ")
    for i in range(len(composition)):
    	end_code = end_code + ''.join(composition[i])
    	if "\n" in composition[i]:
    		composition[i] = composition[i].replace("\n", "")
    	print(composition[i])
    return ''.join(end_code)



