# 0. 사용할 패키지 불러오기

from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.utils import np_utils
from keras.models import load_model
from keras.models import model_from_json
import os
import random

# 랜덤시드 고정시키기
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

# 데이터셋 생성 함수
def seq2dataset(seq, window_size):
    dataset = []
    for j in range(len(seq)):
        for i in range(len(seq[j]) - window_size):
            subset = seq[j][i:(i + window_size + 1)]
            dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)

def open_seq(code_list):
    data = []
    learn_data = []
    data_ls = []
    data_num = []
    for j in range(len(code_list)):
        data.append([])
        for k in range(len(code_list[j])):
            data[j].append(code_list[j][k])
    for j in range(len(data)):
        learn_data.append([])
        for l in range(len(data[j])):
            if data[j][l] is ',':
                if (data[j][l - 1] is 'G') or (data[j][l - 1] is 'A') or (data[j][l - 1] is 'B'):
                    data[j][l - 1] = data[j][l - 1] + data[j][l]
                    learn_data[j].pop()
                    learn_data[j].append(data[j][l-1])
            elif data[j][l] is "'":
                if data[j][l - 1] is 'c' or data[j][l - 1] is 'd':
                    data[j][l - 1] = data[j][l- 1] + data[j][l]
                    learn_data[j].pop()
                    learn_data[j].append(data[j][l - 1])
            elif data[j][l] is '|' and data[j][l - 1] is ':':
                data[j][l - 1] = data[j][l - 1] + data[j][l]
                learn_data[j].pop()
                learn_data[j].append(data[j][l - 1])
            elif data[j][l] is ':' and data[j][l - 1] is '|':
                data[j][l - 1] = data[j][l- 1] + data[j][l]
                learn_data[j].pop()
                learn_data[j].append(data[j][l - 1])
            elif data[j][l] is ':' and data[j][l - 1] is ':':
                data[j][l - 1] = data[j][l - 1] + data[j][l]
                learn_data[j].pop()
                learn_data[j].append(data[j][l - 1])
            else:
                learn_data[j].append(data[j][l])

    for g in range(len(learn_data)):
        for h in range(len(learn_data[g])):
            if not learn_data[g][h] in data_ls:
                data_ls.append(learn_data[g][h])

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


# 2. 데이터셋 생성하기
n_steps = 6  # step
n_inputs = 1  # 특성수

n = int(input("0: 밝음 1: 잔잔 2: 긴박 ="))
if n is 0:
    rhythm, code_len, chords, quick, X_code = open_file("happy.txt")
elif n is 1:
    rhythm, code_len, chords, quick, X_code = open_file("calm.txt")
else:
    rhythm, code_len, chords, quick, X_code = open_file("thrill.txt")

seq, code2idx, idx2code = open_seq(X_code)
dataset = seq2dataset(seq, window_size=n_steps)
print(dataset.shape)

# 입력(X)과 출력(Y) 변수로 분리하기
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

print("one hot encoding vector size is ", one_hot_vec_size)

# 3. 모델 구성하기
epo_count = 0
if n is 0:
    checkpoint_path = "./happy/cp.ckpt"
elif n is 1:
    checkpoint_path = "./calm/cp.ckpt"
elif n is 2:
    checkpoint_path = "./thrill/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=1)

model = Sequential()
model.add(LSTM(
    units=128,
    kernel_initializer='glorot_normal',
    bias_initializer='zero',
    batch_input_shape=(1, n_steps, n_inputs),
    stateful=True,
    name = 'lstm'
))

model.add(Dense(
    units=one_hot_vec_size,
    kernel_initializer='glorot_normal',
    bias_initializer='zero',
    activation='softmax'
))

# 4. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# 5. 모델 학습시키기
latest = tf.train.latest_checkpoint(checkpoint_dir)
#model.load_weights(latest)  #load할때만 쓰고 처음돌릴때는 주석처리

num_epochs = 500

for epoch_idx in range(num_epochs):
    epo_count = epo_count +1
    print('epoch : %d' % epo_count, end= '')
    model.fit(
        x=x_train,
        y=y_train,
        epochs=1,
        batch_size=1,
        verbose=0,
        shuffle=False,
        callbacks=[cp_callback]
    )
    model.reset_states()


# 7. 모델 평가하기
scores = model.evaluate(x_train, y_train)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
model.reset_states()

if n is 0:
    model.save('model_h.h5')
elif n is 1:
    model.save('model_c.h5')
elif n is 2:
    model.save('model_t.h5')
