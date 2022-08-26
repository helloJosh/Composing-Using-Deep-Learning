
# 0. 사용할 패키지 불러오기

from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.utils import np_utils
from keras.models import load_model
from keras.models import model_from_json
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
if n == 0:
    rhythm, code_len, chords, quick, X_code = open_file("happy.txt")
elif n == 1:
    rhythm, code_len, chords, quick, X_code = open_file("calm.txt")
else:
    rhythm, code_len, chords, quick, X_code = open_file("thrill.txt")

seq, code2idx, idx2code = open_seq(X_code)
dataset = seq2dataset(seq, window_size=n_steps)
print(seq)
print(code2idx)
print(idx2code)
print(dataset.shape)
print(dataset)
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


# 8. 모델 사용하기
'''
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("model_w.h5")
print("loaded model from disk")
'''
if n is 0:
    model = keras.models.load_model('model_h.h5')
elif n is 1:
    model = keras.models.load_model('model_c.h5')
elif n is 2:
    model = keras.models.load_model('model_t.h5')

pred_count = 200  # 최대 예측 개수 정의


# 곡 전체 예측


in_list = ['G,','A,','B,','C','D','E','F','G','A','B','c','d','e','f','g','a','b',"c'","d'", 1,2,3,4]
seq_in=[]
state = 0  # 0: 알파벳 , 1: 숫자
random_num = 0 #랜덤으로 뽑은 횟수
random_count = 0 # 문자가 데이터셋에 포함 유무/  0 : 무, 1 : 유
while random_num <n_steps:
    select_ch = random.choice(in_list)
    for i in range(len(seq)):
        if select_ch in seq[i]:
            random_count = 1
            break
    if random_count == 0:
        continue
    if random_num == 0: #문자만 가능
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
    random_count = 0
print(seq_in)
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
count = 0 #도돌이표 수
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
print(seq_out)
code = ''.join(seq_out)

composition = ["X:1", "T:sample", "M:"+m_result, "L:"+l_result, "Q:"+q_result, "K:"+k_result, code]
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

print("full song prediction : ")
for i in range(len(composition)):
	if "\n" in composition[i]:
		composition[i] = composition[i].replace("\n", "")
	print(composition[i])

'''
import music21

note_seq = ""
for note in seq_out:
    note_seq += note + " "

conv_midi = music21.converter.subConverters.ConverterMidi()

m = music21.converter.parse("2/4 " + note_seq, format='tinyNotation')

m.show("midi")
'''
