import tflearn
from tflearn.data_utils import *
import pandas as pd

names = pd.read_csv('townlands-no-geom.csv', usecols=['NAME_TAG'])
names['NAME_TAG'] = names.NAME_TAG.apply(lambda x: x.split('(')[0])
names = names[names.NAME_TAG.apply(len) <= 20]
maxlen = 20

string_utf8 = (names.NAME_TAG + '\n').sum()[:-1]

X, Y, char_idx = \
    string_to_semi_redundant_sequences(string_utf8, seq_maxlen=maxlen, redun_step=3)

g = tflearn.input_data(shape=[None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='townlands')

m.fit(X, Y, validation_set=0.1, batch_size=128,
  n_epoch=1, run_id='townlands')

out = ''
for i in range(10):
	seed = random_sequence_from_string(string_utf8, maxlen)
	o = m.generate(30, temperature=1.0, seq_seed=seed).encode('utf-8')
	out += o.decode('utf-8')[20:].split('\n')[1] + '\n'

print(out)
