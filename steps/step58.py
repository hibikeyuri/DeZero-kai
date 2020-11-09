if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Model
import dezero.functions as F
import dezero.layers as L
from dezero.utils import plot_dot_graph


rnn = L.RNN(10) # 隠れ層のサイズだけを指定
x = np.random.rand(1, 1)
h = rnn(x)
print(h.shape)


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)
    
    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y


seq_data = [np.random.randn(1, 1) for _ in range(10)] # ダミーの時系列データ
xs = seq_data[0:-1]
ts = seq_data[1:] # xsより1ステップ先のデータ
model = SimpleRNN(10, 1)
loss, cnt = 0, 0


for x, t in zip(xs, ts):
    y = model(x)
    loss += F.mean_squared_error(y, t)
    cnt += 1
    if cnt == 1:
        model.cleargrads()
        loss.backward()
        break

plot_dot_graph(loss, verbose=False, to_file='SimpleRNN.png')