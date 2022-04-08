import torch
import torch.nn as nn

bs, T = 2, 3   #批量大小， 输入序列长度
input_size, hidden_size = 2, 3
input = torch.randn(bs, T, input_size)
h_prev = torch.zeros(bs, hidden_size)

rnn = nn.RNN(input_size, hidden_size, batch_first=True)
rnn_output, state_final = rnn(input, h_prev.unsqueeze(0))
print(rnn_output)
print(state_final)
print('-------------')

def rnn_forward(input, weight_ih, weight_hh, bias_ih, bias_hh, h_prev):
    bs, T, input_size = input.shape
    h_dim = weight_ih.shape[0]
    h_out = torch.zeros(bs, T, h_dim)

    for t in range(T):
        x = input[:, t, :].unsqueeze(2)
        w_ih_batch = weight_ih.unsqueeze(0).tile(bs, 1, 1)
        w_hh_batch = weight_hh.unsqueeze(0).tile(bs, 1, 1)

        w_time_x = torch.bmm(w_ih_batch, x).squeeze(-1)
        w_time_h = torch.bmm(w_hh_batch, h_prev.unsqueeze(2)).squeeze(-1)
        h_prev = torch.tanh(w_time_x + bias_ih + w_time_h + bias_hh)

        h_out[:, t, :] = h_prev
    return h_out, h_prev.unsqueeze(0)
custom_rnn_output, custom_state_final = rnn_forward(input, rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0, h_prev)

print(custom_rnn_output)
print(custom_state_final)

