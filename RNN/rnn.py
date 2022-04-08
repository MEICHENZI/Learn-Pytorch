import torch
import torch.nn as nn


#Ht = Act(Wih * Xi + Bih + Whh * H(t-1) + bhh)


#单向RNN,单层
'''
Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two RNNs together to form a `stacked RNN`,
            with the second RNN taking in outputs of the first RNN and
            computing the final results. Default: 1
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
'''
single_rnn = nn.RNN(4,3,1, batch_first=True)
# print(single_rnn)
input = torch.randn(1, 2, 4)
# hn = torch.randn(1, 2, 4)
'''
output tensor([[[ 0.4304,  0.4892,  0.3170],
         [ 0.8546,  0.2835, -0.1296]]], grad_fn=<TransposeBackward1>)
h_n(The output of the last layer):tensor([[[ 0.8546,  0.2835, -0.1296]]], grad_fn=<StackBackward0>)

'''
output, h_n = single_rnn(input)
# print(output,'\n')
# print(h_n)
# print(output.shape)torch.Size([1, 2, 3])
# print(h_n.shape)torch.Size([1, 1, 3])
#双向RNN,单层
bidirectional_rnn = nn.RNN(4,3,1, batch_first=True,bidirectional=True)
output_b, h_n = bidirectional_rnn(input)
# bidirectional_rnn: hidden_size * 2
print(output_b)
# print(output_b.shape) torch.Size([1, 2, 6])
# print(h_n.shape) torch.Size([2, 1, 3])
print(h_n)

#手动实现
batch_size, T = 2, 3
input_size, hidden_size = 2, 3
input_x = torch.randn(batch_size, T, input_size)
h_prev = torch.zeros(batch_size, hidden_size)

#rnn forward
def rnn_forward(input, weight_ih, bias, weight_hh, bias_hh, h_prev):




