import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#word_embedding
batch_size = 2

max_num_src_words = 8
max_num_tgt_words = 8

max_src_seq_len = 5
max_tgt_seq_len = 5

max_position_len = 5
model_dim = 8

src_len = torch.Tensor([2, 4]).to(torch.int32)
tgt_len = torch.Tensor([4, 3]).to(torch.int32)

#padding concat
# tensor([[3, 3, 0, 0, 0],
#         [7, 2, 7, 3, 0]])
src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)), (0, max(src_len) - L)),0)
                     for L in src_len])
#tensor([[4, 1, 3, 1, 0],
#       [1, 2, 2, 0, 0]])
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_tgt_words, (L,)), (0, max(tgt_len) - L)),0)
                     for L in tgt_len])

src_embedding_table = nn.Embedding(max_num_src_words+1, model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words+1, model_dim)

src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)





#position embedding
# PE(pos,2i) = sin(pos / 10000_2i / d_model)
# PE(pos,2i+1) = cos(pos / 10000_2i / d_model)
pos_mat = torch.arange(max_position_len).reshape((-1,1))

i_mat = torch.pow(10000, torch.arange(0, 8, 2).reshape((1,-1)) / model_dim)

pe_embedding_table = torch.zeros(max_position_len, model_dim)
pe_embedding_table[:,0::2] = torch.sin(pos_mat / i_mat)
pe_embedding_table[:,1::2] = torch.cos(pos_mat / i_mat)

pe_embedding = nn.Embedding(max_position_len,model_dim)

pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False)

src_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)),0) for _ in src_len]).to(torch.int32)
tgt_pos = torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)),0) for _ in tgt_len]).to(torch.int32)

src_pe_embedding = pe_embedding(src_pos)
tgt_pe_embedding = pe_embedding(tgt_pos)
# print(src_pe_embedding)
# Encoder_Muti_Head_Attention
# Scaled Dot-Product Attention Attention(Q, K, V) = softmax(Q K.T / sqr(d_k)) V
#1.sotfmax
s = torch.randn(5)   # tensor([-0.6681, -0.2248, -0.8520, -1.2385, -0.3548])
# print(s)
s_softmax = F.softmax(s, -1)

# print(s_softmax) # tensor([0.1879, 0.2927, 0.1563, 0.1062, 0.2570])
alpha1 = torch.tensor(0.1)
alpha2 = torch.tensor(10)
s1_softmax = F.softmax(s * alpha1, -1) #tensor([0.1999, 0.2089, 0.1962, 0.1888, 0.2062])
s2_softmax = F.softmax(s * alpha2, -1) #tensor([9.2353e-03, 7.7738e-01, 1.4672e-03, 3.0763e-05, 2.1189e-01])


# print(s1_softmax)
# print(s2_softmax)

def jaco(s):
    return F.softmax(s)

jaco_mat1 = torch.autograd.functional.jacobian(jaco, s*alpha1)
jaco_mat2 = torch.autograd.functional.jacobian(jaco, s*alpha2)

# print(jaco_mat1)
# print(jaco_mat2)

vaild_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0, max(src_len) - L)),0) for L in src_len]),2)
vaild_encoder_pos_matrix = torch.bmm(vaild_encoder_pos, vaild_encoder_pos.transpose(1,2))




invaild_encoder_pos_matrix = 1 - vaild_encoder_pos_matrix
print(invaild_encoder_pos_matrix)
masked_encoder_self_attention = invaild_encoder_pos_matrix.to(torch.bool)
print(masked_encoder_self_attention)
#score = Q K.T
score = torch.randn(batch_size, max(src_len), max(src_len))

masked_score = score.masked_fill(masked_encoder_self_attention, -1e9)
prob = F.softmax(masked_score, -1)
print(masked_score)
print(prob)



























# print(src_seq)
# print(tgt_seq)



