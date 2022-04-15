import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

'''
Transformer的特点
1. 无先验假设(局部关联性，有序建模型）
2. self_attention
3. 数据量的要求与先验的程度成反比

使用模型
Encoder only  BERT, Classific
Decoder only  GPT....
Encoder-Decoder: translation

RNN VS Transformers
RNN 不能很好的进行并行运算， 但是计算复杂度相对较低

Transformers 能较好的进行并行运算，但是随着序列长度的增长， 复杂度以平方速度增长
'''


#word_embedding
batch_size = 2

max_num_src_words = 8
max_num_tgt_words = 8

max_src_seq_len = 5
max_tgt_seq_len = 5

max_position_len = 5
model_dim = torch.tensor(8)

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

# print(vaild_encoder_pos)


invaild_encoder_pos_matrix = 1 - vaild_encoder_pos_matrix
# print(invaild_encoder_pos_matrix)
masked_encoder_self_attention = invaild_encoder_pos_matrix.to(torch.bool)
# print(masked_encoder_self_attention)
#score = Q @ K.T
score = torch.randn(batch_size, max(src_len), max(src_len))

masked_score = score.masked_fill(masked_encoder_self_attention, -1e9)
prob = F.softmax(masked_score, -1)
# print(masked_score)


#intra-attention(decoder-2)
vaild_decoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0, max(tgt_len) - L)),0) for L in tgt_len]),2)
# print(vaild_decoder_pos)
vaild_cross_pos_matrix = torch.bmm(vaild_decoder_pos, vaild_encoder_pos.transpose(1,2))
# print(vaild_cross_pos_matrix)



invaild_cross_pos_matrix = 1 - vaild_cross_pos_matrix
masked_cross_attention = invaild_cross_pos_matrix.to(torch.bool)

#decoder_self_attention
valid_deocder_tri_matrix = torch.cat([torch.unsqueeze(F.pad(torch.tril(torch.ones((L, L))),(0, max(tgt_len) - L, 0, max(tgt_len) - L)),0)  for L in tgt_len],0)
invaild_decoder_tri_matrix = 1 - valid_deocder_tri_matrix
invaild_decoder_tri_matrix = invaild_decoder_tri_matrix.to(torch.bool)

score_decoder = torch.randn(batch_size, max(tgt_len), max(tgt_len))

masked_score_decoder = score_decoder.masked_fill(invaild_decoder_tri_matrix, -1e9)
prob_decoder = F.softmax(masked_score_decoder, -1)
# print(prob_decoder)


#self_attention
def scaled_dot_product_attention(Q, K, V, attention_mask):
    score = torch.bmm(Q, K.transpose(-1,-2)) / torch.sqrt(model_dim)
    masked_score = score.masked_fill(attention_mask, -1e9)
    prob = F.softmax(masked_score, -1)
    context = torch.bmm(prob, V)

    return context



# loss function

# logits ：predict
logits = torch.randn(2, 3, 4).transpose(1,2) # batch_size, seq_len, model_dim

lable = torch.randint(0,4,(2,3))

# loss = F.cross_entropy(logits, lable)
tgt_lens = torch.Tensor([2,3]).to(torch.int32)

mask = torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0, max(tgt_lens) - L)),0) for L in tgt_len])

loss = F.cross_entropy(logits, lable, reduction='none') * mask
print(loss)
























# print(src_seq)
# print(tgt_seq)



