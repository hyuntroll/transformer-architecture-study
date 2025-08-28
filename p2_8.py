from math import sqrt
import torch.nn.functional as F
from torch import nn
import torch
from p2_5 import compute_attention


class MultiheadAttention(nn.Module):
    def __init__(self, token_embed_dim, d_model, n_head, is_causal=False): # d_model -> 헤드의 수
        # head_dim은 각 헤드별 Q, K, V의 차원임 
        # 반면 d_model은 트렌스포머 전체에서 사용하는 임베딩 차원임 | 출력 백터 크기를 뜻함

        super().__init__()
        self.n_head = n_head
        self.is_causal = is_causal
        # 선형 층 생성
        self.weight_q = nn.Linear(token_embed_dim, d_model) 
        self.weight_k = nn.Linear(token_embed_dim, d_model)
        self.weight_v = nn.Linear(token_embed_dim, d_model)
        self.concat_linear = nn.Linear(d_model, d_model)

    def forward(self, querys, keys, values):

        # 1. q, k, v를 n_head 갯수만큼 쪼갬 | 이때 dim_head 를 작게 만들어서 이 스케일만큼 실행
        B, T, C = querys.size()
        querys = self.weight_q(querys).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        keys = self.weight_k(keys).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        values = self.weight_v(values).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 각각의 어텐션을 구함 h번 하게 됨
        attention = compute_attention(querys, keys, values, self.is_causal)

        #입력과 같은 형태로 변환
        output = attention.transpose(1, 2).contiguous().view(B, T, C) # 어텐션을 연결하는 단계
        output = self.concat_linear(output) # 하나로 합침
        return output
    
if __name__ == "__main__":

    # 띄어쓰기 단위로 분리
    input_text = "지금 심각한 문제가 발생함"
    input_text_list = input_text.split() 
    # 토근 -> 아이디 딕셔너리와 아이디 -> 토큰 딕셔너리 생성
    str2idx = {word:idx for idx, word in enumerate(input_text_list)}
    input_ids = [str2idx[word] for word in input_text_list]

    embedding_dim = 16 # 임베딩 차원
    max_position = 12 # 최대 토큰 수
    embed_layer = nn.Embedding(len(str2idx), embedding_dim) # 임베딩 레이어 생성
    position_embed_layer = nn.Embedding(max_position, embedding_dim) # 위치 임베딩 층

    position_ids = torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0) # 위치 아이디
    position_embeddings = position_embed_layer(position_ids)
    token_embeddings = embed_layer(torch.tensor(input_ids)) # (4, 16) 임의의 숫자 집합으로 바꿔줌
    token_embeddings = token_embeddings.unsqueeze(0) # (1, 4, 16)
    input_embeddings = token_embeddings + position_embeddings  # 최종 입력 임베딩


    n_head = 4
    mh_attention = MultiheadAttention(embedding_dim, embedding_dim, n_head)
    after_attention_embeddings = mh_attention(input_embeddings, input_embeddings, input_embeddings)  

    print(input_embeddings.shape, after_attention_embeddings.shape)