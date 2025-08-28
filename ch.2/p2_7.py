from math import sqrt
import torch.nn.functional as F
from torch import nn
import torch
from p2_5 import compute_attention


class AttentionHead(nn.Module):
    def __init__(self, token_embed_dim, head_dim, is_causal=False): # head_dim -> 출력 차원
        super().__init__()
        self.is_causal = is_causal
        
        # 선형 층 생성
        self.weight_q = nn.Linear(token_embed_dim, head_dim) 
        self.weight_k = nn.Linear(token_embed_dim, head_dim)
        self.weight_v = nn.Linear(token_embed_dim, head_dim)

    def forward(self, querys, keys, values):
        outputs = compute_attention(
            self.weight_q(querys),
            self.weight_k(keys),
            self.weight_v(values),
            self.is_causal
        )
        return outputs
    
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

    attention_head = AttentionHead(embedding_dim, embedding_dim)
    after_attention_embeddings = attention_head(input_embeddings, input_embeddings, input_embeddings)  


    print(input_embeddings.shape, after_attention_embeddings.shape)