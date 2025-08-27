from math import sqrt
import torch.nn.functional as F


def compute_attention(querys, keys, values, is_causal=False):
    dim_k = querys.size(-1)
    scores = querys @ keys.transpose(-2, -1) / sqrt(dim_k) # 쿼리랑 키를 곱함. 그다음 분산이 커지는 것을 막기 위해 임베딩 차원 수의 제곱근으로 나눔
    weights = F.softmax(scores, dim=1) # 스코어를 합이 1이 되도록 소프트맥스를 취해 가중치로 바꿈
    return weights @ values
