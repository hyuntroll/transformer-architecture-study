from math import sqrt
import torch.nn.functional as F

"""
쿼리, 키, 값을 통해 관계를 계산하는 어텐션 연산

여러가지 방법이 있지만 여기서는 
"스케일 점곱 방식" 사용

1. 쿼리와 키를 곱함 (관계 계산) => 이를 통해 관련도를 구할 수 있음
2. 이때 분산이 커지는 것을 방지하기 위해 임베딩 차원수(dim_k)의 제곱근으로 나눔 
Score(i, j) = j가 i에 얼마나 관련도가 있는지
3. 쿼리와 키를 곱해 계산한 스코어(scores)를 합이 1이 되도록 소프트맥스를 취해 '가중치' 로 바꿈
weights = j가 i와 관련있는지를 0~1사이 값으로
4. 가중치와 값을 곱해 입력과 동일한 형태의 출력을 반환

"""


def compute_attention(querys, keys, values, is_causal=False):
    dim_k = querys.size(-1)
    scores = querys @ keys.transpose(-2, -1) / sqrt(dim_k) # 쿼리랑 키를 곱함. 그다음 분산이 커지는 것을 막기 위해 임베딩 차원 수의 제곱근으로 나눔 
    weights = F.softmax(scores, dim=1) # 스코어를 합이 1이 되도록 소프트맥스를 취해 가중치로 바꿈
    return weights @ values # 우리가 구하려는 가중합을 구함
