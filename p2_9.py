from torch import nn
from p2_x import generate_input_embedding


# 데이터가 특정 부분을 과도하게 반영하여 정확한 예측이 힘듦
# 데이터를 정규화 하여 모든 입력변수가 비슷한 범위와 분포를 가지도록 조정

#정규화 방법 
"""
백터 x를 정규화한 norm_x
norm_x = (백터x - x의 평균) / x의 표준편차

평균이 0, 표편이 1인 분포를 가지게 됨
"""

embedding_dim = 16
max_position = 12
input_embeddings = generate_input_embedding("지금 심각한 문제가 발생함", embedding_dim, max_position)

norm = nn.LayerNorm(embedding_dim) # 층 정규화 레이어 생성
norm_x = norm(input_embeddings) # 층 정규화 레이어에 input_embeddings을 통과시켜 '층 정규화된 임베딩'으로 생성
print(norm_x.shape)

print(norm_x.mean(dim=-1).data, norm_x.std(dim=-1)) # 평균과 표준편차 확인