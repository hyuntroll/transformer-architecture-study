import torch

# 띄어쓰기 단위로 분리
input_text = "지금 심각한 문제가 발생함"
input_text_list = input_text.split() 
# 토근 -> 아이디 딕셔너리와 아이디 -> 토큰 딕셔너리 생성
str2idx = {word:idx for idx, word in enumerate(input_text_list)}

input_ids = [str2idx[word] for word in input_text_list]

embedding_dim = 16 # 임베딩 차원
max_position = 12 # 최대 토큰 수
embed_layer = torch.nn.Embedding(len(str2idx), embedding_dim) # 임베딩 레이어 생성
position_embed_layer = torch.nn.Embedding(max_position, embedding_dim) # 위치 임베딩 층

position_ids = torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0) # 위치 아이디
position_embeddings = position_embed_layer(position_ids)
token_embeddings = embed_layer(torch.tensor(input_ids)) # (4, 16) 임의의 숫자 집합으로 바꿔줌
token_embeddings = token_embeddings.unsqueeze(0) # (1, 4, 16)
input_embeddings = token_embeddings + position_embeddings  # 최종 입력 임베딩


head_dim = 16

#쿼리 키 값을 계산하기 위한 변환 ( 가중치는 nn.Linear층을 사용하여 구현 가능 )
weight_q = torch.nn.Linear(embedding_dim, head_dim)
weight_k = torch.nn.Linear(embedding_dim, head_dim)
weight_v = torch.nn.Linear(embedding_dim, head_dim)

# 변환 수행
querys = weight_q(input_embeddings)
keys = weight_k(input_embeddings)
values = weight_v(input_embeddings)

# print(querys, keys, values)

