import torch

# 띄어쓰기 단위로 분리
input_text = "지금 심각한 문제가 발생함"
input_text_list = input_text.split() 
# 토근 -> 아이디 딕셔너리와 아이디 -> 토큰 딕셔너리 생성
str2idx = {word:idx for idx, word in enumerate(input_text_list)}

input_ids = [str2idx[word] for word in input_text_list]

embedding_dim = 16 # 임베딩 차원
embed_layer = torch.nn.Embedding(len(str2idx), embedding_dim) # 임베딩 레이어 생성

input_embeddings = embed_layer(torch.tensor(input_ids)) # (4, 16) 임의의 숫자 집합으로 바꿔줌
input_embeddings = input_embeddings.unsqueeze(0) # (1, 4, 16)

print(input_embeddings.shape)