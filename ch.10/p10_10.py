import faiss
from p10_89 import text_embedding

embedding = text_embedding()

# 인덱스 만들기
index = faiss.IndexFlatL2(embedding.shape[1])

# 인덱스에 임베딩 저장
index.add(embedding)