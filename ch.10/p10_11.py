import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from p10_89 import text_embedding

klue_mrc_dataset = load_dataset('klue', 'mrc', split='train') # 데이터 셋 load
sentence_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device='cpu') # 모델 불러옴


# 데이터셋을 학습 데이터와 테스트 데이터로 구분 | 학습 데이터 크기를 1000개로 설정
klue_mrc_dataset = klue_mrc_dataset.train_test_split(train_size=1000, shuffle=False)['train']

# 텍스트 임베딩
embeddings = sentence_model.encode(klue_mrc_dataset['context'])


# 인덱스 만들기
index = faiss.IndexFlatL2(embeddings.shape[1])

# 인덱스에 임베딩 저장
index.add(embeddings)


query = "이번 연도에는 언제 비가 많이 올까?"
query_embedding = sentence_model.encode([query])
distance, indices = index.search(query_embedding, 3)

print(distance, indices)

for idx in indices[0]:
    print(klue_mrc_dataset['context'][idx][:50])