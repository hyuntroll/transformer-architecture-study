
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def text_embedding():
    klue_mrc_dataset = load_dataset('klue', 'mrc', split='train') # 데이터 셋 load
    sentence_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS') # 모델 불러옴


    # 데이터셋을 학습 데이터와 테스트 데이터로 구분 | 학습 데이터 크기를 1000개로 설정
    klue_mrc_dataset = klue_mrc_dataset.train_test_split(train_size=1000, shuffle=False)['train']

    # 텍스트 임베딩
    embeddings = sentence_model.encode(klue_mrc_dataset['context'])
    return embeddings

embeddings = text_embedding()

print(embeddings, embeddings.shape)

