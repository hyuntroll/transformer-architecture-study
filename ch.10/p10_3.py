from sentence_transformers import SentenceTransformer, models

# 사용한 BERT 모델 설정 [ 인코더 ]
word_embedding_model = models.Transformer('klue/roberta-base')

# 폴링 층 차원 입력
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# 두 모듈 결합
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

print(model)