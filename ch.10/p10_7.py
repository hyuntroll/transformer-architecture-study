from PIL import Image
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('clip-ViT-B-32')

img_embs = model.encode(Image.open('ch.10/puppy.jpg'), Image.open('ch.10/cat.png'))
text_embs = model.encode(['황금 리트리버 강아지', '고양이'])

cos_scores = util.cos_sim(img_embs, text_embs)
print(cos_scores)