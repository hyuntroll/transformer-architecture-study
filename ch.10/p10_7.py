from PIL import Image
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('clip-ViT-B-32')

img_embs = model.encode([Image.open('ch.10/puppy.jpg'), Image.open('ch.10/cat.jpg')])
text_embs = model.encode(['dog', 'cat'])

cos_scores = util.cos_sim(img_embs, text_embs)
print(cos_scores)

"""
tensor([[0.2655, 0.1879],
        [0.2118, 0.2611]])
"""