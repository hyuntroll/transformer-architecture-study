import torch
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from p6_2 import make_prompt


# 이미 학습된 모델 아이디를 가져와서 토크나이저와 모델을 불러옴
# 하나의 파이프라인으로 만들어서 반환
def make_intference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

model_id = 'beomi/Yi-Ko-6B'
hf_pipe = make_intference_pipeline(model_id)

example = make_prompt("""CREATE TABLE players (
player_id INT PRIMARY KEY AUTO_INCREMENT,
username VARCHAR(255) UNIQUE NOT NULL,
email VARCHAR(255) UNIQUE NOT NULL,
password_hash VARCHAR(255) NOT NULL,
date_joined DATETIME NOT NULL,
last_login DATETIME
);""", "모든 플레이어 정보를 조회해 줘", "SELECT * FROM players;")

hf_pipe(example, do_sample=False,
        return_full_text=False, max_length=1024, truncation=True)