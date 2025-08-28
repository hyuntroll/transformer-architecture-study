# 띄어쓰기 단위로 분리
input_text = "지금 심각한 문제가 발생함"
input_text_list = input_text.split() 
print("input_text_list:", input_text_list)


# 토근 -> 아이디 딕셔너리와 아이디 -> 토큰 딕셔너리 생성
str2idx = {word:idx for idx, word in enumerate(input_text_list)}
idx2str = {idx:word for idx, word in enumerate(input_text_list)}
print("str2idx:", str2idx)
print("idx2str:", idx2str)


# 토큰을 토큰 아이디로 변환
input_ids = [str2idx[word] for word in input_text_list]
print("input_dis:", input_ids)

