import numpy as np
from numpy import ndarray


def cos_sim(v1: ndarray, v2: ndarray)-> np.float64:

    # 내적 거리
    dot_product = np.dot(v1, v2) 

    # 벡터 길이
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    return dot_product / ( norm_v1 * norm_v2 )


if __name__ == "__main__":

    np.random.seed(10)

    A = np.random.rand(16)
    B = np.random.rand(16)

    cos_s = cos_sim(A, B)


    print("vector A: ", A)
    print("vector B: ", B)
    print(" cos_sim(A, B): ", cos_s)