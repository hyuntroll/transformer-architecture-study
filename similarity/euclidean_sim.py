import numpy as np
from numpy import ndarray


# 반향과 거리 모두 생각하는 거

def euclidean_dist(v1: ndarray, v2: ndarray)-> np.float64:

    return np.linalg.norm(v1 - v2)

if __name__ == "__main__":

    np.random.seed(0)

    A = np.random.rand(16)
    B = np.random.rand(16)

    euclidean_sim = euclidean_dist(A, B)
    print("sim: ", euclidean_sim) 

# np.linalg.norm()  각각을 제곱해서 더하고 루트 씌운거임 A-B 하면 각 차원마다 원소를 빼서 계산
