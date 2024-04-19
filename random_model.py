import numpy as np

class RandomModel:
    
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
    
    def predict(self, sample):
        out = []
        for _ in range(3):
            if np.random.random() > 0.5:
                out.append(1)
            else:
                out.append(0)
        return out