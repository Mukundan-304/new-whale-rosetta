import numpy as np

embeddings = np.load(r"D:\marine\watkins_stage4\train\embeddings.npy")
print(embeddings.shape)
print(embeddings[0])  # first segment's 512-value vector