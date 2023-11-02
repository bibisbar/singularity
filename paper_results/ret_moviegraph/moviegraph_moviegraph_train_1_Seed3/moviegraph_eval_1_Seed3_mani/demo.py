import torch
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization

from examples import utils

from sklearn.model_selection import train_test_split



#load tensor to cpu
#torch load -cpu
text_feature_pos = torch.load('text_feature_pos.pt')
text_feature_neg = torch.load('text_feature_neg.pt')
#-numpy
text_feature_pos = text_feature_pos.numpy()
text_feature_neg = text_feature_neg.numpy()
#600*512
pca_text_feature_pos = initialization.pca(text_feature_pos, random_state=42)
#600*2
plt.scatter(pca_text_feature_pos[:, 0], pca_text_feature_pos[:, 1], c='r', label='pos')
