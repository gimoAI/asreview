import numpy as np

from sentence_transformers.SentenceTransformer import SentenceTransformer  # noqa  # NOQA
from asreview.models.feature_extraction.base import BaseFeatureExtraction

model = SentenceTransformer('all-mpnet-base-v2')

texts = ['This framework generates embeddings for each input',
             'Sentences are passed as a list of string yea.',
             'The quick brown fox jumps over the']

X = np.array(model.encode(texts))
print(X)
for b in X:
    print(len(b))