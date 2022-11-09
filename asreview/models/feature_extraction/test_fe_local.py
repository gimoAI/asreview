import numpy as np
import torch
from transformers import FeatureExtractionPipeline
from transformers import LongformerTokenizer, LongformerModel

from asreview import ASReviewData


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


name = "longformer"
label = "Longformer"

sentences = ['This framework generates embeddings for each input',
             'Sentences are passed as a list of string yea.',
             'The quick brown fox jumps over the']


asr = ASReviewData.from_file("C:/Users/gijsm/OneDrive - Universiteit Utrecht/ASReview Stage/Datasets/Patients Retransitioning from Biosimilar TNFα Inhibitor/SLR_rosanne_composed - kopie.ris")

# Load models
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

# Tokenize sentences
encoded_input = tokenizer(sentences, padding="longest", truncation=True, return_tensors='pt')

print(encoded_input['attention_mask'])

# Compute token embedddings:
with torch.no_grad():
    model_output = model(**encoded_input)

print(model_output)

# Perform pooling. In this case, max pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# feature_extractor = FeatureExtractionPipeline(model=model, tokenizer=tokenizer, device='cuda:0')
# X = np.array(feature_extractor(sentences, max_length=True, padding="max_length"))

print(np.array(sentence_embeddings))

# for b in X:
#     for y in b:
#         print(len(y))
#         for z in y:
#             print(len(z))
