import numpy as np
import torch
import math

from transformers import FeatureExtractionPipeline
from transformers import BigBirdModel, BigBirdTokenizer

from asreview.models.feature_extraction.base import BaseFeatureExtraction


class BigB(BaseFeatureExtraction):
    name = "bigbird"
    label = "bigbird"

    def __init__(self, *args, transformer_model="google/bigbird-roberta-large", **kwargs):
        super(BigB, self).__init__(*args, **kwargs)
        self.transformer_model = transformer_model

    def transform(self, texts):
        device = "cuda:0"
        torch.cuda.empty_cache()
        tokenizer = BigBirdTokenizer.from_pretrained(self.transformer_model)
        model = BigBirdModel.from_pretrained(self.transformer_model)
        X = []
        batch_size=8

        for x in range(int(len(texts)/batch_size)+1):
            x = x*batch_size
            y = x+batch_size
            if y > len(texts):
                y = len(texts)
            print("test: ", x, y)
            batch = texts[x:y]
            encoded_input = tokenizer(batch.tolist(), return_tensors='pt').to(device)
            model = model.to(device)

            # Compute token embedddings:
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Some classifiers cannot handle negative features
            # model_output -= model_output.min(1, keepdim=True)[0]
            # model_output /= model_output.max(1, keepdim=True)[0]

            # Perform pooling. In this case, mean pooling.
            X = X + self.mean_pooling(model_output, encoded_input['attention_mask']).tolist()

        X = np.array(X)

        # feature_extractor = FeatureExtractionPipeline(model=model, tokenizer=tokenizer, device='cuda:0')
        # X = np.array(feature_extractor(texts.tolist()), dtype=object)
        # X = np.array(model.encode(texts))
        return X

    def mean_pooling(self, model_output, attention_mask):
        # https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.clamp(torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9), min=0,max=1)