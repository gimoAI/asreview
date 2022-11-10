# Copyright 2019-2022 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import math

from transformers import FeatureExtractionPipeline
from transformers import LongformerTokenizer, LongformerModel

from asreview.models.feature_extraction.base import BaseFeatureExtraction


class LongF(BaseFeatureExtraction):
    name = "longformer"
    label = "Longformer"

    def __init__(self, *args, transformer_model="allenai/longformer-base-4096", **kwargs):
        super(LongF, self).__init__(*args, **kwargs)
        self.transformer_model = transformer_model

    def transform(self, texts):
        device = "cuda:0"
        torch.cuda.empty_cache()
        tokenizer = LongformerTokenizer.from_pretrained(self.transformer_model)
        model = LongformerModel.from_pretrained(self.transformer_model)
        X = []
        batch_size=8

        for x in range(int(len(texts)/batch_size)+1):
            x = x*batch_size
            y = x+batch_size
            if y > len(texts):
                y = len(texts)
            print("test: ", x, y)
            batch = texts[x:y]
            encoded_input = tokenizer(batch.tolist(), padding="longest", truncation=True, return_tensors='pt').to(device)
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
