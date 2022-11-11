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

from tqdm import tqdm
from transformers import LongformerTokenizer, LongformerModel

from asreview.models.feature_extraction.base import BaseFeatureExtraction


class Longformer(BaseFeatureExtraction):
    name = "longformer"
    label = "Longformer"

    def __init__(self, *args, transformer_model="allenai/longformer-base-4096", **kwargs):
        super(Longformer, self).__init__(*args, **kwargs)
        self.transformer_model = transformer_model

    def transform(self, texts):
        device = "cuda:0"
        torch.cuda.empty_cache()
        tokenizer = LongformerTokenizer.from_pretrained(self.transformer_model)
        model = LongformerModel.from_pretrained(self.transformer_model)
        X = []
        batch_size = 16

        print("Longformer feature extraction progress: ")
        for x in tqdm(range(math.ceil(len(texts)/batch_size))):
            x = x*batch_size
            y = x+batch_size
            if y > len(texts):
                y = len(texts)
            batch = texts[x:y]
            encoded_input = tokenizer(batch.tolist(), padding="longest", truncation=True, return_tensors='pt').to(device)
            model = model.to(device)

            # Compute token embedddings:
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling. In this case, mean pooling.
            X = X + self.mean_pooling(model_output, encoded_input['attention_mask']).tolist()

        X = np.array(X)
        return X

    def mean_pooling(self, model_output, attention_mask):
        # https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.clamp(torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9), min=0,max=1)
