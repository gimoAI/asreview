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
        tokenizer = LongformerTokenizer.from_pretrained(self.transformer_model)
        model = LongformerModel.from_pretrained(self.transformer_model)

        encoded_input = tokenizer(texts.tolist(), padding="longest", truncation=True, return_tensors='pt')

        # Compute token embedddings:
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, mean pooling.
        X = np.array(self.mean_pooling(model_output, encoded_input['attention_mask']))

        # feature_extractor = FeatureExtractionPipeline(model=model, tokenizer=tokenizer, device='cuda:0')
        # X = np.array(feature_extractor(texts.tolist()), dtype=object)
        # X = np.array(model.encode(texts))
        return X

    def mean_pooling(self, model_output, attention_mask):
        # https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
