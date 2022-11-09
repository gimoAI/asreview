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

        feature_extractor = FeatureExtractionPipeline(model=model, tokenizer=tokenizer, device='cuda:0')
        X = np.array(feature_extractor(texts.tolist()))
        # X = np.array(model.encode(texts))
        return X