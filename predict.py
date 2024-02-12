# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
from typing import List
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
TOKENIZER_NAME = "bert-base-uncased"
CACHE_DIR = "checkpoints"

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME,
            model_max_length=8192,
            cache_dir=CACHE_DIR
        )
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            rotary_scaling_factor=2,
            cache_dir=CACHE_DIR
        )
        self.model.eval()

    def predict(
        self,
        sentences: str = Input(description="Input Sentence list - Each sentence should be split by a newline"),
    ) -> List[float]:
        """Run a single prediction on the model"""
        sentences = sentences.strip().splitlines()
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.tolist()[0]
        return embeddings