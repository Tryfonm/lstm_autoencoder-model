import sys
from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel

sys.path.append('src')
from utils.logger import get_logger

LOGGER = get_logger(Path(__file__).stem)


class SentenceEncoder:
    def __init__(self, model_name: str) -> None:
        """_summary_

        Args:
            model_name (str): _description_
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, sentence: str) -> List:
        tokens = self.tokenizer.tokenize(sentence)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids])
        with torch.no_grad():
            outputs = self.model(input_ids)
            sentence_representation = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return list(sentence_representation.numpy())


class CustomDataset(Dataset):
    def __init__(
        self,
        csv_file: Union[str, Path],
        train: bool = True,
        encoder='distilbert-base-uncased'
    ) -> None:
        """_summary_

        Args:
            csv_file (Union[str, Path]): _description_
            train (bool, optional): _description_. Defaults to True.
            encoder (str, optional): _description_. Defaults to 'distilbert-base-uncased'.
        """
        self.raw_data = pd.read_csv(csv_file, header=0, sep=':::', engine='python', index_col=0)
        self._encoder = SentenceEncoder(encoder)
        if train:
            self.raw_data = self.raw_data.iloc[0:round(0.8*self.raw_data.shape[0])]
        else:
            self.raw_data = self.raw_data.iloc[0:round(0.2*self.raw_data.shape[0])]
        self.processed_data = self.raw_data.applymap(self._encoder)

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, index) -> torch.tensor:
        self.processed_data = self.raw_data.applymap(self._encoder)
        input = torch.tensor(self.processed_data.iloc[index, :].values.tolist())
        input_data = input

        return input_data
