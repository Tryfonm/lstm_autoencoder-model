import sys
from pathlib import Path

import torch.nn as nn
import torch

sys.path.append('src')
from utils.logger import get_logger

LOGGER = get_logger(Path(__file__).stem)


class Encoder(nn.Module):
    def __init__(self, feature_size, embedding_size=5, num_layers=1):
        """_summary_

        Args:
            feature_size (_type_): _description_
            embedding_size (int, optional): _description_. Defaults to 5.
            num_layers (int, optional): _description_. Defaults to 1.
        """
        super(Encoder, self).__init__()
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.embedding_size,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.embedding_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.embedding_size).to(x.device)
        out, (_, _) = self.lstm(x, (h0, c0))
        LOGGER.debug(f'encoder output shape: {out.shape} {c0.shape} {h0.shape}')
        return out[:, -1, :]  # fetch the last hidden state and return it


class Decoder(nn.Module):
    def __init__(self, sequence_length, feature_size, output_size, num_layers=1):
        """_summary_

        Args:
            sequence_length (_type_): _description_
            feature_size (_type_): _description_
            output_size (_type_): _description_
            num_layers (int, optional): _description_. Defaults to 1.
        """
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.hidden_size = 2 * feature_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x, (_, _) = self.lstm(x, (h0, c0))
        x = x.reshape((-1, self.sequence_length, self.hidden_size))

        out = self.fc(x)
        LOGGER.debug(out.shape, out.shape , out[:,-1,:].shape)s
        return out


class LSTM_Autoencoder(nn.Module):
    def __init__(self, sequence_length, feature_size, embedding_size, num_layers=1):
        """_summary_

        Args:
            sequence_length (_type_): _description_
            feature_size (_type_): _description_
            embedding_size (_type_): _description_
            num_layers (int, optional): _description_. Defaults to 1.
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.embedding_size = embedding_size

        self.encoder = Encoder(
            self.feature_size,
            self.embedding_size,
            num_layers
        )
        self.decoder = Decoder(
            self.sequence_length,
            self.embedding_size,
            self.feature_size,
            num_layers
        )

    def forward(self, x):

        encoded_tensor = self.encoder(x)
        decoded_tensor = self.decoder(encoded_tensor)

        return encoded_tensor, decoded_tensor
