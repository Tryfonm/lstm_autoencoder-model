from datasets.custom_dataset import CustomDataset
from models.lstm_autoencoder import LSTM_Autoencoder
import sys
import argparse
from pathlib import Path
import datetime
from typing import Dict, Union, Type

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append('src')
from utils.logger import get_logger

LOGGER = get_logger(Path(__file__).stem)


def parse_arguments() -> Dict[str, Union[int, str]]:
    """_summary_

    Returns:
        Dict[str, Union[int, str]]: _description_
    """
    parser = argparse.ArgumentParser(
        description='')

    parser.add_argument('--num_epochs', type=int,
                        default=10, help='(default: 10)')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='(default: 8)')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='(default: 0.001)')
    parser.add_argument('--embedding_size', type=int,
                        default=5, help='(default: 5)')

    args = parser.parse_args()
    return args.__dict__


def get_save_location() -> Path:
    """_summary_

    Returns:
        Path: _description_
    """
    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%Y%m%d_%H%M")

    models_dir = Path("./models")
    if not models_dir.exists():
        models_dir.mkdir(parents=True)

    return models_dir / f"{formatted_datetime}.pth"


def train(
    model: Type[nn.Module],
    train_dataloader: Type[DataLoader],
    val_dataloader: Type[DataLoader],
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    log_dir: Union[str, Path] = './runs',
    patience: int = 5
) -> None:
    """_summary_

    Args:
        model (Type[nn.Module]): _description_
        train_dataloader (Type[DataLoader]): _description_
        val_dataloader (Type[DataLoader]): _description_
        num_epochs (int, optional): _description_. Defaults to 10.
        learning_rate (float, optional): _description_. Defaults to 0.001.
        log_dir (Union[str, Path], optional): _description_. Defaults to './runs'.
        patience (int, optional): _description_. Defaults to 5.
    """

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir=log_dir)
    save_path = get_save_location()

    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0

        model.train()
        for batch_idx, inputs in enumerate(train_dataloader):

            _, reconstructed_sequence = model(inputs)
            loss = criterion(reconstructed_sequence, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)

        val_loss = eval(model, val_dataloader, criterion)

        LOGGER.debug(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1
            if counter >= patience:
                LOGGER.debug(f"Validation loss did not improve for {patience} epochs. Training stopped.")
                break

    writer.close()


def eval(
        model: Type[nn.Module],
        val_dataloader: Type[DataLoader],
        criterion: nn.modules.loss._Loss
) -> float:
    """_summary_

    Args:
        model (Type[nn.Module]): _description_
        val_dataloader (Type[DataLoader]): _description_
        criterion (nn.modules.loss._Loss): _description_

    Returns:
        float: _description_
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, inputs in enumerate(val_dataloader):
            _, reconstructed_sequence = model(inputs)
            loss = criterion(reconstructed_sequence, inputs)
            val_loss += loss.item()
    val_loss /= len(val_dataloader)

    return val_loss


def main(
    batch_size: int,
    file_path: Union[str, Path] = '/workspace/data/job_2-29062023_2214/part-00000-9b476169-bb33-471c-a19e-ffae958ee119-c000.csv',
    embedding_size: int = 5,
    **kwargs
) -> None:
    """_summary_

    Args:
        batch_size (int): _description_
        file_path (Union[str, Path], optional): _description_. Defaults to '/workspace/data/job_2-29062023_2214/part-00000-9b476169-bb33-471c-a19e-ffae958ee119-c000.csv'.
        embedding_size (int, optional): _description_. Defaults to 5.
    """

    model = LSTM_Autoencoder(
        sequence_length=10,
        feature_size=768,
        embedding_size=embedding_size,
        num_layers=1
    )

    train_dataset = CustomDataset(file_path, train=True)
    val_dataset = CustomDataset(file_path, train=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **kwargs
    )


if __name__ == '__main__':
    args = parse_arguments()
    main(**args)
