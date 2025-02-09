import pytorch_lightning as pl
import torch
from tango.common import Registrable


class DataModule(pl.LightningDataModule, Registrable):
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        raise NotImplementedError

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        raise NotImplementedError

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        raise NotImplementedError
