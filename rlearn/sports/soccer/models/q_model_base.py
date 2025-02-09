import pytorch_lightning as pl
from tango.common import Registrable


class QModelBase(pl.LightningModule, Registrable):
    def __init__(self) -> None:
        super().__init__()
