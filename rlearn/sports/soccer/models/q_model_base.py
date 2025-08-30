import pytorch_lightning as pl
from tango.common import Registrable
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class QModelBase(pl.LightningModule, Registrable):
    def __init__(self) -> None:
        super().__init__()
