import sys

from .avmnist import *
from .pnlp import *
from datasets.mm_imdb.get_processed_mmimdb import *
from .mm_imdb import *
from .multioff import *
from datasets.mm_imdb.get_processed_mmimdb import *
from .mmhs150 import *
from .mimic import *


def get_data_module(data_type: str) -> type[pl.LightningDataModule]:
    return getattr(sys.modules[__name__], data_type)
