import logging
import warnings
import sys

from .data import process as data_process
from .models import modules
from .models.core import (MCANModel, BinaryClassifier)

sys.modules['mcan.modules'] = modules

warnings.filterwarnings('always', module='mcan')

logging.basicConfig()
logging.getLogger('mcan.data.field').setLevel(logging.INFO)


def process(*args, **kwargs):
    return data_process(*args, **kwargs)


__all__ = [
    'process', 'modules'
]

_check_nan = True


def disable_nan_checks():
    _check_nan = False


def enable_nan_checks():
    _check_nan = True
