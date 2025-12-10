"""Frankenstein PyTorch multiprocessing multi-stashes.

"""
from dataclasses import dataclass
from zensols.multi import MultiProcessFactoryStash
from zensols.deeplearn.batch import TorchMultiProcessStash


@dataclass(init=False)
class TorchMultiProcessFactoryStash(MultiProcessFactoryStash):
    pass


TorchMultiProcessFactoryStash._invoke_pool = TorchMultiProcessStash._invoke_pool
TorchMultiProcessFactoryStash._invoke_work = TorchMultiProcessStash._invoke_work
