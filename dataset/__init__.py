from sympy import im

from .ConvLSTM.ConvLSTM import ConvLSTMPreCorrectDataset
from .UASMLSTM.PreCorrect import PreCorrectDataset
from .UNET.unet import UNetDataset

dataset_list = {
    "PreCorrect_UASMLSTM": PreCorrectDataset,
    "UNET": UNetDataset,
    "ConvLSTM": ConvLSTMPreCorrectDataset,
}
