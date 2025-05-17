from .cv.ImageFromJson import ImageFromJsonDataset
from .nlp.JsonFromFiles import JsonFromFilesDataset
from .others.FilenameOnly import FilenameOnlyDataset
from .UASMLSTM.PreCorrect import PreCorrectDataset
from .UNET.unet import UNetDataset

dataset_list = {
    "ImageFromJson": ImageFromJsonDataset,
    "JsonFromFiles": JsonFromFilesDataset,
    "FilenameOnly": FilenameOnlyDataset,
    "PreCorrect_UASMLSTM": PreCorrectDataset,
    "UNET": UNetDataset,
}
