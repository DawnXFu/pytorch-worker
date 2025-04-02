from .nlp.BasicBert import BasicBert
from .UASMLSTM.model import UASMLSTM

model_list = {
    "BasicBert": BasicBert,
    "UASMLSTM": UASMLSTM
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
