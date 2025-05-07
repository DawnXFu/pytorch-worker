from .accuracy_tool import (
    PrecipitionCorrection_accuracy_function,
    multi_label_accuracy,
    null_accuracy_function,
    single_label_top1_accuracy,
)

accuracy_function_dic = {
    "PrecipitionCorrection": PrecipitionCorrection_accuracy_function,
    "SingleLabelTop1": single_label_top1_accuracy,
    "MultiLabel": multi_label_accuracy,
    "Null": null_accuracy_function,
}


def init_accuracy_function(config, *args, **params):
    name = config.get("output", "accuracy_method")
    if name in accuracy_function_dic:
        return accuracy_function_dic[name]
    else:
        raise NotImplementedError
