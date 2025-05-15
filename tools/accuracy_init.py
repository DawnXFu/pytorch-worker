from .accuracy_tool import PrecipitionCorrection_accuracy_function, null_accuracy_function

accuracy_function_dic = {
    "PrecipitionCorrection": PrecipitionCorrection_accuracy_function,
    "Null": null_accuracy_function,
}


def init_accuracy_function(config, *args, **params):
    name = config.get("output", "accuracy_method")
    if name in accuracy_function_dic:
        return accuracy_function_dic[name]
    else:
        raise NotImplementedError
