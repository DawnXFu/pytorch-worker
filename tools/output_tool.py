import json

from .accuracy_tool import gen_micro_macro_result, gen_precipition_result


def null_output_function(data, config, *args, **params):
    return ""


def precipitation_output_function(data, config, *args, **params):
    result = gen_precipition_result(data)
    return json.dumps(result, sort_keys=True)


def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)
