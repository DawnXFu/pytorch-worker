import json

from .accuracy_tool import gen_precipition_result


def null_output_function(data, config, *args, **params):
    return ""


def precipitation_output_function(data, config, *args, **params):
    result = gen_precipition_result(data)
    return json.dumps(result, sort_keys=True)
