from .output_tool import null_output_function, precipitation_output_function

output_function_dic = {
    "Precipition": precipitation_output_function,
    "Null": null_output_function,
}


def init_output_function(config, *args, **params):
    name = config.get("output", "output_function")

    if name in output_function_dic:
        return output_function_dic[name]
    else:
        raise NotImplementedError
