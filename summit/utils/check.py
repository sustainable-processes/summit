
def check(variable_name, variable, variable_type):
    if type(variable_type) != lis:
        variable_type = [variable_type]
    for variable in variable_type:
        if type(variable) == variable_type:
            pass
        else:
            raise ValueError(f'{variable_name} must be of type {variable_type}.')