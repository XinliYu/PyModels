def is_class(variable):
    return isinstance(variable, type)


def is_basic_type(variable):
    return type(variable) in (int, float, str)


def is_str(variable):
    return type(variable) is str


def take_element_if_list(potential_list, i:int):
    return potential_list[i] if isinstance(potential_list, list) else potential_list