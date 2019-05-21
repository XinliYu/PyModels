from colorama import Fore
import sys


class flogger(object):
    def __init__(self, path, print_terminal=True):
        self.terminal = sys.stdout
        self.log = open(path, "w")
        self.print_to_terminal = print_terminal

    def write(self, message):
        if self.print_to_terminal:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
        pass

    def reset(self):
        self.flush()
        self.log.close()
        sys.stdout = self.terminal


def color_print(title, content, title_color=Fore.CYAN, content_color=Fore.WHITE):
    print('{}{}: {}{}'.format(title_color, title, content_color, content))


def highlight_print(title, content):
    color_print(title, content, title_color=Fore.YELLOW)


def error_print(error_tag, message):
    if is_class(error_tag):
        error_place_str = error_tag.__module__ + '.' + error_tag.__name__
    elif is_basic_type(error_tag):
        error_place_str = str(error_tag)
    else:
        error_place_str = error_tag.__class__

    color_print("ERROR ({}):".format(error_place_str), message, title_color=Fore.RED, content_color=Fore.MAGENTA)


def kv_list_format(keys, values, kv_delimiter=':', pair_delimiter: str = ', ', value_format='{}', value_transform=None, value_idx=-1):
    return pair_delimiter.join([keys[i] + kv_delimiter +
                                ('{}' if value_format is None else (value_format if type(value_format) is str else value_format[i]))
                               .format((values[i][value_idx] if value_transform is None else value_transform(values[i][value_idx])) if type(values[i]) in (list, tuple)
                                       else (values[i]) if value_transform is None else value_transform(values[i])) for i in range(len(keys))])


def kv_tuple_format(kv_tuples, kv_delimiter, pair_delimiter, value_idx=-1):
    return pair_delimiter.join([tup[0] + kv_delimiter +
                                ('{}' if len(tup) == 2 else tup[2]).format((tup[1][value_idx] if len(tup) < 4 else tup[3](tup[1][value_idx])) if type(tup[1]) in (list, tuple)
                                                                           else tup[1] if len(tup) < 4 else tup[3](tup[1]))
                                for tup in kv_tuples])


def is_class(variable):
    return isinstance(variable, type)


def is_basic_type(variable):
    return type(variable) in (int, float, str)


def is_str(variable):
    return type(variable) is str


def take_element_if_list(potential_list, i: int):
    return potential_list[i] if isinstance(potential_list, list) else potential_list


def str2bool(s: str):
    return s in ('true', 'True', 'yes', 'Yes', 'y', 'Y', 'OK', 'ok', '1')


def list2d(row_count: int, col_count: int, init_val=None):
    return [[init_val] * col_count for _ in range(row_count)]


def exec_method(obj, callable_names, arguments, error_tag: str = None, error_msg: str = None):
    """
    Checks if the `obj` has any callable member of a name specified in `method_names`. The first found method will be executed.
    :param obj: the object whose callable member of a name first found in `method_names` will be executed.
    :param callable_names: the callable names to check.
    :param arguments: the arguments for the callable.
    :param error_tag: the error tag to display if none of `callable_names` is found as a callable member of `obj`.
    :param error_msg: the error message to display.
    :return: whatever the found callable returns.
    """
    for method_name in callable_names:
        op = getattr(obj, method_name, None)
        if callable(op):
            return op(*arguments)
    error_print(error_tag if error_tag is not None else exec_method.__name__,
                error_msg if error_msg is not None else "no method '" + "' or '".join(callable_names) + "' is found for type '" + type(obj).__name__ + "'")
