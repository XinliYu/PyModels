from colorama import Fore
from util.type_ext import *
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


def error_print(error_place, message):
    if is_class(error_place):
        error_place_str = error_place.__module__ + '.' + error_place.__name__
    elif is_basic_type(error_place):
        error_place_str = str(error_place)
    else:
        error_place_str = error_place.__class__

    color_print("ERROR ({}):".format(error_place_str), message, title_color=Fore.RED, content_color=Fore.MAGENTA)
