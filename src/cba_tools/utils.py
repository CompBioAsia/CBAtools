# utils.py
from cba_tools.tasks import which

# Source - https://stackoverflow.com/a/27674608
def _exception_handler(exception_type, exception, traceback):
    print(f"{exception_type.__name__}: {exception}")

def suppress_tracebacks():
    import sys
    sys.excepthook = _exception_handler

def installed(command):
    result = which(command)
    return result.returncode == 0

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'