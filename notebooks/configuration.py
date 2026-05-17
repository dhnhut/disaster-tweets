from IPython import get_ipython
import sys
import os

ipython = get_ipython()
# avoid  'ZMQInteractiveShell' object has no attribute 'magic'
if ipython is not None and hasattr(ipython, "magic"):
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
sys.path.append(os.path.abspath(os.path.join("..")))
