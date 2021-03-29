import os
import pathlib
import sys

parent = str(pathlib.Path(__file__).absolute().parent.parent)
sys.path.insert(0, parent)

import src
