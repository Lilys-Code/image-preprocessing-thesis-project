import os
import sys

# add src folder to python path for tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# ROOT now equals the 'src' directory; that's what we want on sys.path
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
