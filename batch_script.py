import os
import sys

try:
    os.system(f"python main.py -m DenseNet121 -f {sys.argv[1]}")
except:
    pass

try:
    os.system(f"python main.py -m DenseNet121 {sys.argv[1]}")
except:
    pass

try:
    os.system(f"python main.py -m VGG16 -f {sys.argv[1]}")
except:
    pass

try:
    os.system(f"python main.py -m VGG16 {sys.argv[1]}")
except:
    pass

try:
    os.system(f"python main.py -m VGG-based {sys.argv[1]}")
except:
    pass
