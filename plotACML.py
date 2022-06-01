import os
from PIL import Image
import numpy as np

sea = np.array(Image.open(f"data{os.sep}sea.png"))/255