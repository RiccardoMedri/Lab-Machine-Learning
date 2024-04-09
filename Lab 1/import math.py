import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid 

colo_img = cv2.imread('./imgs/pig/color.png', cv2.IMREAD_COLOR)    # carico l'immagine colore.

print(colo_img)
