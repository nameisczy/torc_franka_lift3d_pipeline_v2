import os
import sys
import time

import cv2
import numpy as np
from task_planner.prompt import Prompter
from task_planner.prompt import create_labeled_image

img_dir = sys.argv[1]

img = cv2.imread(os.path.join(img_dir, 'color.jpg'))
seg = np.load(os.path.join(img_dir, 'mask.npy'))

# p = Prompter(url=None)
# labeled = p.create_labeled_image(img, seg)
labeled, visible = create_labeled_image(img, seg)
# cv2.imwrite(os.path.join(img_dir, 'labeled.jpg'), labeled)
cv2.imshow('labeled', labeled)
try:
    while True:
        cv2.pollKey()
except:
    pass
