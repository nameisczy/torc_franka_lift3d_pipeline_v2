import cv2
import numpy as np

m0 = np.load('../images/img_ss_30/mask.npy')
m1 = np.load('../experiments/runs/trial_2025-09-17_13-13-43__unstructured__vlm_dg/30__obj_000049_0__vlm_dg/mask_1.npy')

for i in range(32):
    mask0 = ((1 << i) & m0).astype(bool)
    mask1 = ((1 << i) & m1).astype(bool)
    color = np.zeros((*mask0.shape, 3), dtype=np.uint8)
    color[mask0] = [0, 255, 0]
    color[mask1] = [0, 0, 255]
    cv2.imshow(f'mask {i}', color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
