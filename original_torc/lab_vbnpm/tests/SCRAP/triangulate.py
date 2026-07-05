import cv2
import time
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

img = cv2.imread('../images/img0/color.jpg')
seg = np.load('../images/img0/mask.npy')

mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
mask = (seg == 0).astype(np.uint8)

contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
)

points = np.vstack(contours).squeeze()
if points.ndim == 1:  # Handle single point contour case
    points = points[np.newaxis, :]

# Apply Delaunay triangulation
tri = Delaunay(points)


# Point-in-polygon check using OpenCV's pointPolygonTest
def is_inside_mask(point, contours):
    for contour in contours:
        # The result of pointPolygonTest will be positive if the point is inside the contour
        if cv2.pointPolygonTest(contour, 1.0*point, False) >= 0:
            return True
    return False


# Prune the triangles: Remove edges that have vertices outside the mask
valid_simplices = []
for simplex in tri.simplices:
    # Check all edges of the triangle
    valid = True
    for i in range(3):
        p1, p2 = simplex[i], simplex[(i + 1) % 3]
        print(points[p1])
        print(points[p2])
        if not (is_inside_mask(points[p1], contours)
                and is_inside_mask(points[p2], contours)):
            valid = False
            break
    if valid:
        valid_simplices.append(simplex)

valid_simplices = np.array(valid_simplices)

# Convert the image to RGB for Matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot the filtered triangulation over the image
plt.figure(figsize=(10, 8))
plt.imshow(img)
for simplex in valid_simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'cyan', linewidth=1)
plt.plot(points[:, 0], points[:, 1], 'ro', markersize=2)
plt.title('Filtered Triangulation Over Segmented Region')
plt.axis('off')
plt.show()

# cv2.imshow('Image', img)
# cv2.waitKey(0)
input('Press Enter to close the window...')
