"""Print the HSV value of a pixel and a suggested range.

    python -m acva.hsv screenshot.png 540 1200
"""
import sys

import cv2

if __name__ == "__main__":
    path, x, y = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"cannot read {path}")
    h, s, v = (int(c) for c in cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[y, x])
    print(f"BGR={tuple(int(c) for c in img[y, x])}  HSV=({h}, {s}, {v})")
    print(f"suggested range: (({max(0, h - 10)}, {max(0, s - 60)}, {max(0, v - 60)}), ({min(179, h + 10)}, 255, 255))")
