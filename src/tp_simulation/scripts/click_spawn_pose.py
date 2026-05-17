#!/usr/bin/env python3

import cv2
import yaml
import matplotlib.pyplot as plt
from pathlib import Path


map_path = Path("/home/pedro/Projects/motion_planning/src/tp_simulation/maps/maze.pgm")
yaml_path = Path("/home/pedro/Projects/motion_planning/src/tp_simulation/maps/maze.yaml")

img = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise RuntimeError(f"Could not read map: {map_path}")

with open(yaml_path, "r") as f:
    info = yaml.safe_load(f)

resolution = float(info["resolution"])
origin_x = float(info["origin"][0])
origin_y = float(info["origin"][1])

height, width = img.shape

plt.figure(figsize=(8, 8))
plt.imshow(img, cmap="gray", origin="upper")
plt.title("Click on the desired TurtleBot3 spawn position")
plt.xlabel("image col")
plt.ylabel("image row")
plt.grid()

points = plt.ginput(1)
plt.close()

if not points:
    raise RuntimeError("No point selected.")

col_float, row_float = points[0]

col = int(round(col_float))
row = int(round(row_float))

if row < 0 or row >= height or col < 0 or col >= width:
    raise RuntimeError("Selected point is outside the map.")

pixel_value = int(img[row, col])

x = origin_x + (col + 0.5) * resolution
y = origin_y + (height - row - 0.5) * resolution

print()
print("Selected image cell:")
print(f"  row = {row}")
print(f"  col = {col}")
print(f"  pixel value = {pixel_value}")

if pixel_value < 127:
    print()
    print("WARNING: selected point is probably occupied/black.")
    print("Choose a white/free cell for spawning.")

print()
print("World/map coordinates:")
print(f"  x = {x:.3f}")
print(f"  y = {y:.3f}")

print()
print("Launch command:")
print(
    "ros2 launch tp_simulation maze_tb3.launch.py "
    f"x_pose:={x:.3f} y_pose:={y:.3f}"
)
print()
