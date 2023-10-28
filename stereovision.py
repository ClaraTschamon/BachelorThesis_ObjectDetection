import pyrealsense2 as rs
import numpy as np
import cv2
import sys

pipeline = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30) #848, 480
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

print("[INFO] Starting streaming…")
pipeline.start(cfg)
print("[INFO] Camera ready.")

frameset = pipeline.wait_for_frames()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()

color = np.asanyarray(color_frame.get_data())
cv2.imshow('rgb', color)
cv2.waitKey(0)
cv2.destroyAllWindows()

colorizer = rs.colorizer()
colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
cv2.imshow('colorized depth', colorized_depth)
cv2.waitKey(0)
cv2.destroyAllWindows()

#depth_image = np.asanyarray(depth_frame.get_data())
#depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)


#cv2.imshow('depth', depth_cm)

