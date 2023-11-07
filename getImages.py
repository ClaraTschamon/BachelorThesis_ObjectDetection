import cv2
# used for camera calibration
cap = cv2.VideoCapture(1)

num = 0

while cap.isOpened():

    succes, img = cap.read()
    # Get the frame size (height and width)
    frame_height, frame_width, _ = img.shape

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('./calibrationImages/img' + str(num) + '.png', img)
        num += 1
    print(f"Frame size: {frame_width}x{frame_height}")

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()