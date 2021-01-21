import cv2
import numpy as np
import utils


def lane_detector(img_path):
    """
    As in https://medium.com/@yogeshojha/self-driving-cars-beginners-guide-to-computer-vision-finding-simple-lane-lines-using-python-a4977015e232
    Status: working, but needs better masking
    :return:
    """
    # Loading the image
    lane_image = cv2.imread(utils.get_abs_path(img_path))

    # Converting into grayscale
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

    # Reduce Noise and Smoothen Image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection (canny)
    canny_image = cv2.Canny(blur, 50, 150)

    # Masking region of interest
    height = lane_image.shape[0]
    triangle = np.array([[(200, height), (550, 250), (1100, height), ]], np.int32)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, triangle, 255)
    cropped_image = cv2.bitwise_and(canny_image, mask)

    # Hough transform
    rho = 2
    theta = np.pi / 180
    threshold = 100
    lines = cv2.HoughLinesP(cropped_image, rho, theta, threshold, np.array([]), minLineLength=40, maxLineGap=5)

    line_image = np.zeros_like(lane_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    else:
        raise AssertionError("lines == None")
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    utils.show_image(image=combo_image)


if __name__ == '__main__':
    lane_detector("road_photos/road1.jpeg")
