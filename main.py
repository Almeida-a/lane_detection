import cv2
import numpy as np
import utils

# References
# https://www.learnpython.org/en/Multiple_Function_Arguments


def lane_detector(img_path, **parameters):
    """
    As in https://medium.com/@yogeshojha/self-driving-cars-beginners-guide-to-computer-vision-finding-simple-lane-lines-using-python-a4977015e232
    Status: working, but needs better masking
    :return: Image with lines drawn over lane separators

    :keyword Arguments:
        * *blur_level* (``int``) --
            level of blurring: 3, 5 or 7 TODO confirm this
        * *thresholds* (``tuple``) --
            format (threshold1, threshold2)
            TODO explain
        * ...
        ...
    """
    # ---- Parameters: get in kwargs if it's there, if not set as default ----

    # Blurring
    if "blur_level" not in parameters.keys() or \
            parameters['blur_level'] not in [3, 5, 7]:
        blur_level = 5
    else:
        blur_level = parameters['blur_level']
    blur = (blur_level, blur_level)

    # Canny edge detection
    thresholds = (50, 150)  # Default value TODO evaluate this
    if "thresholds" in parameters:
        assert all([0 <= elem < 256 for elem in parameters['thresholds']]),\
            "Invalid threshold values!"
        thresholds = parameters['parameters']

    # Masking
    ...

    # ---- Pipeline stages of image processing ----
    # Loading the image
    lane_image = cv2.imread(utils.get_abs_path(img_path))

    # Converting into grayscale
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

    # Reduce Noise and Smoothen Image
    blur = cv2.GaussianBlur(
        src=gray, ksize=blur, sigmaX=0
    )

    # Edge detection (canny)
    canny_image = cv2.Canny(
        image=blur, threshold1=thresholds[0], threshold2=thresholds[1]
    )

    # Masking region of interest
    height, width = lane_image.shape[0:2]
    triangle = np.array(
        [[(200, height), (550, 250), (1100, height), ]],
        np.int32
    )
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, triangle, 255)
    cropped_image = cv2.bitwise_and(canny_image, mask)

    # Hough's transform
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
    utils.show_image(image=combo_image, title=img_path.split('/')[-1])


if __name__ == '__main__':

    im_path = "road_photos/road2.jpeg"
    title = im_path.split("/")[-1]

    lane_detector(im_path)
