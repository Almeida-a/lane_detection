import cv2
import numpy as np
import utils


# References
# https://www.learnpython.org/en/Multiple_Function_Arguments
# https://medium.com/@yogeshojha/self-driving-cars-beginners-guide-to-computer-vision-finding-simple-lane-lines-using-python-a4977015e232


def masking(image, polygon):
    """

    :param image: Must be greyscaled
    :param polygon: list of points
    :return: masked image
    """
    # Masking region of interest
    height, width = image.shape[0:2]
    for i, point in enumerate(polygon):
        # 1 - y because image matrix is processed from top to down
        polygon[i] = point[0] * width, (1 - point[1]) * height

    polygon = np.array(polygon, np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [polygon], 255)
    cropped_image = cv2.bitwise_and(image, mask)

    return cropped_image


def houghP(original_img, cropped_img):
    # Hough's transform
    rho = 2
    theta = np.pi / 180
    threshold = 100  # Minimum votes to be recognized as a line
    lines = cv2.HoughLinesP(cropped_img, rho, theta, threshold, np.array([]), minLineLength=40, maxLineGap=5)

    # Mid code:
    slopes = []

    line_image = np.zeros_like(original_img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            if x2 != x1:
                slopes.append((y2-y1) / (x2-x1))
            else:
                # Big number
                slopes.append(100_000)
    else:
        raise AssertionError("lines == None")

    av_slope = np.average(slopes)

    return cv2.addWeighted(original_img, 0.8, line_image, 1, 1), av_slope


def lane_detector(img_path, **parameters):
    """
    Status: working, but needs better masking
    :return: Image with lines drawn over lane separators

    :keyword Arguments:
        * *blur_level* (``int``) --
            level of blurring: 3, 5 or 7 TODO confirm this
        * *blur_funct* (``funct``) --
            choose from available opencv blur functions
        * *thresholds* (``tuple``) --
            format (threshold1, threshold2)
            TODO explain
        * *triangle* (``array``)
            format [point1, point2, point3]
            Since in most cases the roads shape like a triangle,
                our mask will be this triangle,
                defined by fraction values.
                E.g.: (.5, .5) means the point at the center of the image
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

    if "blur_funct" not in parameters.keys():
        blur_funct = cv2.GaussianBlur
    else:
        blur_funct = parameters['blur_funct']

    # Canny edge detection
    thresholds = (50, 150)  # Default value TODO evaluate this
    if "thresholds" in parameters:
        assert all([0 <= elem < 256 for elem in parameters['thresholds']]),\
            "Invalid threshold values!"
        thresholds = parameters['thresholds']

    # Masking
    triangle = [(0, .2), (.5, .99), (.99, .2), ]
    if "triangle" in parameters:
        assert len(triangle) == 3 and\
               all([len(tupl) == 2 for tupl in triangle]), "Bad Shape"
        triangle = parameters["triangle"]

    # ---- Pipeline stages of image processing ----
    # Loading the image
    lane_image = cv2.imread(utils.get_abs_path(img_path))

    # Converting into grayscale
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

    # Reduce Noise and Smoothen Image
    blur = blur_funct(
        src=gray, ksize=blur, sigmaX=0
    )

    # Edge detection (canny)
    canny_image = cv2.Canny(
        image=blur, threshold1=thresholds[0], threshold2=thresholds[1]
    )

    # Mask region of interest
    cropped_image = masking(canny_image, triangle)

    # Hough Transform
    combo_image, slope = houghP(original_img=lane_image, cropped_img=cropped_image)

    # Direction values
    angle_rad = np.arctan(slope)
    angle_deg = np.degrees(angle_rad)
    print(f"Slope: {slope}", f"Angle (rad): {round(angle_rad, 2)}*pi",
          f"Angle (degrees) {round(float(angle_deg), 2)}º", sep="\n", end=".")
    utils.show_image(image=combo_image, title=img_path.split('/')[-1])


if __name__ == '__main__':

    im_path = "road_photos/road2.jpeg"
    title = im_path.split("/")[-1]

    lane_detector(im_path,
                  blur_level=5,
                  blur_funct=cv2.GaussianBlur,
                  thresholds=(50, 150),
                  triangle=[(0, 0), (.5, .66), (.99, 0)]
                  )
