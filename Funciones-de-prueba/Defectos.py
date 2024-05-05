import cv2
import numpy as np

def calculate_difference(img1, img2):
    diff = cv2.absdiff(img1, img2)
    _, diff = cv2.threshold(diff, 11, 255, cv2.THRESH_BINARY)
    return diff

def denoise_binary_image(binary_image, kernel_size=3):
    # Apply erosion to remove small white regions and protrusions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)

    # Apply dilation to add small white regions and broaden protrusions
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

    return dilated_image

def process_contours(img1, reference, ksize=3):
    # Convert input images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Calculate the difference between the images
    diff = calculate_difference(img1_gray, reference_gray)

    # Perform morphological operations on the binary difference image
    closed = denoise_binary_image(diff)
    dilated = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)))

    # Find contours on the dilated image of the current image
    contours_current, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours on the dilated image of the reference image
    diff_reference = calculate_difference(reference_gray, reference_gray)
    closed_reference = denoise_binary_image(diff_reference)
    dilated_reference = cv2.dilate(closed_reference, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)))
    contours_reference, _ = cv2.findContours(dilated_reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tam_max_defecto = 5

    # Process contours found in the current image
    for c in contours_current:
        x, y, w, h = cv2.boundingRect(c)

        # Check if the contour is not present in the reference image
        contour_present = False
        for c_ref in contours_reference:
            area_diff = cv2.contourArea(cv2.convexHull(c)) - cv2.contourArea(cv2.convexHull(c_ref))
            if abs(area_diff) < tam_max_defecto:
                contour_present = True
                break

        # Draw rectangle if contour is not present in the reference image
        if not contour_present:
            cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 1)

    return img1

img1 = cv2.imread('Error.png')
reference = cv2.imread('Base.png')
result = process_contours(img1, reference)
cv2.imwrite('result.png', result)