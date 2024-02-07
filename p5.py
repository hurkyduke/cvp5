import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('/Users/venkateshsanwal/Desktop/ml/computer_vision/Pi7_Tool_100-VENKETESHPHOTO-hadbomb_com.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Unable to load the image.")
else:
    # Define a kernel for erosion and dilation
    kernel = np.ones((5, 5), np.uint8)

    # Perform erosion
    erosion_result = cv2.erode(image, kernel, iterations=1)

    # Perform dilation
    dilation_result = cv2.dilate(image, kernel, iterations=1)

    # Display the original, eroded, and dilated images
    plt.figure(figsize=(10, 5))

    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(132)
    plt.imshow(erosion_result, cmap='gray')
    plt.title('Erosion')

    plt.subplot(133)
    plt.imshow(dilation_result, cmap='gray')
    plt.title('Dilation')

    plt.show()
