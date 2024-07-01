import cv2
import numpy as np

def maskCircle(img):
    height, width = img.shape[:2]
    # Create a black mask img of the same size as the original img
    mask = np.zeros((height, width), dtype=np.uint8)

    # Calculate the radius of the circle (touching all sides of the square)
    radius = int(min(height, width) // 2 * 0.98)

    # Calculate the center coordinates of the circle
    center = (width // 2, height // 2)

    # Draw the white circle (255) on the black mask
    cv2.circle(mask, center, radius, 255, thickness=cv2.FILLED)

    # Apply the mask to the original img
    result = cv2.bitwise_and(img, img, mask=mask)

    return result





# Load image
image = cv2.imread('src/samples/sampleImg.jpg')
imgH, imgW = image.shape[:2]
image = cv2.resize(image, (imgW//12, imgH//12))

blackimg = cv2.imread('src/samples/blackimg.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Mask circle
maskedCircle = maskCircle(gray)

# # Blurred image
# blurredImg = cv2.GaussianBlur(maskedCircle, (5, 5), 0)

# # Detect edges using Canny edge detector
# edges = cv2.Canny(blurredImg, 10, 200)

# # Threshold to create binary image
# _, binary = cv2.threshold(blurredImg, 100, 255, cv2.THRESH_BINARY)

# Find contours of particles within the circular region
contours, _ = cv2.findContours(maskedCircle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# # Filter contours based on area (particle size)
# min_area = 0  # adjust as needed
# filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
#
# # Function to check if contour resembles a line (based on aspect ratio)
# def is_line(cnt):
#     x, y, w, h = cv2.boundingRect(cnt)
#     aspect_ratio = w / float(h)
#     # Adjust aspect ratio threshold based on your filter paper's lines
#     return 5 < aspect_ratio < 50
#
# # Filter out lines
# particles = [cnt for cnt in filtered_contours if not is_line(cnt)]
#
# # Count particles
# particle_count = len(particles)
#
# print(f"Number of particles detected: {particle_count}")

# Display the results (optional)

cv2.drawContours(maskedCircle, contours, -1, (0, 0, 255), 5)
# height, width = blackimg.shape[:2]
# blackimg = cv2.resize(blackimg, (width // 5, height // 5))
cv2.imshow('Particles', maskedCircle)
cv2.waitKey(0)
cv2.destroyAllWindows()
