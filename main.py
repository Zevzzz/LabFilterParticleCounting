# Designed and Written by Donson Xie
# Advised by Andrew Oldag

import cv2
import numpy as np
import time
from os import mkdir
from datetime import datetime
from math import floor, ceil


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

def getCoolVisualImg(componentsImg):
    # Map component labels to hue value
    label_hue = np.uint8(179 * componentsImg / np.max(componentsImg))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Convert to BGR format for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # Set background label to black
    labeled_img[label_hue == 0] = 0

    return labeled_img

def getComponents(img):

    # Mask circle
    maskedImg = maskCircle(img)

    # Grayscale
    grayImg = cv2.cvtColor(maskedImg, cv2.COLOR_BGR2GRAY)

    # Binary threshold
    binaryImg = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 12)

    # Invert image
    binaryImg = cv2.bitwise_not(binaryImg)

    # # Optional: Dilate the image to connect disjointed parts of contours
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # dilated = cv2.dilate(binaryImg, kernel, iterations=1)

    # Find connected components
    numLabels, componentsImg = cv2.connectedComponents(binaryImg)

    # Get ratios
    ratios = []
    areas = []
    for component in range(1, numLabels):
        compMask = (componentsImg == component).astype(np.uint8) * 255
        contours, _ = cv2.findContours(compMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        ratio = max(w, h) / min(w, h)
        area = w * h

        imgArea = img.shape[0] * img.shape[1]
        areaBelowTenthPerc = area < 0.001 * imgArea
        areaAboveTwoTenThouPerc = area > 0.000002 * imgArea
        # areaBelowHalfPerc = True
        ratioBelow5 = ratio < 4
        # ratioBelow5 = True

        if areaBelowTenthPerc and areaAboveTwoTenThouPerc and ratioBelow5:
            ratios.append((component, ratio, w, h))
            areas.append((component, area))

    # resultImg = cv2.drawContours(resultImg, contours, -1, (255, 0, 0), 1)
    # print(len(contours))

    # for contour in contours:
    #     imgTemp = resultImg.copy()
    #     cv2.imshow('Particles', cv2.drawContours(imgTemp, contour, -1, (0, 0, 255), 5))

    # Draw connected components

    return img, len(ratios), ratios, componentsImg

def getDatetime():
    now = datetime.now()
    dtString = now.strftime("%m-%d-%Y_%H-%M-%S")
    return dtString






if __name__ == '__main__':
    outputSnapshotMargin = 150

    # # Show cool visual :)
    # coolImg = getCoolVisualImg(componentsImg)
    #
    # cv2.imshow('Final Result', img)
    # cv2.imshow('Cool Img', coolImg)
    # cv2.imshow('Binary Img', binaryImg)
    # cv2.imshow('Masked Original Img', maskedImg)

    # Load image
    imgRaw = cv2.imread('src/samples/sampleImg.jpg')
    imgH, imgW = imgRaw.shape[:2]
    imgResized = cv2.resize(imgRaw, (3000, 3000))

    startTime = time.time()
    print('Extracting Particle Count...')

    componentsRet, particleCount, particleRatios, compImg = getComponents(imgResized)

    copiedRawImg = imgRaw.copy()
    previewImg = cv2.resize(imgRaw, (1000, 1000))

    particleImgs = []
    for label, ratio, width, height in particleRatios:
        x, y, w, h = cv2.boundingRect((compImg == label).astype(np.uint8) * 255)

        previewImg = cv2.rectangle(previewImg, ((x//3 - 10), (y//3 - 10)), ((x + w)//3 + 10, (y + h)//3 + 10), (0, 0, 255), 2)
        previewImg = cv2.putText(previewImg, str(label), ((x//3), (y//3) - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 225), 2)

        startX = int(floor((x / compImg.shape[1]) * imgRaw.shape[1]))
        endX = int(ceil(((x + w) / compImg.shape[1]) * imgRaw.shape[1]))
        startY = int(floor((y / compImg.shape[0]) * imgRaw.shape[0]))
        endY = int(ceil(((y + h) / compImg.shape[0]) * imgRaw.shape[0]))

        canStartXMargin = startX - outputSnapshotMargin >= 0
        canEndXMargin = endX + outputSnapshotMargin <= imgRaw.shape[1]
        canStartYMargin = startY - outputSnapshotMargin >= 0
        canEndYMargin = endY + outputSnapshotMargin <= imgRaw.shape[0]

        if canStartXMargin and canEndXMargin and canStartYMargin and canEndYMargin:
            cv2.rectangle(copiedRawImg, (startX - 20, startY - 20), (endX + 20, endY + 20), (0, 0, 255), 3)

            startX -= outputSnapshotMargin
            endX += outputSnapshotMargin
            startY -= outputSnapshotMargin
            endY += outputSnapshotMargin


        particleImgs.append(copiedRawImg[startY:endY, startX:endX])



    # Output folter
    outputFolder = 'outputs/' + getDatetime()
    mkdir(outputFolder)
    mkdir(outputFolder + '/images')

    # Write data
    outputData = [f'Particle Count: {particleCount}\n\n', 'Particle Label, Width, Height\n']
    for label, ratio, width, height in particleRatios:
        outputData.append(f'{label} {width} {height}\n')

    with open(outputFolder + '/Output.txt', 'a') as file:
        file.writelines(outputData)

    # Write images
    for i in range(len(particleImgs)):
        label, _, __, ___ = particleRatios[i]
        # print(outputFolder + '/images/' + str(label))
        snapshotImg = particleImgs[i]
        # cv2.rectangle(snapshotImg, (startX, startY), (endX, endY), (255, 0, 255), 100)
        cv2.imwrite(outputFolder + '/images/' + str(label) + '.jpg', snapshotImg)


    processingTimeSec = round(time.time() - startTime, 3)
    print(f'Particle Count Extracted, processing time: {processingTimeSec}s')
    print(f'Particle Count: {particleCount}')
    print(f'Data saved to {outputFolder}')

    cv2.imshow('Final Result A', previewImg)

    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()

