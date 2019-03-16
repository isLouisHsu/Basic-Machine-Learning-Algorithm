import cv2
import numpy as np

from HOG import HOG

def main():

    image = cv2.imread('./joker.jpg')
    # cv2.imshow("", image); cv2.waitKey(0)

    hog = HOG()
    hog.fit(image)

if __name__ == "__main__":
    main()