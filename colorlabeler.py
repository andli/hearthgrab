# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2


class ColorLabeler:
    def __init__(self):
        # initialize the colors dictionary, containing the color
        # name as the key and the RGB tuple as the value
        colors = OrderedDict({
            "blank": (219, 198, 144),
            "legendary": (170, 122, 45),
            "epic": (136, 60, 160),
            "uncommon": (38, 81, 143),
            "common": (127, 139, 152)})

        # allocate memory for the L*a*b* image, then initialize
        # the color names list
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []

        # loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)

        # convert the L*a*b* array from the RGB color space
        # to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def label(self, image):
        # construct a mask for the contour, then compute the
        # average L*a*b* value for the masked region
        # mask = np.zeros(image.shape[:2], dtype="uint8")
        # cv2.drawContours(mask, [c], -1, 255, -1)
        # mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image)[:3]

        # initialize the minimum distance found thus far
        mindist = (np.inf, None)

        # loop over the known L*a*b* color values
        for (i, row) in enumerate(self.lab):
            # compute the distance between the current L*a*b*
            # color value and the mean of the image
            # d = dist.euclidean(row[0], mean)
            d = np.linalg.norm(row[0] - mean)

            # if the distance is smaller than the current distance,
            # then update the bookkeeping variable
            if d < mindist[0]:
                mindist = (d, i)

        # return the name of the color with the smallest distance
        if mindist[0] > 40:
            return "error"
        print mindist[0]
        return self.colorNames[mindist[1]]
