
# coding: utf-8

# In[4]:


import cv2

class Crop:

    def __init__(self, img):
        self.cv2img = img


    def do_crop(self):
        """
        method for cropping image on coordinates returned by 'detect_edge' module
        :return: cropped image
        """

        image = self.cv2img

        # blur the image slightly, then convert it to grayscale and
        # consecutively to the L*a*b* color spaces for better thresholding
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        (t, binary) = cv2.threshold(lab, 20, 255, cv2.THRESH_BINARY_INV)
        # get coordinates of black edge
        min_x, max_x, min_y, max_y = self.detect_edge(binary)
        # crop the image with coordinates
        cropped = image[min_y:max_y,min_x:max_x]

        # return cropped image
        return cropped


    def detect_edge(self, binary):
        """
        Retrieve coordinates of the black edges by finding the nearest
        non-black pixel to the edge of x-axis and y-axis
        :param binary: image resulting from thresholding algorithm (b&w)
        :return: coordinates of: minimal and maximal x-axis, minimal and maximal y-axis
        """
        # convert image to grayscale for single channel image
        gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

        # retrieving x-axis coordinates
        min_x = []
        max_x = []
        for frac in [0.5, 0.55, 0.6, 0.65, 0.7]:        
            # compute x-axis at 100*frac% of the y-axis
            # above this area, text may be present on the black edge
            mid = int(gray.shape[0] * frac)
            # get a list of pixels on the x-axis at 100*frac% of y-axis
            x_axis = list(gray[mid, ::])
            x_axis_len = len(x_axis)
            # get minimal index of x_axis of black pixel
            min_x.append(x_axis.index(0))
            # get maxiumum value of x_axis of black pixel
            x_axis_rev = x_axis[::-1]
            max_x.append(x_axis_len - x_axis_rev.index(0))
        
        min_x = max(min_x)##trick
        max_x = max(max_x)

        # retrieving y-axis coordinates
        min_y = []
        max_y = []
        for frac in [0.4, 0.45, 0.5, 0.55, 0.6]:
            # compute x-axis at 100*frac% of the image
            mid = int((gray.shape[1] + min_x) * frac)
            # get an y-axis as list at 100*frac% of the x-axis
            y_axis = list(gray[::, mid])
            y_axis_len = len(y_axis)
            # get minimal y index
            min_y.append(y_axis.index(0))
            # get maximal y index
            y_axis_rev = y_axis[::-1]
            max_y.append(y_axis_len - y_axis_rev.index(0))
        
        min_y = min(min_y)
        max_y = max(max_y)

        return min_x, max_x, min_y, max_y

