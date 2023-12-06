import os
import cv2
import imutils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class ObjectTracker():
    def __init__(self, greenUpper, greenLower, folder):
        self.greenUpper = greenUpper
        self.greenLower = greenLower
        self.folder = folder
    
    def get_mask(self, img):
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, self.greenLower, self.greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    def get_center(self, mask):
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            return (x, y), radius
        
        else:
            return (None, None), None


    def obj_tracker(self):
        location = []
        for filename in os.listdir(self.folder):
            if filename.endswith(".jpg"):
                img = cv2.imread(os.path.join(self.folder, filename))
                mask = self.get_mask(img)
                (x, y), radius = self.get_center(mask)
                # draw the circle and centroid on the frame
                # circle1 = plt.Circle((x, y), radius, color='r', fill=False)
                # fig, ax = plt.subplots()
                # ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # ax.add_artist(circle1)
                # plt.show()
                location.append(
                    {
                        'filename': filename,
                        'x': x,
                        'y': y,
                    }
                )
                
        df = pd.DataFrame(location)
        file_names = df['filename'].values
        frame_num = [f.split(' ')[1].split('.')[0] for f in file_names]
        frame_num = np.array(frame_num, dtype=int)
        df['frame_num'] = frame_num
        camera = [c.split('_')[2] for c in file_names]
        df['camera'] = camera

        return df