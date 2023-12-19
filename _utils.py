import os
import cv2
import imutils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from numpy.polynomial import Polynomial
from scipy import ndimage
import math
import sympy

class ObjectTracker():
    def __init__(self, greenUpper, greenLower, folder):
        self.greenUpper = greenUpper
        self.greenLower = greenLower
        self.folder = folder
        self.rotL = None
        self.rotR = None
    
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
    
    def set_angle(self):
        file_names = os.listdir(self.folder)
        for c in ['_L_', '_R_']:
            mean_angle = []
            file_names_filt = [i for i in file_names if c in i]
            for filename in file_names_filt:
                if filename.endswith(".jpg"):
                    img = cv2.imread(os.path.join(self.folder, filename))
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
                    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
                    
                    angles = []
                    for [[x1, y1, x2, y2]] in lines:
                        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                        angles.append(angle)
                    
                    median_angle = np.median(angles)
                    if median_angle <= -45:
                        median_angle+=90
                    elif median_angle >= 45:
                        median_angle -= 90

                    # print(median_angle)
                    mean_angle.append(median_angle)
                    # img_rotated = ndimage.rotate(img, median_angle)
                    # img_rotated = cv2.resize(img_rotated, (640,480))
                    # plt.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
                    # plt.show()
            if c == '_L_':
                self.rotL = np.mean(mean_angle)
            else:
                self.rotR = np.mean(mean_angle)


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

                if self.rotL is not None:
                    if '_L_' in filename:
                        img = imutils.rotate(img, angle=self.rotL)
                
                if self.rotR is not None:
                    if '_R_' in filename:
                        img = imutils.rotate(img, angle=self.rotR)

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


# curve fitting model
# define a function for fitting
class Parabolic3D():
    def __init__(self, x2=0, x1=0, x0=0, y1=0, y0=0, z1=0, z0=0):
        self.x2 = x2; self.x1 = x1; self.x0 = x0
        self.y1 = y1; self.y0 = y0
        self.z1 = z1; self.z0 = z0
    
    def free_fall_with_res(self, t, x2, x1, x0):
        return x2*t - x1*np.exp((t/x1)) + x0
    
    def func(self, t, x2, x1, x0, y1, y0, z1, z0):
        # we have an g-force on x axis
        Px=Polynomial([x2, x1, x0]) # 2 order
        Py=Polynomial([y1, y0])
        Pz=Polynomial([z1, z0])
        return np.concatenate([Px(t), Py(t), Pz(t)])

    def fit(self, tracked):
        t = np.arange(len(tracked))
        xyz = np.concatenate([tracked[:,0], tracked[:,1], tracked[:,2]])
        weights, _ = curve_fit(self.func, t, xyz)
        self.x2 = weights[0]
        self.x1 = weights[1]
        self.x0 = weights[2]
        self.y1 = weights[3]
        self.y0 = weights[4]
        self.z1 = weights[5]
        self.z0 = weights[6]
        
    def predict(self, t):
        return self.func(t, self.x2, self.x1, self.x0, self.y1, self.y0, self.z1, self.z0).reshape(3, -1)
    
    def calc_dist_error(self, tracked):
        err = []
        t = np.arange(len(tracked))
        p_hat = self.predict(t).T
        p = tracked
        # print(p_hat, p)
        dist = np.sqrt(np.sum(np.power((p-p_hat), 2), axis=1))

        return np.mean(dist)

    def get_params(self):
        return self.x2, self.x1, self.x0, self.y1, self.y0, self.z1, self.z0

# curve fitting model
# define a function for fitting
class Exp3D():
    def __init__(self, x2=0, x1=0, x0=0, y1=0, y0=0, z1=0, z0=0):
        self.x2 = x2; self.x1 = x1; self.x0 = x0
        self.y1 = y1; self.y0 = y0
        self.z1 = z1; self.z0 = z0
    
    def free_fall_with_res(self, t, x2, x1, x0):
        return x2*t - x1*np.exp((t/x1)) + x0
    
    def func(self, t, x2, x1, x0, y1, y0, z1, z0):
        # we have an g-force on x axis
        # Px=Polynomial([x2, x1, x0]) # 2 order
        Px = self.free_fall_with_res(t, x2, x1, x0)
        Py=Polynomial([y1, y0])
        Pz=Polynomial([z1, z0])
        return np.concatenate([Px, Py(t), Pz(t)])

    def fit(self, tracked):
        t = np.arange(len(tracked))
        xyz = np.concatenate([tracked[:,0], tracked[:,1], tracked[:,2]])
        weights, _ = curve_fit(self.func, t, xyz, p0=[1,1,1,0,0,0,0] )
        self.x2 = weights[0]
        self.x1 = weights[1]
        self.x0 = weights[2]
        self.y1 = weights[3]
        self.y0 = weights[4]
        self.z1 = weights[5]
        self.z0 = weights[6]
        
    def predict(self, t):
        return self.func(t, self.x2, self.x1, self.x0, self.y1, self.y0, self.z1, self.z0).reshape(3, -1)
    
    def calc_dist_error(self, tracked):
        err = []
        t = np.arange(len(tracked))
        p_hat = self.predict(t).T
        p = tracked
        # print(p_hat, p)
        dist = np.sqrt(np.sum(np.power((p-p_hat), 2), axis=1))

        return np.mean(dist)

    def get_params(self):
        return self.x2, self.x1, self.x0, self.y1, self.y0, self.z1, self.z0