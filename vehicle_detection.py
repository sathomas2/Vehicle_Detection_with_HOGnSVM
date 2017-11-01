# Import necessary libraries
import glob
import os
import time
import numpy as np
import pandas as pd
import cv2
from moviepy.editor import VideoFileClip
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.ndimage.measurements import label

# Read image filenames, separating out a test set using the first 10% of the chronological images
fns_vehicle1 = glob.glob('vehicles/GTI_Far/*.png')
fns_vehicle1TEST = fns_vehicle1[:len(fns_vehicle1)//10]
fns_vehicle2 = glob.glob('vehicles/GTI_Left/*.png')
fns_vehicle2TEST = fns_vehicle2[:len(fns_vehicle2)//10]
fns_vehicle3 = glob.glob('vehicles/GTI_Right/*.png')
fns_vehicle3TEST = fns_vehicle3[:len(fns_vehicle3)//10]
fns_vehicle4 = glob.glob('vehicles/GTI_MiddleClose/*.png')
fns_vehicle4TEST = fns_vehicle4[:len(fns_vehicle4)//10]
fns_vehicleTEST = fns_vehicle1TEST + fns_vehicle2TEST + fns_vehicle3TEST + fns_vehicle4TEST
fns_vehicle5 = glob.glob('vehicles/KITTI_extracted/*.png')
fns_vehicle = fns_vehicle1[len(fns_vehicle1)//10:] + fns_vehicle2[len(fns_vehicle2)//10:] + fns_vehicle3[len(fns_vehicle3)//10:] + fns_vehicle4[len(fns_vehicle4)//10:] + fns_vehicle5

fns_novehicle1 = glob.glob('novehicles/GTI/image*.png')
fns_novehicleTEST = fns_novehicle1[:len(fns_novehicle1)//10]
fns_novehicle2 = glob.glob('novehicles/Extras/extra*.png')
fns_novehicle = fns_novehicle1[len(fns_novehicle1)//10:] + fns_novehicle2

# Function for color conversion.
def convert_color(img, color_space='RGB'):
    return cv2.cvtColor(img, eval('cv2.COLOR_BGR2'+color_space))

# Get Histogram of Gradient Features. The vis argument lets you visualize the results, which helps when deciding 
# how many orientation bins, pixels per cell, and cells per block to use.
def get_hog_features(img, orient=12, pix_per_cell=(8,8), cell_per_block=(2,2),
                     vis=False, feature_vec=True, bnorm='L2-Hys', cspace='GRAY'):
    if cspace != 'RGB':
        img = cv2.cvtColor(img, eval('cv2.COLOR_RGB2'+cspace))
        
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell[0], pix_per_cell[1]),
                                  cells_per_block=(cell_per_block[0], cell_per_block[1]), 
                                  transform_sqrt=False, block_norm=bnorm,
                                  visualise=vis, feature_vector=feature_vec)
        return hog_image
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell[0], pix_per_cell[1]),
                       cells_per_block=(cell_per_block[0], cell_per_block[1]), 
                       transform_sqrt=False, block_norm=bnorm,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Spatial features. This is simply raw pixel data of the image in HLS color space downsized to 16x16.
def bin_spatial(img, color_space='HLS', size=(16, 16)):
    if color_space != 'RGB':
        img = cv2.cvtColor(img, eval('cv2.COLOR_RGB2'+color_space))
    features = cv2.resize(img, size).ravel()
    return features

# Color histogram features of image in YCrCb color space.
def color_hist(img, color_space='YCrCb', nbins=32):
    if color_space != 'RGB':
        img = cv2.cvtColor(img, eval('cv2.COLOR_RGB2'+color_space))
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

# Combine all three above features into a single vector. 
def extract_features(img_fn, HOG=True, SPACE=True, COLOR=True, fn=True, flip=False):
    features = []
    if fn:
        img = convert_color(cv2.imread(img_fn))
        if flip:
            img = cv2.flip(img, 1)
    else: 
        img = cv2.resize(img_fn, (64,64))
    if HOG:
        features.append(get_hog_features(img))
    if SPACE:
        features.append(bin_spatial(img))
    if COLOR:
        features.append(color_hist(img))
    return np.concatenate(features)

# Sliding window function described in README. I used the code from the quiz in the Udacity Self-Driving Car lesson
# with slight alterations.
def slide_window(img_orig, clf, X_scaler, x_start_stop=[700, None], y_start_stop=[100, 170], 
                    xy_window=(30, 15), xy_overlap=(0.5, 0.5), color=(255, 0, 0), thick=4):
    img = np.copy(img_orig)
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_img = img[starty:endy, startx:endx]
            window_features = extract_features(window_img, fn=False)
            features_scaled = X_scaler.transform(window_features.reshape(1,-1))
            if clf.predict(features_scaled) == 1:
                window_list.append(((startx, starty), (endx, endy)))
    return window_list

# Function to get heat map and then use scipy.ndimage.measurements.label to distinguish between separate detections. 
# This code was inspired by the code from Udacity's lessons.
def add_heat(img, bbox_list, threshold):
    heatmap = np.zeros_like(img[:,:,0])
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    heatmap[heatmap <= threshold] = 0
    labels = label(heatmap)
    return labels

# Draw labeled bounding boxes, thresholding on a single frame image as well as smoothing boxes over previous 5 frames.
# Most of this code was taken from the Udacity lesson, though with my own alterations for the thresholding and smoothing.
def draw_labeled_bboxes(orig_img, threshold=5):
    img = np.copy(orig_img)
    bbox_list = slide_window(img,svc, X_scaler, y_start_stop=[400, 500], xy_window=(64, 64), xy_overlap=(0.5, 0.5)) + \
                slide_window(img,svc, X_scaler, y_start_stop=[400, 600], xy_window=(96, 96), xy_overlap=(0.75, 0.75)) + \
                slide_window(img,svc, X_scaler, y_start_stop=[400, 700], xy_window=(128, 128), xy_overlap=(0.75, 0.75)) + \
                slide_window(img,svc, X_scaler, y_start_stop=[400, 700], xy_window=(160, 160), xy_overlap=(0.75, 0.75))
    # Single frame threshold, i.e. for a patch to count at least 2 windows must have detected the patch. See README section 4.
    if len(bbox_list) > 1:
        bboxes.append(bbox_list)
    if len(bboxes) < threshold:
        thresh = len(bboxes)
    else:
        thresh = threshold
    # Previous 5 frames smoothing threshold. See README section 4.
    if len(bboxes) > 0:
        labels = add_heat(img, np.concatenate(bboxes[-threshold:]), thresh)
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 4)
    # Return the image
    return img

# Extract the features. Also flip the no_car images and extract the features. See README end of Section 1.
t = time.time()
car_features = [extract_features(fn) for fn in fns_vehicle]
nocar_features_noflip = [extract_features(fn) for fn in fns_novehicle]
nocar_features_flip = [extract_features(fn, flip=True) for fn in fns_novehicle]
nocar_features = nocar_features_noflip + nocar_features_flip
TESTcar_features = [extract_features(fn) for fn in fns_vehicleTEST]
TESTnocar_features = [extract_features(fn) for fn in fns_novehicleTEST]
print('{} sec to extract features'.format(round(time.time() - t)))
print('')

# Combine nocar and car features, scale them to have zero mean and unit variance. 
# Split data into training, validation, and test sets.
features = np.vstack((car_features, nocar_features))
test_features = np.vstack((TESTcar_features, TESTnocar_features))
X_scaler = StandardScaler().fit(features)
X_scaled = X_scaler.transform(features)
X_test = X_scaler.transform(test_features)
y_test = np.hstack((np.ones(len(TESTcar_features)), np.zeros(len(TESTnocar_features))))
y = np.hstack((np.ones(len(car_features)), np.zeros(len(nocar_features))))
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.1)
print('{} cars in training & validation set'.format(len(car_features)))
print('{} NO cars in training & validation set'.format(len(nocar_features)))
print('')
print('{} cars in test set'.format(len(TESTcar_features)))
print('{} NO cars in test set'.format(len(TESTnocar_features)))
print('')

# Train Support Vector Machine Classifier with penalty parameter of 100 and an RBF kernel with a coefficient of 0.0003.
svc = SVC(C=100.0, kernel='rbf', gamma='auto')
t = time.time()
svc.fit(X_train, y_train)
print('{} sec to train SVC'.format(round(time.time() - t)))
print('')
print('Validation Accuracy: {}'.format(svc.score(X_valid, y_valid)))
print('Test Accuracy: {}'.format(svc.score(X_test, y_test)))
print('')

# Make output_video directory if it doesn't already exist.
if not os.path.exists('output_video'):
    os.makedirs('output_video')

# Create bboxes list to keep track of past frames' detections. 
bboxes = []
# Run algorithm on test_video and save the result.
project_video = 'output_video/test_video.mp4'
clip1 = VideoFileClip('test_video.mp4')
clip_lines = clip1.fl_image(draw_labeled_bboxes)
clip_lines.write_videofile(project_video, audio=False)

# Create bboxes list to keep track of past frames' detections. 
bboxes = []
# Run algorithm on project_video and save the result.
project_video = 'output_video/project_video.mp4'
clip1 = VideoFileClip('project_video.mp4')
clip_lines = clip1.fl_image(draw_labeled_bboxes)
clip_lines.write_videofile(project_video, audio=False)