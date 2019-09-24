# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:58:12 2019

@author: hh_s
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter



base_Std=np.load('C:/Users/hh_s/Desktop/bases_compri/stdbase.npy')
base_avg=np.load('C:/Users/hh_s/Desktop/bases_compri/avg_Base.npy')
wf_base=base_Std/base_avg


#%%

bin_test=2000

sample_data_std=base_Std[:,bin_test]

#%%

plt.plot(sample_data_std)

#%%
base_avg=np.load('C:/Users/hh_s/Desktop/bases_compri/avg_Base.npy')
sample_data_avg=base_avg[:,bin_test]
wf=sample_data_std/sample_data_avg
#%%
plt.plot(wf)
#%%
sns.set_style('darkgrid')
sns.distplot(wf)

#%%
hist, bin_edges = np.histogram(sample_data_std)

#%%
n, bins, patches = plt.hist(x=wf, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.axvline(wf.mean(), color='k', linestyle='dashed', linewidth=1)
anomalies_start=wf.mean()+10*wf.std()
plt.axvline(anomalies_start, color='r', linestyle='dashed', linewidth=1)


#%%

q75, q25 = np.percentile(wf, [75 ,25])
iqr = q75 - q25

n, bins, patches = plt.hist(x=wf, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#plt.axvline(wf.mean(), color='k', linestyle='dashed', linewidth=1)
anomalies_start=q75+1.5*iqr
plt.axvline(anomalies_start, color='r', linestyle='dashed', linewidth=1)
#%%
def plot_hist(data):
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    print iqr
    
#    n, bins, patches = plt.hist(data,weights=np.ones(len(data)) / len(data), bins=50,color='#0504aa',
#                                alpha=0.7, rwidth=0.85)
    n, bins, patches = plt.hist(x=data,bins='auto',color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim(xmax=np.percentile(data,99.999999))
#    maxfreq = n.max()
    # Set a clean upper y-axis limit.
#    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    #plt.axvline(wf.mean(), color='k', linestyle='dashed', linewidth=1)
    anomalies_start=q75+1.5*iqr
#    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.axvline(anomalies_start, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(np.percentile(data,99.99), color='b', linestyle='dashed', linewidth=1)

#%%
    
#plot_hist(np.mean(base_Std[:,1998:2002]/base_avg[:,1998:2002],axis=1))
#plot_hist(base_Std[:,1013]/base_avg[:,1013])
#%%
q75,q25=np.percentile(wf_base, [75 ,25],axis=0)
iqr=q75-q25
ref_99,ref_iqr=np.percentile(wf_base,99.999,axis=0),q75+1.5*iqr
#%%
#plt.plot(ref_iqr)
#plt.plot(ref_99)
zona_test=wf_base[:,2000]
#plt.plot(ref_iqr)
#plot_hist(zona_test)
plt.plot(zona_test>ref_99[2000],'.')
#%%
for i in range(20):
    plt.figure()
    plt.imshow(wf_base[i*1000:(i+1)*1000,:]-ref_99>0,aspect='auto')
#%%LA DATA PIOLA
wf_real=np.load('C:/Users/hh_s/Desktop/stds_Bardeados/wf.npy')
#%%
for i in range(26):
    plt.figure()
    plt.imshow(wf_real[i*1000:(i+1)*1000,:]>ref_99,aspect='auto')
#    plt.figure()
#    plt.imshow(wf_real[i*1000:(i+1)*1000,:],aspect='auto',vmax=0.2)
#%%
plot_hist(wf_real[:,2920])
#%%lets go MAD
from astropy.stats import median_absolute_deviation
mad_data=median_absolute_deviation(wf_base,axis=0)
median_data=np.median(wf_base,axis=0)
#%%
for i in range(26):
#    fig=plt.figure(figsize=(12, 12))
    wf_mad=np.abs(wf_real[i*1000:(i+1)*1000,:]-median_data)/mad_data
#    fig.add_subplot(1,2,1)
#    plt.imshow(wf_mad>90,aspect='auto',cmap=plt.cm.gray)
    plt.imsave(str(i)+'.png',wf_mad[:,1000:4000]>90, cmap=plt.cm.gray)
#    fig.add_subplot(1,2,2)
#    plt.imshow(wf_real[i*1000:(i+1)*1000,:],aspect='auto',vmax=0.2,cmap='jet')
    
#%%
import cv2
import glob

img = cv2.imread('9.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

print(n_labels)

size_thresh = 10
for i in range(1, n_labels):
    if stats[i, cv2.CC_STAT_AREA] >= size_thresh:
        print(stats[i, cv2.CC_STAT_AREA])
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)

cv2.imwrite("out.jpg", img)

#%%
import cv2
import warnings
warnings.filterwarnings("ignore", category=UnicodeWarning)

# Read image
im = cv2.imread("9.png", cv2.IMREAD_GRAYSCALE)
 
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector()
 
# Detect blobs.
keypoints = detector.detect(im)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
#cv2.startWindowThread()
#cv2.namedWindow("preview")
cv2.imwrite("ou2.jpg",im_with_keypoints)
#cv2.imshow("Keypoints", im_with_keypoints)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
