import pandas as pd
import numpy as np
import pylab
import colorsys
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

def RGB_HSV_composite(data):
    rgb_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))

        rgb = np.dstack((r, g, b))
        rgb = rgb * 255
        rgb_arrays.append(rgb)
            
    rgb_arrays = np.array(rgb_arrays)
    hsv_arrays = RGB_to_HSV(rgb_arrays/255)
    # use uint8 0-255 for RGB code
    rgb_arrays = np.uint8(rgb_arrays)
    return rgb_arrays, hsv_arrays


def MP_composite(data):
    MP_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        M = np.hypot(band_1,band_2)
        P = np.arctan2(band_1,band_2)

        MP = np.dstack((M,P))
        MP_arrays.append(MP)
        
    #MP_arrays = np.uint8(MP_arrays)
    return np.array(MP_arrays)

"""
def plotmy3d(c, name):

    data = [
        go.Surface(
            z=c
        )
    ]
    layout = go.Layout(
        title=name,
        autosize=False,
        width=700,
        height=700,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
"""


def RGB_to_HSV(rgb_data):
    hsv_data = np.zeros_like(rgb_data)
    for i in range(np.shape(rgb_data)[0]):
        for x in range(np.shape(rgb_data)[1]):
            for y in range(np.shape(rgb_data)[2]):
                hsv = list(colorsys.rgb_to_hsv(rgb_data[i,x,y,0],rgb_data[i,x,y,1],rgb_data[i,x,y,2]))
                #print(hsv)
                hsv_data[i,x,y,:] = hsv
                
    return hsv_data


def HSV_to_RGB(hsv_data):
    rgb_data = np.zeros_like(hsv_data)
    for i in range(np.shape(hsv_data)[0]):
        for x in range(np.shape(hsv_data)[1]):
            for y in range(np.shape(hsv_data)[2]):
                rgb = np.array(colorsys.hsv_to_rgb(hsv_data[i,x,y,0],hsv_data[i,x,y,1],hsv_data[i,x,y,2]))
                #print(hsv)
                rgb_data[i,x,y,:] = rgb * 255
    #rgb_data = np.array(rgb_data)            
    return np.uint8(rgb_data)

    
def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def flatten_img(img,labels,size):
    processed_data_list = []
    for i in range(np.shape(img)[0]):
        img_flattened = []
        img_flattened.append(img[i,:,:,0].reshape(size[0]*size[1]))
        img_flattened.append(img[i,:,:,1].reshape(size[0]*size[1]))
        img_flattened.append(img[i,:,:,2].reshape(size[0]*size[1]))
        img_flattened.append(labels)
        processed_data_list.append(img_flattened)
        
    return np.array(processed_data_list)