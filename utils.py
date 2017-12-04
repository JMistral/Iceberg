import pandas as pd
import numpy as np
import pylab
import colorsys
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from skimage.transform import rotate


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
        rgb_arrays.append(rgb)
            
    rgb_arrays = np.array(rgb_arrays)
    hsv_arrays = RGB_to_HSV(rgb_arrays)
    # use uint8 0-255 for RGB code
    #rgb_arrays = np.uint8(rgb_arrays)
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
                rgb_data[i,x,y,:] = rgb
    rgb_data = np.array(rgb_data)            
    return rgb_data

    
def lee_filter(img, size):
    img_mean = uniform_filter(img, size)
    img_sqr_mean = uniform_filter(img**2, size)
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def flatten_img(img,labels,angles,size):
    processed_data_list = []
    for i in range(np.shape(img)[0]):
        img_flattened = []
        img_flattened.append(img[i,:,:,0].reshape(size[0]*size[1]))
        img_flattened.append(img[i,:,:,1].reshape(size[0]*size[1]))
        img_flattened.append(img[i,:,:,2].reshape(size[0]*size[1]))
        img_flattened.append(labels)
        img_flattened.append(angles)
        processed_data_list.append(img_flattened)
        
    return np.array(processed_data_list)



def transform(img):
    degrees = np.array(range(30,330))
    
    new_array = np.zeros_like(img)
    for i in range(img.shape[0]):
        angle =  np.random.choice(degrees, 1)
        new_img = rotate(img[i,:,:,:], angle, mode = 'edge')
        new_array[i,:,:,:] = new_img
    return new_array

def simple_augmentation(X_train, X_angle_train, y_train, percent ):
    
    X_angle_train = X_angle_train.reshape((X_angle_train.shape[0],1))
    y_train = y_train.reshape((y_train.shape[0],1))
    
    ships = np.where(y_train ==0)[0]
    chosen_ships = np.random.choice(ships, np.int(ships.shape[0]*percent))
    icebergs = np.where(y_train ==1)[0]
    chosen_icebergs = np.random.choice(icebergs, np.int(icebergs.shape[0]*percent))
    
    new_ships = transform(X_train[chosen_ships,:,:,:])
    new_icebergs = transform(X_train[chosen_icebergs,:,:,:])
    
    new_X = np.concatenate( (X_train, new_ships, new_icebergs) )
    new_angle = np.concatenate( (X_angle_train, X_angle_train[chosen_ships], X_angle_train[chosen_icebergs]) )
    new_y = np.concatenate( (y_train, y_train[chosen_ships], y_train[chosen_icebergs]) )
    
    return new_X, new_angle, new_y


def Preprocessing(df_SAR,train = True,color_space = 'RGB'):
    # get color composite
    rgb, hsv = RGB_HSV_composite(df_SAR)
    
    # speckle filtering
    rgb_filtered = np.zeros_like(rgb)
    for i in range(np.shape(rgb)[0]):
        rgb_filtered[i,:,:,:] = lee_filter(rgb[i,:,:,:], 3)
    
    # other info
    angle = np.array(df_SAR.inc_angle)
    if train:
        y = np.array(df_SAR["is_iceberg"])
    else:
        # dummy label for test data
        y = 2*np.ones_like(df_SAR["id"].values)
    
    #image augmentation
    new_rgb, new_angle, new_y = simple_augmentation(rgb_filtered, angle, y, 0.5 )
        
    if color_space is 'HSV':
        new_hsv = HSV_to_RGB(new_rgb)
        if train:
            return new_hsv, new_angle, new_y
        else:
            return new_hsv, new_angle
        
    elif color_space is 'RGB':
        if train:
            return new_rgb, new_angle, new_y
        else:
            return new_rgb, new_angle
    else:
        print('wrong color space name')
    
    