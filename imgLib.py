#!/usr/bin/python
'''
Python library for image processing.

The script requires the following python libraries:
 * numpy
 * pandas
 * matplotlib
 * Python Image Library (PIL) or Pillow
 * scipy
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import scipy



class imgLib():

	def apply_Gauss_filter(self, img, sigma, show=True):
		'''apply Gaussian filter to an image and plot it using a greyscale.

		Arguments:
		---------
		img 	:	image imported with PIL Image.open 
		sigma	:	standard deviation
		show  	: 	boolean for showing plot

		Return:
		---------
		img_filt:	image filtered
		'''
		from scipy import ndimage
		im_filt = scipy.ndimage.gaussian_filter(img, sigma=sigma, truncate=2.0)

		if show:
			fig, (ax1, ax2) = plt.subplots(1,2)
			ax1.set_title('original')
			ax1.imshow(img,cmap='gray')
			ax2.set_title('filtered with Gaussian kernel')
			ax2.imshow(img_filt,cmap='gray')
			plt.show()
		return img_filt


	def replace_RGB_value(self, img, colour_old, colour_new, show=True):
		'''replace a specific RGB value in image imported with PIL and output the result.

		Arguments:
		---------
		img 		:	input image (imported with PIL Image.open)
		colour_old	:	vector with RGB values for old colour
		colour_new	:	vector with RGB values for new colour
		show	  	: 	boolean for showing plot

		Return:
		---------
		img1		:	output image
		'''

		im = np.array(img) #Height x Width x R x G x B (xA, only with RGBA, which is Image default) 
		#temporarily unpack the bands for readability
		red, green, blue = im.T 
		
		#replace white with red... (leaves alpha values alone...)
		white_areas = (red == colour_old[0]) & (green == colour_old[1]) & (blue == colour_old[2])
		im[..., :][white_areas.T] = (colour_new[0], colour_new[1], colour_new[2]) # Transpose back needed

		img1 = Image.fromarray(im)
		if show:
			img1.show()
		return img1


	def img_RBG_to_greyScale(self, img, show=True):
		'''convert image imported with PIL from RGB to greyscale.

		Arguments:
		---------
		img 	:	input image (imported with PIL Image.open)
		show  	: 	boolean for showing plot

		Return:
		---------
		img1	:	output image
		'''

		im = img.convert('1')
		img1 = (np.array(im)==True).astype(float)

		if show:
			plt.imshow(img1,cmap='gray')
			plt.show()
		return img1


	def img_norm(self, img, norm='01', axis=(0,1), show=True): 
		'''normalize image along a dimension, by applying minMax or meanStd normalization.

		Arguments:
		---------
		img 	:	input image (imported with PIL Image.open)
		norm 	:	type of normalization ('01'=minMax, 'z'=meanStd)
		axis	: 	axis for normalization (0=x, 1=y)
		show  	: 	boolean for showing plot
		Return:
		---------
		img_norm:	normalized image
		'''

		if not isinstance(img, np.ndarray):
			img = np.array(img)

		if norm=='z':
			# axis param denotes axes along which mean & std reductions are to be performed
			mean = np.mean(img, axis=axis, keepdims=True)
			std = np.sqrt(((img - mean)**2).mean(axis=axis, keepdims=True))
			img_norm = (img - mean) / std
		elif norm=='01':
			img_norm = (img - np.min(img))/np.ptp(img)

		if show:
			fig, (ax1, ax2) = plt.subplots(1,2)
			ax1.set_title('original')
			ax1.imshow(img,cmap='gray')
			if norm=='z':
				ax2.set_title('z-normalized (mean=0, SD=1)')
			elif norm=='01':
				ax2.set_title('normalized to range [0,1]')
			ax2.imshow(img_norm,cmap='gray')
			plt.show()
		return img_norm


	def init_AOI_map(self, img, sigma, norm, show=[True, True, True]):
		'''initialize API map: convert to grescale; smooth with Gaussian filter;
		invert values for object and background so that the object has the min values;
		normalize the image

		Arguments:
		---------
		img 	:	input image (imported with PIL Image.open)
		sigma	: 	standard deviation for Gaussian filter
		norm 	:	type of normalization ('01'=minMax, 'z'=meanStd)
		show  	: 	boolean for showing plots
		
		Return:
		---------
		img3 	:	output image: AOI map
		'''

		#convert image from RGB to grey scale
		img1 = self.img_RBG_to_greyScale(img, show=show[0])

		#apply Gaussian filter 
		img2 = self.apply_Gauss_filter(img1, sigma=sigma, show=show[1])

		#invert values for object-background from max-min to min-max
		img2 = abs(img2-img2.max())

		#normalize the images
		img3 = self.img_norm(img2, norm, axis=(0,1), show=show[2])

		return img3


	def init_eccentricity_map(self, img, norm, show=True):
		'''initialize eccentricity map: for each pixel, calculate Euclidean distance from 
		the centre of the image; normalize the values either with minMax or meanStd methods.

		Arguments:
		---------
		img 	:	input image (imported with PIL Image.open)
		norm 	:	type of normalization ('01'=minMax, 'z'=meanStd)
		show  	: 	boolean for showing plot

		Return:
		---------
		img1 	:	output image
		'''

		im = np.zeros(img.size).T
		#im = np.zeros([5,5]).T
		sti=-int(im.shape[0]/2.0)
		stj=-int(im.shape[1]/2.0)

		for i in range(im.shape[0]):
			stix=sti+i;
			for j in range(im.shape[1]):

				stjx=stj+j;
				im[i,j]= np.sqrt(np.power(stix,2)+np.power(stjx,2))

		#invert values: center-periphery = max-min
		#im = abs(im-im.max())
		
		#normalize the image
		img1 = self.img_norm(im, norm, axis=(0,1), show=show)

		return img1


	def show_map(self, map):
		'''plot map or image in grescale'''
		plt.imshow(map,cmap='gray')
		plt.show()

		
