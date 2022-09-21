# main.py
#
# Read in training data, initialize GAN, train

import sys
import cv2
import os
import numpy as np
import configparser
from model import LF_edit_model
import tensorflow as tf
import pandas as pd

# avoid warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# avoid error messages
# ccc = tf.ConfigProto()
# ccc.gpu_options.allow_growth = True
# sess = tf.Session(config=ccc)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess =tf.compat.v1.InteractiveSession(config=config)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# read in config
config = configparser.ConfigParser()
config.read('config.ini')
data_dir = config['Data']['data_dir']
training_pairs = eval(config['Data']['training_pairs'])
testing_pairs = eval(config['Data']['testing_pairs'])
# num_pairs = len(training_pairs)
num_rows = int(config['LF_image']['num_rows'])
num_cols = int(config['LF_image']['num_cols'])
num_channels = int(config['LF_image']['num_channels'])
height = int(config['LF_image']['height'])
width = int(config['LF_image']['width'])
epochs = int(config['Training']['epochs'])
batch_size = int(config['Training']['batch_size'])

if __name__ == '__main__':


	if len(sys.argv) == 2 and sys.argv[1] == "test":
		num_pairs = len(testing_pairs)
		original_imgs = np.zeros((num_pairs, num_rows, num_cols, height, width, 1), dtype=np.uint8)
		modified_imgs = np.zeros((num_pairs, num_rows, num_cols, height, width, 1), dtype=np.uint8)
		modified_central_img = np.zeros((num_pairs, 1, 1, height, width, num_channels), dtype=np.uint8)
		modified_central_depth = np.zeros((num_pairs, 1, 1, height, width, 1), dtype=np.float)
		for pair_index, testing_pair in enumerate(testing_pairs):
			# read in original image
			original_img_dir = os.path.join(data_dir, testing_pair, "original")
			for row in range(num_rows):
				for col in range(num_cols):
					original_img_path = os.path.join(original_img_dir, str(row) + '_' + str(col) + '.png')
					original_img = cv2.imread(original_img_path)
					original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2YCR_CB)
					original_imgs[pair_index, row, col] = original_img[:,:,0:1]
			# read in modified image
			modified_img_dir = os.path.join(data_dir, testing_pair, "modified")
			for row in range(num_rows):
				for col in range(num_cols):
					modified_img_path = os.path.join(modified_img_dir, str(row) + '_' + str(col) + '.png')
					modified_img = cv2.imread(modified_img_path)
					modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2YCR_CB)
					if (row == (num_rows-1)/2 and col  == (num_cols-1)/2):
						modified_central_img[pair_index, 0, 0] = modified_img
					modified_imgs[pair_index, row, col] = modified_img[:,:,0:1]

			# depth
			depth_path = os.path.join('depth', '20', 'depth.csv')
			df = pd.read_csv(depth_path, header=None)
			modified_central_depth[pair_index, 0, 0, :, :, 0] = df.values
		
		gan = LF_edit_model(session=sess)
		gan.test(original_imgs=original_imgs, modified_imgs=modified_imgs, modified_central_img=modified_central_img, modified_central_depth=modified_central_depth)

	else:
		num_pairs = len(training_pairs)
		original_imgs = np.zeros((num_pairs, num_rows, num_cols, height, width, 1), dtype=np.uint8)
		modified_imgs = np.zeros((num_pairs, num_rows, num_cols, height, width, 1), dtype=np.uint8)
		modified_central_img = np.zeros((num_pairs, 1, 1, height, width, 1), dtype=np.uint8)
		modified_central_depth = np.zeros((num_pairs, 1, 1, height, width, 1), dtype=np.float)
		for pair_index, training_pair in enumerate(training_pairs):
			# read in original image
			original_img_dir = os.path.join(data_dir, training_pair, "original")
			for row in range(num_rows):
				for col in range(num_cols):
					original_img_path = os.path.join(original_img_dir, str(row) + '_' + str(col) + '.png')
					original_img = cv2.imread(original_img_path)
					original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2YCR_CB)
					original_imgs[pair_index, row, col] = original_img[:,:,0:1]
			# read in modified image
			modified_img_dir = os.path.join(data_dir, training_pair, "modified")
			for row in range(num_rows):
				for col in range(num_cols):
					modified_img_path = os.path.join(modified_img_dir, str(row) + '_' + str(col) + '.png')
					modified_img = cv2.imread(modified_img_path)
					modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2YCR_CB)
					if (row == (num_rows-1)/2 and col  == (num_cols-1)/2):
						modified_central_img[pair_index, 0, 0] = modified_img[:,:,0:1]
					modified_imgs[pair_index, row, col] = modified_img[:,:,0:1]

			# depth
			# depth_path = os.path.join('depth', modified + '.csv')
			# df = pd.read_csv(depth_path, header=None)
			# modified_central_depth[pair_index, 0, 0, :, :, 0] = df.values
			
		gan = LF_edit_model(session=sess)
		gan.train(original_imgs=original_imgs, modified_imgs=modified_imgs, modified_central_img=modified_central_img, epochs=epochs, batch_size=batch_size)
