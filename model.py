import cv2
import numpy as np
import os
import configparser
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Flatten, Lambda, concatenate, Permute, Add
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling3D, Conv2DTranspose, UpSampling2D, UpSampling3D, ZeroPadding2D, ZeroPadding3D, Cropping3D
from tensorflow.keras.layers import BatchNormalization, Activation
# from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from datetime import datetime

class LF_edit_model(object):
	"""docstring for LF_edit_model"""
	def __init__(self, session, training=True):
		# read in config
		config = configparser.ConfigParser()
		config.read('config.ini')
		self.training_pairs = eval(config['Data']['training_pairs'])
		self.session = session
		self.num_rows = int(config['LF_image']['num_rows'])
		self.num_cols = int(config['LF_image']['num_cols'])
		self.num_channels = int(config['LF_image']['num_channels'])
		self.height = int(config['LF_image']['height'])
		self.width = int(config['LF_image']['width'])
		checkpoints = bool(config['Training']['checkpoints'])
		# optimizer = Adam(0.0002, 0.5)

		# build discriminator and generator for training
		if training:
			# self.discriminator = self.build_discriminator()
			if os.path.isfile('checkpoints/generator.h5') and checkpoints:
				self.generator = load_model('checkpoints/generator.h5')
			else:
				self.generator = self.build_generator()
				self.generator.compile(loss='mean_squared_logarithmic_error', optimizer='sgd')


	def Conv4D(self, input_tensor, output_features, angular_filter, spatial_filter, angular_padding, spatial_padding):
		shape = input_tensor.get_shape().as_list()
		num_rows = shape[1]
		num_cols = shape[2]
		input_features = shape[5]

		# reshape and angular filter
		angular = Reshape([num_rows, num_cols, self.height*self.width, input_features])(input_tensor)
		angular = Conv3D(output_features, angular_filter+(1,), padding=angular_padding)(angular)
		# angular = Activation("relu")(angular)
		# angular = BatchNormalization()(angular)

		if angular_padding == "valid":
			num_rows = num_rows - angular_filter[0] + 1
			num_cols = num_cols - angular_filter[1] + 1

		# reshape and spatial filter
		spatial = Reshape([num_rows, num_cols, self.height, self.width, output_features])(angular)
		spatial = Reshape([num_rows*num_cols, self.height, self.width, output_features])(spatial)
		spatial = Conv3D(output_features, (1,)+spatial_filter, padding=spatial_padding)(spatial)
		# spatial = Activation("relu")(spatial)
		# spatial = BatchNormalization()(spatial)
		output = Reshape([num_rows, num_cols, self.height, self.width, output_features])(spatial)
		return output


	def build_generator(self):
		original_imgs = Input(shape=(self.num_rows, self.num_cols, self.height, self.width, 1))
		modified_central_img = Input(shape=(1, 1, self.height, self.width, 1))

		# conv on modified central
		# modified_central = concatenate([modified_central_img, modified_central_depth])
		modified_central = modified_central_img
		modified_central_features = self.Conv4D(modified_central, 24, (1,1), (3,3), "valid", "same")
		modified_central_features = self.Conv4D(modified_central_features, 24, (1,1), (3,3), "valid", "same")
		modified_central_features = self.Conv4D(modified_central_features, 24, (1,1), (3,3), "valid", "same")

		# conv on original light field image
		original_imgs_features = self.Conv4D(original_imgs, 24, (3,3), (3,3), "valid", "same")
		original_imgs_features = self.Conv4D(original_imgs_features, 24, (3,3), (3,3), "valid", "same")
		original_imgs_features = self.Conv4D(original_imgs_features, 24, (3,3), (3,3), "valid", "same")

		# combine
		all_features = concatenate([modified_central_features, original_imgs_features])
		intermedia_output = self.Conv4D(all_features, 48, (1,1), (3,3), "valid", "same") # 1*1*376*541*48

		intermedia_output = Flatten()(intermedia_output)

		intermedia_output = Dense(1)(intermedia_output)

		intermedia_output = Dense(9763968)(intermedia_output)

		intermedia_output = Reshape([1,1,376,541,48])(intermedia_output)

		output = intermedia_output

		# # intermedia reshape & crop & concatenate
		# intermedia = Reshape([1, self.height, self.width, 48])(intermedia_output)
		# intermedia = Permute((4,2,3,1))(intermedia)
		# intermedia_first_half = Cropping3D(cropping=((0,24),(0,0),(0,0)))(intermedia)
		# intermedia_first_half = Permute((4,2,3,1))(intermedia_first_half)
		# intermedia_second_half = Cropping3D(cropping=((0,24),(0,0),(0,0)))(intermedia)
		# intermedia_second_half = Permute((4,2,3,1))(intermedia_second_half)

		# modified_central_img_single = Reshape([1, self.height, self.width, 1])(modified_central_img)

		# # refinement
		# refine_input = concatenate([intermedia_first_half, modified_central_img_single, intermedia_second_half])	
		# refine_input = Reshape([self.num_rows, self.num_cols, self.height, self.width, 1])(refine_input)
		# refine_output = self.Conv4D(refine_input, 24, (3,3), (3,3), "valid", "same") # 1*1*376*541*48
		# refine_output = self.Conv4D(refine_output, 24, (3,3), (3,3), "valid", "same")
		# refine_output = self.Conv4D(refine_output, 24, (3,3), (3,3), "valid", "same")
		# refine_output = self.Conv4D(refine_output, 48, (1,1), (3,3), "valid", "same")

		# output = Add()([intermedia_output, refine_output])

		model = Model(inputs=[original_imgs, modified_central_img], outputs=output)

		# model.summary()
		# for layer in model.layers:
		# 	print(layer.get_output_at(0).get_shape().as_list())

		return model


	def build_discriminator(self):

		return model

	def train(self, original_imgs, modified_imgs, modified_central_img, epochs, batch_size=4):
		# create log file
		time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
		log_file = open('checkpoints/logs/%s.txt' % time, 'a+')

		# read ground truth
		self.num_pairs = len(self.training_pairs)
		ground_truth = np.zeros((self.num_pairs, 1, 1, self.height, self.width, self.num_rows*self.num_cols-1), dtype=np.uint8)
		for pair in range(self.num_pairs):
			counter = 0
			for row in range(self.num_rows):
				for col in range(self.num_cols):
					if row != (self.num_rows-1)/2 or col != (self.num_cols-1)/2:
						ground_truth[pair, 0, 0, :, :, counter] = modified_imgs[pair, row, col, :, :, 0]
						counter += 1


		for epoch in range(epochs):
			# ---------------------
			#  Train Generator
			# ---------------------

			for iteration in range(50):
				loss = self.generator.train_on_batch([original_imgs/1, modified_central_img/1], ground_truth/1)
				output_str = "Epoch %d iteration %d: %.2f" % (epoch, iteration, loss)
				print(output_str)
				log_file.write(output_str + '\n')
			self.generator.save('checkpoints/generator.h5')

		log_file.close()


	def test(self, modified_imgs, original_imgs, modified_central_img, modified_central_depth):
		time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
		os.mkdir(os.path.join("outputs", time))
		self.generator = load_model('checkpoints/generator.h5')

		cr_cbs = modified_central_img[:,:,:,:,:,1:3]

		modified_central_depth = np.zeros([1,7,7,376,541,1])

		predicteds = self.generator.predict([original_imgs/1, modified_central_img[:,:,:,:,:,0:1]/1, modified_central_depth/1])
		predicteds[predicteds < 0] = 0
		predicteds = predicteds.astype(np.uint8)
		print(np.amin(predicteds))
		print(np.amax(predicteds))
		print(np.mean(predicteds))


		# read ground truth
		self.num_pairs = len(original_imgs)
		ground_truth = np.zeros((self.num_pairs, 1, 1, self.height, self.width, self.num_rows*self.num_cols-1), dtype=np.uint8)
		for pair in range(self.num_pairs):
			counter = 0
			for row in range(self.num_rows):
				for col in range(self.num_cols):
					if row != (self.num_rows-1)/2 or col != (self.num_cols-1)/2:
						ground_truth[pair, 0, 0, :, :, counter] = modified_imgs[pair, row, col, :, :, 0]
						counter += 1
		print(np.amin(ground_truth))
		print(np.amax(ground_truth))
		print(np.mean(ground_truth))

		for i in range(self.num_pairs):
			os.mkdir(os.path.join("outputs", time, str(i)))
			# predicted = predicteds[i]
			cr_cb = cr_cbs[i]
			cr_cb = np.reshape(cr_cb, (self.height, self.width, 2))
			for j in range(ground_truth.shape[5]):
				ground_truth_y = ground_truth[i,:,:,:,:,j]
				predicted_y = predicteds[i,:,:,:,:,j]
				ground_truth_y = np.reshape(ground_truth_y, (self.height, self.width, 1))
				predicted_y = np.reshape(predicted_y, (self.height, self.width, 1))
				
				ground_truth_img = np.concatenate((ground_truth_y, cr_cb), axis=2)
				predicted_img = np.concatenate((predicted_y, cr_cb), axis=2)
				ground_truth_output_img = cv2.cvtColor(ground_truth_img, cv2.COLOR_YCR_CB2BGR)
				predicted_output_img = cv2.cvtColor(predicted_img, cv2.COLOR_YCR_CB2BGR)

				if j >= 24:
					j = j+1
				row = int(j / 7)
				col = j % 7
				ground_truth_output_path = os.path.join("outputs", time, str(i), "ground_truth_"+str(j)+".png")
				# predicted_output_path = os.path.join("outputs", time, str(i), "predicted_"+str(j)+".png")
				predicted_output_path = os.path.join("outputs", time, str(i), str(row)+"_"+str(col)+".png")
				# cv2.imwrite(ground_truth_output_path, ground_truth_output_img)				
				cv2.imwrite(predicted_output_path, predicted_output_img)
