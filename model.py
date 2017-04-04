import sklearn
import csv, os.path, math
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, Cropping2D, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split

### Hyperparameters

# data locations
data_dir = './mixed/'
image_dir = data_dir + 'IMG/'
driving_log = data_dir + 'driving_log.csv'
save_model = data_dir + 'model.h5'
save_samples = data_dir + 'samples.csv'

# data info
cropping = ((60, 25), (0, 0))
input_size = (160, 320, 3)
lc = 0.2
rc = -0.2

# model info
p1 = 0.5
p2 = 0.5
p3 = 0.5
p4 = 0.5

# training info
loss = 'mse'
optimizer = 'adam'
epochs = 10
valid_split = 0.2
batch_size = 32

# Utilities

def flatten_data(driving_log, output_file=None, lc=0.2, rc=-0.2):
	"""
	Flattens data from driving log output from simulator to the format:
	(image_filename, steering_angle, throttle, brake, speed, flipped)
	
	image_filename: string
	steering_angle: float
	      throttle: float
	         brake: float
	         speed: float
	       flipped: boolean
	
	Arguments
	---------
	    driving_log: Path to the driving_log.csv file.
	    lc, rc     : Left and right correction values, which will be
		             added to the steering angle of the center image
		             as the angle for the left and right images
		             respectively.
	    output_file: If specified, the formatted tuples will be saved
	                 to 'output_file' as csv file.
	
	Returns a list of formatted tuples.
	"""
	samples = []
	with open(driving_log) as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			# read data from driving_log and populate "samples"
			center, left, right = row[0], row[1], row[2]
			angle, throttle, brake, speed = row[3:]
			angle, throttle = eval(angle), eval(throttle)
			brake, speed = eval(brake), eval(speed)
			samples.append((center, angle, throttle, brake, speed, False))
			samples.append((left, angle+lc, throttle, brake, speed, False))
			samples.append((right, angle+rc, throttle, brake, speed, False))
			samples.append((center, -angle, throttle, brake, speed, True))
			samples.append((left, -(angle+lc), throttle, brake, speed, True))
			samples.append((right, -(angle+rc), throttle, brake, speed, True))
	
	# save file if "output_file" is provided
	if output_file != None:
		with open(output_file, 'w') as out:
			writer = csv.writer(out)
			for row in samples:
				writer.writerow(row)

	return samples


def load_samples(sample_file):
	"""
	Loads a saved sample file.
	
	Arguments
	---------
	
	    sample_file: input csv file containing tuples of the form
	                 -- (name, angle, throttle, brake, speed, flipped)
	"""
	samples = []
	with open(sample_file) as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			name = row[0]
			angle, throttle = eval(row[1]), eval(row[2])
			brake, speed = eval(row[3]), eval(row[4])
			flipped = eval(row[-1])
			samples.append((name, angle, throttle, brake, speed, flipped))
	
	return samples


def generator(samples, image_dir, batch_size=32):
	"""
	Creates a batch data generator from samples.
	
	Arguments
	---------
	       samples: Output from `flatten_data` 
	              --list of (image_file, angle, throttle, brake, speed, flipped)
	    images_dir: Directory containing images of `image_file`.
	    batch_size: Size of a mini-batch. Defaults to 32.
	
	Returns a generator which yields mini-batches of shuffled (X, y) pairs.
	"""
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		# shuffle the samples at each epoch
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			
			images = []
			angles = []
			for batch_sample in batch_samples:
				name = image_dir + batch_sample[0].split('/')[-1]
				angle = batch_sample[1]
				image = load_img(name)
				image = img_to_array(image)
				
				# flip image if the "flipped" flag is on
				if batch_sample[-1]:
					image = np.fliplr(image)
				
				images.append(image)
				angles.append(angle)
			
			X = np.array(images)
			y = np.array(angles)
			yield sklearn.utils.shuffle(X, y)
			

def define_model(input_shape, cropping):
	"""
	Defines the architecture of network.
	"""
	model = Sequential()
	model.add(Cropping2D(cropping=cropping, input_shape=input_shape))
	model.add(Lambda(lambda x: x/127.5 - 1))
	model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Conv2D(64, 3, 3, activation='relu'))
	model.add(Conv2D(64, 3, 3, activation='relu'))
	model.add(Flatten())
	model.add(Dropout(p1))
	model.add(Dense(100))
	model.add(Dropout(p2))
	model.add(Dense(50))
	model.add(Dropout(p3))
	model.add(Dense(10))
	model.add(Dropout(p4))
	model.add(Dense(1))
	
	return model


if __name__ == "__main__":
	# load samples
	if os.path.isfile(save_samples):
		samples = load_samples(save_samples)
	else:
		samples = flatten_data(driving_log, save_samples, lc, rc)
	
	# create data generators
	train_samples, valid_samples = train_test_split(samples,
	                                                test_size=valid_split,
	                                                random_state=9527)
	train_data = generator(train_samples, image_dir, batch_size)
	valid_data = generator(valid_samples, image_dir, batch_size)
	nb_train_data = len(train_samples)
	nb_valid_data = len(valid_samples)
	
	# load model if one exists, to allow fine-tuning of trained model
	if os.path.isfile(save_model):
		model = load_model(save_model)
	else:
		# create model
		model = define_model(input_size, cropping)
		# compile model
		model.compile(optimizer=optimizer, loss=loss)
		
	# train model
	model.fit_generator(generator=train_data,
	                    validation_data=valid_data,
	                    samples_per_epoch=nb_train_data,
	                    nb_epoch=epochs,
	                    nb_val_samples=nb_valid_data)
	# save model
	model.save(save_model)
