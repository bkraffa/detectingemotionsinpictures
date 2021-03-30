#Imports
import pandas as pd 
import numpy as np 
import os, sys, inspect
from six.moves import cPickle as pickle 
import scipy.misc as misc
import tensorflow as tf
import joblib as jl

#Parameters
IMAGE_SIZE = 48
NUM_LABELS = 7

#10% of the data as validation
VALIDATION_PERCENT = 0.1

#Normalization
IMAGE_LOCATION_NORM = IMAGE_SIZE // 2

#For training	
train_error_list = []
train_step_list = []

#For validation
valid_error_list = []
valid_step_list = []

#Emotions dictionary
emotion = {0:'anger', 1:'disgust', 2: 'fear',
           3:'happy', 4:'sad', 5:'suprised', 6:'neutral'}

#Creating a class for the test result
class testResult:

	def __init__(self):
		self.anger = 0
		self.disgust = 0
		self.fear = 0
		self.happy = 0
		self.sad = 0
		self.suprised = 0
		self.neutral = 0

	def evaluate (self, label):
		if (0 == label):
			self.anger = self.anger + 1
		elif (1 == label):
			self.disgust = self.disgust + 1
		elif (2 == label):
			self.fear = self.fear + 1
		elif (3 == label):
			self.happy = self.happy + 1
		elif (4 == label):
			self.sad = self.sad + 1
		elif (5 == label):
			self.suprised = self.suprised + 1
		elif (6 == label):
			self.neutral = self.neutral + 1

	def display_result (self, evaluations):
		print("anger = " + str((self.anger/float(evaluations))*100) + "%")
		print("disgust = " + str((self.disgust/float(evaluations))*100) + "%")
		print("fear = " + str((self.fear/float(evaluations))*100) + "%")
		print("happy = " + str((self.happy/float(evaluations))*100) + "%")
		print("sad = " + str((self.sad/float(evaluations))*100) + "%")
		print("suprised = " + str((self.suprised/float(evaluations))*100) + "%")

#Function used to read the data
def read_data(data_dir, force = False):
	def create_onehot_label(x):
		label = np.zeros((1,NUM_LABELS), dtype = np.float32)
		label[:,int(x)] = 1
		return label

	pickle_file = os.path.join(data_dir, 'EmotionDetectorData.pickle')
	if force or not os.path.exists(pickle_file):
		train_filename = os.path.join(data_dir, "train.csv")
		df = pd.read_csv(train_filename)
		df['Pixels'] = df['Pixels'].apply(lambda x: np.fromstring(x, sep= " ") / 255) #Pixel range is 0:255, so we normalize it to 0:1
		df = df.dropna()
		print('Reading train.csv ...')

		train_images = np.vstack(df['Pixels']).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) #(rows, height, width, channel)
		print(train_images.shape)	
		train_labels = np.array(list(map(create_onehot_label, df['Emotion'].values))).reshape(-1, NUM_LABELS) #(rows,emotion_label)
		print(train_labels.shape)

		permutations = np.random.permutation(train_images.shape[0]) #set a random permutation to separate the validation and train datasets
		train_images = train_images[permutations]
		train_labels = train_labels[permutations]
		validation_percent = int(train_images.shape[0] * VALIDATION_PERCENT)
		validation_images = train_images[:validation_percent]
		validation_labels = train_labels[:validation_percent]
		train_images = train_images[validation_percent:]
		train_labels = train_labels[validation_percent:]

		print('Reading test.csv ...')
		test_filename = os.path.join(data_dir, 'test.csv')
		df2 = pd.read_csv(test_filename)
		df2['Pixels'] = df2['Pixels'].apply(lambda x: np.fromstring(x, sep = ' ') / 255)
		df2 = df2.dropna()
		test_images = np.vstack(df2['Pixels']).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

		with open(pickle_file, "wb") as file:
			try:
				print('\n Saving ...')
				save = {
				'train_images':train_images,
				'train_labels':train_labels,
				'validation_images':validation_images,
				'validation_labels':validation_labels,
				'test_images':test_images
				} 
				pickle.dump(save,file,pickle.HIGHEST_PROTOCOL)
			except:
				print("It wasn't possible to save =( ...")

	with open(pickle_file, 'rb') as file:
		save  = pickle.load(file)
		train_images = save['train_images']
		train_labels = save['train_labels']
		validation_images = save['validation_images']
		validation_labels = save['validation_labels']
		test_images = save['test_images']	

	return train_images, train_labels, validation_images, validation_labels, test_images

#Other auxiliar functions

def add_to_regularization_loss(W, b):
	tf.add_to_collection('losses', tf.nn.l2_loss(W))
	tf.add_to_collection('losses', tf.nn.l2_loss(b))

def weight_variable(shape, stddev = 0.2, name = None):
	initial = tf.random.truncated_normal(shape, stddev = stddev)
	if name is None:
		return tf.variable(initial)
	else:
		return tf.get_variable(name, initializer = initial)

def bias_variable(shape, name = None):
	initial = tf.constant(0.0, shape = shape)
	if name is None:
		return tf.variable(initial)
	else:
		return tf.get_variable(name, initializer = initial)







