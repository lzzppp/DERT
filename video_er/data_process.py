
import os
import csv
import math
import torch
import random
import os.path as osp
from PIL import Image
# from lbtransforms import bulid_transforms
from torchvision.transforms import *
from torch.utils.data import Dataset, DataLoader

def read_image(img_path):
	"""Keep reading image until succeed.
	This can avoid IOError incurred by heavy IO process."""
	got_img = False
	if not osp.exists(img_path):
		raise IOError("{} does not exist".format(img_path))
	while not got_img:
		try:
			img = Image.open(img_path).convert('RGB')
			got_img = True
		except IOError:
			print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
			pass
	return img

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

class Random2DTranslation(object):
	"""
	With a probability, first increase image size to (1 + 1/8), and then perform random crop.

	Args:
	- height (int): target image height.
	- width (int): target image width.
	- p (float): probability of performing this transformation. Default: 0.5.
	"""

	def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
		self.height = height
		self.width = width
		self.p = p
		self.interpolation = interpolation

	def __call__(self, img):
		"""
		Args:
		- img (PIL Image): Image to be cropped.
		"""
		if random.uniform(0, 1) > self.p:
			return img.resize((self.width, self.height), self.interpolation)

		new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
		resized_img = img.resize((new_width, new_height), self.interpolation)
		x_maxrange = new_width - self.width
		y_maxrange = new_height - self.height
		x1 = int(round(random.uniform(0, x_maxrange)))
		y1 = int(round(random.uniform(0, y_maxrange)))
		croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
		return croped_img

def get_transforms(height, width, command, is_train=False):

	imagenet_mean = [0.485, 0.456, 0.406]
	imagenet_std = [0.229, 0.224, 0.225]
	normalize = Normalize(mean=imagenet_mean, std=imagenet_std)
	data_augment = set(command.split())
	print('Using augmentation:', data_augment)
	transforms = []
	if 'crop' in data_augment:
		transforms.append(Random2DTranslation(height, width))
	else:
		transforms.append(Resize((height, width)))
	transforms.append(RandomHorizontalFlip())
	if 'color-jitter' in data_augment:
		transforms.append(ColorJitter())
	transforms.append(ToTensor())
	transforms.append(normalize)
	if 'random-erase' in data_augment:
		transforms.append(RandomErasing())
	transforms = Compose (transforms)
	if is_train:
		print ('Using transform:', transforms)
	return transforms


def get_test_transforms(height, width):
	imagenet_mean = [0.485, 0.456, 0.406]
	imagenet_std = [0.229, 0.224, 0.225]
	normalize = Normalize (mean=imagenet_mean, std=imagenet_std)

	transforms = []
	transforms += [Resize ((height, width))]

	transforms += [ToTensor ()]
	transforms += [normalize]
	transforms = Compose (transforms)

	return transforms


class ImageDataset (Dataset):
	def __init__(self, dataset1, dataset2, labels, path, transform=None):
		self.dataset1 = dataset1
		self.dataset2 = dataset2
		self.labels = labels
		self.transform = transform
		self.PATH = path
	
	def __len__(self):
		return len (self.dataset1)
	
	def __getitem__(self, index):
		img_path1 = self.dataset1[index]
		img_path2 = self.dataset2[index]
		label = self.labels[index]
		image1 = read_image (self.PATH + img_path1)
		image2 = read_image (self.PATH + img_path2)
		if self.transform is not None:
			img1 = self.transform (image1)
			img2 = self.transform (image2)
		return img1, img2, label


def data_processer(path):
	transformer = get_transforms (128, 64, "crop random-erase", is_train=True)
	transformer_test = get_test_transforms (128, 64)
	file = open (path + "train_list.csv", "r")
	reader = csv.reader (file)
	next (reader)
	dataset1, dataset2, labels = [], [], []
	for line in reader:
		dataset1.append ('/'.join (line[1].split ('\\')))
		dataset2.append ('/'.join (line[2].split ('\\')))
		labels.append (int (line[3]))
	train_dataset = ImageDataset(dataset1, dataset2, labels, path, transformer)
	train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4,
	                              pin_memory=True, drop_last=True)
	file.close ()
	
	test_file = open (path + "test_list.csv", "r")
	test_reader = csv.reader (test_file)
	next (test_reader)
	dataset1, dataset2, labels = [], [], []
	for line in test_reader:
		dataset1.append ('/'.join (line[1].split ('\\')))
		dataset2.append ('/'.join (line[2].split ('\\')))
		labels.append (int (line[3]))
	test_file.close ()
	test_dataset = ImageDataset (dataset1, dataset2, labels, path, transformer_test)
	test_dataloader = DataLoader (test_dataset, batch_size=32, num_workers=4, pin_memory=True)
	
	valid_file = open (path + "valid_list.csv", "r")
	valid_reader = csv.reader (valid_file)
	next (valid_reader)
	dataset1, dataset2, labels = [], [], []
	for line in valid_reader:
		dataset1.append ('/'.join (line[1].split ('\\')))
		dataset2.append ('/'.join (line[2].split ('\\')))
		labels.append (int (line[3]))
	valid_file.close ()
	valid_dataset = ImageDataset (dataset1, dataset2, labels, path, transformer_test)
	valid_dataloader = DataLoader (valid_dataset, batch_size=32, num_workers=4, pin_memory=True)
	
	return train_dataloader, test_dataloader, valid_dataloader, train_dataset, test_dataset
