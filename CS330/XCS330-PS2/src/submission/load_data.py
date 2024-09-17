import numpy as np
import os
import random
import torch
from torch.utils.data import IterableDataset
import time
import imageio


def get_images(paths, labels, nb_samples=None):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    image_labels = [
        (i, os.path.join(path, image))
        for i, path in zip(labels, paths)
        for image in sampler(os.listdir(path))
    ]

    return image_labels


class DataGenerator(IterableDataset):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(
        self,
        num_classes,
        num_samples_per_class,
        batch_type,
        config={},
        cache=True,
    ):
        """
        Args:
            num_classes: Number of classes for classification (N-way)
            num_samples_per_class: num samples to generate per class in one batch (K+1)
            batch_type: train/val/test
            config: data_folder - folder where the data is located
                    img_size - size of the input images
            cache: whether to cache the images loaded
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get("data_folder", "./omniglot_resized")
        self.img_size = config.get("img_size", (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [
            os.path.join(data_folder, family, character)
            for family in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, family))
            for character in os.listdir(os.path.join(data_folder, family))
            if os.path.isdir(os.path.join(data_folder, family, character))
        ]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[:num_train]
        self.metaval_character_folders = character_folders[num_train : num_train + num_val]
        self.metatest_character_folders = character_folders[num_train + num_val :]
        self.image_caching = cache
        self.stored_images = {}

        if batch_type == "train":
            self.folders = self.metatrain_character_folders
        elif batch_type == "val":
            self.folders = self.metaval_character_folders
        else:
            self.folders = self.metatest_character_folders

        self.sample_fn = random.sample

        self.shuffle_fn = np.random.shuffle

    def image_file_to_array(self, filename, dim_input):
        """
        Takes an image path and returns numpy array
        Args:
            filename: Image filename
            dim_input: Flattened shape of image
        Returns:
            1 channel image
        """
        if self.image_caching and (filename in self.stored_images):
            return self.stored_images[filename]
        image = imageio.imread(filename)  # misc.imread(filename)
        image = image.reshape([dim_input])
        image = image.astype(np.float32) / image.max()
        image = 1.0 - image
        if self.image_caching:
            self.stored_images[filename] = image
        return image

    def _sample(self, sample_fn=None, shuffle_fn=None):
        """
        Samples a batch for training, validation, or testing
        Args:
            sample_fn - pointer to a sampling function to be used
            shuffle_fn - pointer to a shuffling function to be used
        Returns:
            A tuple of (1) Image batch and (2) Label batch:
                1. image batch has shape [K+1, N, 784] and is a numpy array
                2. label batch has shape [K+1, N, N] and is a numpy array
            where K is the number of "shots", N is number of classes
        Note:
            1. The numpy functions np.random.shuffle and np.eye (for creating)
            one-hot vectors would be useful.

            2. For shuffling, remember to make sure images and labels are shuffled
            in the same order, otherwise the one-to-one mapping between images
            and labels may get messed up. Hint: there is a clever way to use
            np.random.shuffle here.
            
            3. The value for `self.num_samples_per_class` will be set to K+1 
            since for K-shot classification you need to sample K supports and 
            1 query.

            4. PyTorch uses float32 as default for representing model parameters. 
            You would need to return numpy arrays with the same datatype
        """

        if sample_fn is None:
            sample_fn = self.sample_fn

        if shuffle_fn is None:
            shuffle_fn = self.shuffle_fn

        #############################
        ### START CODE HERE ###
            # Sample N different characters from either the specified train, test, or validation folder.
            character_folders = sample_fn(self.folders, self.num_classes)

            # Load K+1 images per character and collect the associated labels, 
            #using K images per class for the support set and 1 image per class for the query set.
            labels = np.eye(self.num_classes)
            image_labels = get_images(paths=character_folders, labels=labels, nb_samples=self.num_samples_per_class)
            # Creating Support set and query set
            support_set = []
            query_set = []
            
            for image_class in range(0, self.num_classes):
                for class_image_num in range(0, self.num_samples_per_class):
                    if class_image_num == self.num_samples_per_class-1:
                        query_set.append(image_labels[(image_class * self.num_samples_per_class) + class_image_num])
                    else:
                        support_set.append(image_labels[(image_class * self.num_samples_per_class) + class_image_num])
            # Shuffle Query Set
            shuffle_fn(query_set)
            all_set = support_set + query_set

            image_batch = []
            label_batch = []
            for item in all_set:
                image_batch.append(item[1])
                label_batch.append(item[0])

            for i in range(0, len(image_batch)):
                image_batch[i] = self.image_file_to_array(image_batch[i], 784)

            # Format the data and return two tensors, one of flattened images with shape [K + 1, N, 784] and one of one-hot labels[K+1, N, N]
            image_batch = np.array([[image_batch[i * self.num_classes + j] for j in range(self.num_classes)] for i in range(self.num_samples_per_class)])
            label_batch = np.array([[label_batch[i * self.num_classes + j] for j in range(self.num_classes)] for i in range(self.num_samples_per_class)])
            output = (torch.from_numpy(image_batch).float(), torch.from_numpy(label_batch).float())
            return output
        ### END CODE HERE ###

    def __iter__(self):
        while True:
            yield self._sample()
