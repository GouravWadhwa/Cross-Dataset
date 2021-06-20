import tensorflow as tf

from PIL import Image
import numpy as np
import random

class TrainDataset () :
    def __init__ (self, train_file='train_file.txt', num_classes=7, batch_size=1) :
        self.train_file = train_file
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.train_dataset = []
        self.current = 0

        datapoints = open (self.train_file)
        for data in datapoints.readlines() :
            self.train_dataset.append (data[:-1])

        random.shuffle (self.train_dataset)

        self.dataset_size = len (self.train_dataset)

    def __iter__ (self) :
        self.current = 0
        return self

    def __len__ (self) :
        return self.dataset_size

    def get_data (self, index) :
        data = self.train_dataset[self.current]

        type, x_path, encoded_y = data.split (" ")
        
        is_labelled = 1 if type == 'labelled' else 0

        x = Image.open (x_path).convert ("RGB").resize ((227, 227))
        x = np.array (x)
        x = tf.keras.applications.resnet50.preprocess_input (x)

        y = np.zeros (self.num_classes)
        y[int (encoded_y)] = 1
        
        return (is_labelled, x, y)

    def __next__ (self) :
        label = []
        x = []
        y = []
        
        for i in range (self.batch_size) :
            if self.current != self.dataset_size :
                data = self.get_data (self.current)
                
                label.append (data[0])
                x.append (data[1])
                y.append (data[2])

                self.current += 1
            else :
                if x != [] :
                    return tf.convert_to_tensor (label, tf.float32), tf.convert_to_tensor (x, tf.float32), tf.convert_to_tensor (y, tf.float32)
                else :
                    raise StopIteration
        return tf.convert_to_tensor (label, tf.float32), tf.convert_to_tensor (x, tf.float32), tf.convert_to_tensor (y, tf.float32)

class ValDataset () :
    def __init__ (self, train_file='val_file.txt', num_classes=7, batch_size=1) :
        self.train_file = train_file
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.train_dataset = []
        self.current = 0

        datapoints = open (self.train_file)
        for data in datapoints.readlines() :
            self.train_dataset.append (data[:-1])

        random.shuffle (self.train_dataset)

        self.dataset_size = len (self.train_dataset)

    def __iter__ (self) :
        self.current = 0
        return self

    def __len__ (self) :
        return self.dataset_size

    def __next__ (self) :
        if self.current != self.dataset_size :
            data = self.train_dataset[self.current]

            _, x_path, encoded_y = data.split (" ")

            x = Image.open (x_path).convert ("RGB").resize ((227, 227))
            x = np.array (x)
            x = tf.keras.applications.resnet50.preprocess_input (x)

            y = np.zeros (self.num_classes)
            y[int (encoded_y)] = 1
            
            self.current += 1

            return (tf.convert_to_tensor ([x], tf.float32), tf.convert_to_tensor ([y], tf.float32))
        else :
            raise StopIteration

class TestDataset () :
    def __init__ (self, train_file='test_file.txt', num_classes=7, batch_size=1) :
        self.train_file = train_file
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.train_dataset = []
        self.current = 0

        datapoints = open (self.train_file)
        for data in datapoints.readlines() :
            self.train_dataset.append (data[:-1])

        random.shuffle (self.train_dataset)

        self.dataset_size = len (self.train_dataset)

    def __iter__ (self) :
        self.current = 0
        return self

    def __len__ (self) :
        return self.dataset_size

    def __next__ (self) :
        if self.current != self.dataset_size :
            data = self.train_dataset[self.current]

            _, x_path, encoded_y = data.split (" ")

            x = Image.open (x_path).convert ("RGB").resize ((227, 227))
            x = np.array (x)
            x = tf.keras.applications.resnet50.preprocess_input (x)

            y = np.zeros (self.num_classes)
            y[int (encoded_y)] = 1
            
            self.current += 1

            return (tf.convert_to_tensor ([x], tf.float32), tf.convert_to_tensor ([y], tf.float32))
        else :
            raise StopIteration
