import os
from torch.utils.data import Dataset
from PIL import Image
import random
from matplotlib import image
import cv2
import numpy as np

from scipy.signal import convolve2d as conv2


class ExemplarDataset(Dataset):

    def __init__(self, data, transform,transform_style=None):

        labels = []
        data_path = []
        for y, P_y in enumerate(data):
            label = [y] * len(P_y)
            labels.extend(label)
            for P_y_i in P_y:
                data_path.append(P_y_i)
        
        self.data_path = data_path
        self.transform = transform
        self.transform_style = transform_style
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        path = self.data_path[idx]           
        sample_ori = cv2.imread(path)

        label = self.labels[idx]

        sample = self.transform(sample_ori)
        sample_style = self.transform_style(sample_ori)

        return sample,sample_style, label


class Exemplar1K(Dataset):
    def __init__(self, data_root, classes, num_samples, transform):
        self.transform = transform

        self.sample_filepaths = []

        self.train = train
        self.train_sample_cls = []
        self.test_sample_cls = []

        self.train_data = []
        self.test_data = []

        f = open('./data/class_folder_list.txt', 'r')
        lines=f.readlines()
        dir_list=[]
        for x in lines:
            dir_list.append(x.split(' ')[0])
         
        np.random.seed(1993)
        cls_list = [i for i in range(1000)]
        np.random.shuffle(cls_list)
        dir_list = [dir_list[i] for i in cls_list]

        for cls_idx, cls in enumerate(dir_list):

            cls_folder = os.path.join(data_root, cls)

            if cls_idx in classes:
                sample_idx = 0
                for sample in os.listdir(cls_folder):
                    sample_filepath = os.path.join(cls_folder, sample)
                    self.sample_filepaths.append(sample_filepath)
                    if train:
                        self.train_sample_cls.append(cls_idx)
                    else:
                        self.test_sample_cls.append(cls_idx)
        
    def __len__(self):
        if self.train:
            return len(self.train_sample_cls)
        else:
            return len(self.test_sample_cls)

    def __getitem__(self, idx):
    
        if self.train:
            img = cv2.imread(self.sample_filepaths[idx])
            img = self.transform(img)
            label = self.train_sample_cls[idx]
            return img, img, label
        else:
            img = cv2.imread(self.sample_filepaths[idx])
            img = self.transform(img)
            label = self.test_sample_cls[idx]
            return img, img, label
            

    def get_image_class(self, label):
        list_label = []
        list_label = [np.array(cv2.imread(self.sample_filepaths[idx])) for idx, k in enumerate(self.train_sample_cls) if k==label]
        return np.array(list_label)

    def append(self, images, labels):
        """Append dataset with images and labels
        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_sample_cls = self.train_sample_cls + labels


class ImageNet1K(Dataset):
    def __init__(self, data_root, train, classes, transform):
        self.transform = transform

        self.sample_filepaths = []

        self.train = train
        self.train_sample_cls = []
        self.test_sample_cls = []

        self.train_data = []
        self.test_data = []

        f = open('./data/class_folder_list.txt', 'r')
        lines=f.readlines()
        dir_list=[]
        for x in lines:
            dir_list.append(x.split(' ')[0])
         
        np.random.seed(1993)
        cls_list = [i for i in range(1000)]
        np.random.shuffle(cls_list)
        dir_list = [dir_list[i] for i in cls_list]
        self.cls_names = dir_list
        for cls_idx, cls in enumerate(dir_list):

            cls_folder = os.path.join(data_root, cls)

            if cls_idx in classes:
                for sample in os.listdir(cls_folder):
                    sample_filepath = os.path.join(cls_folder, sample)
                    self.sample_filepaths.append(sample_filepath)
                    if train:
                        self.train_sample_cls.append(cls_idx)
                    else:
                        self.test_sample_cls.append(cls_idx)
        
    def __len__(self):
        if self.train:
            return len(self.train_sample_cls)
        else:
            return len(self.test_sample_cls)

    def __getitem__(self, idx):
    
        if self.train:
            img = cv2.imread(self.sample_filepaths[idx])
            img = self.transform(img)
            label = self.train_sample_cls[idx]
            return img, img, label
        else:
            img = cv2.imread(self.sample_filepaths[idx])
            img = self.transform(img)
            label = self.test_sample_cls[idx]
            return img, img, label
            

    def get_image_class(self, label):
        list_label = []
        list_label = [np.array(cv2.imread(self.sample_filepaths[idx])) for idx, k in enumerate(self.train_sample_cls) if k==label]
        return np.array(list_label)
    
    def get_cls_names(self):
        return self.cls_names

    def append(self, images, labels):
        """Append dataset with images and labels
        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_sample_cls = self.train_sample_cls + labels

class ImageNet100(Dataset):
    def __init__(self, data_root, train, classes, transform, transform_style=None):
        self.transform = transform
        self.transform_style = transform_style

        self.sample_filepaths = []

        self.train = train
        self.train_sample_cls = []
        self.test_sample_cls = []

        self.train_data = []
        self.test_data = []

        # ImageNet-100
        f=open('./data/imagenet100_s1993.txt',"r")
        lines=f.readlines()
        dir_list=[]
        for x in lines:
            #print(x.split('\n')[0])
            dir_list.append(x.split(' ')[0])

        np.random.seed(1993)
        cls_list = [i for i in range(100)]
        np.random.shuffle(cls_list)
        dir_list = [dir_list[i] for i in cls_list]
        self.cls_names = dir_list
        for cls_idx, cls in enumerate(dir_list):
            if cls_idx ==100:       # imagenet-100
                break

            cls_folder = os.path.join(data_root, cls)

            if cls_idx in classes:
                for sample in os.listdir(cls_folder):
                    sample_filepath = os.path.join(cls_folder, sample)
                    self.sample_filepaths.append(sample_filepath)
                    if train:
                        self.train_sample_cls.append(cls_idx)
                    else:
                        self.test_sample_cls.append(cls_idx)

    def __len__(self):
        if self.train:
            return len(self.train_sample_cls)
        else:
            return len(self.test_sample_cls)

    def __getitem__(self, idx):
        if self.train:
            path = self.sample_filepaths[idx]
            img_ori = cv2.imread(path)
            img = self.transform(img_ori)
            img_style = self.transform_style(img_ori)

            label = self.train_sample_cls[idx]
            return img, img_style, label
        else:
            img = cv2.imread(self.sample_filepaths[idx])
            img = self.transform(img)
            label = self.test_sample_cls[idx]
            return img, label
    
    def get_image_class(self, label):
        list_label = []
        for idx, k in enumerate(self.train_sample_cls):
            if k==label:
                path = self.sample_filepaths[idx]
                #img = np.array(cv2.imread(path))
                list_label.append(path)
        return list_label#np.array(list_label)
    
    def get_cls_names(self):
        return self.cls_names

    def append(self, images, labels):
        """Append dataset with images and labels
        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_sample_cls = self.train_sample_cls + labels


class ImageNet100C(Dataset):
    def __init__(self, data_root, transform):
        self.transform = transform
        self.sample_filepaths = []
        self.test_sample_cls = []

        # ImageNet-100
        f=open('./data/imagenet100_s1993.txt',"r")
        lines=f.readlines()
        dir_list=[]
        for x in lines:
            #print(x.split('\n')[0])
            dir_list.append(x.split(' ')[0])

        np.random.seed(1993)
        cls_list = [i for i in range(100)]
        np.random.shuffle(cls_list)
        dir_list = [dir_list[i] for i in cls_list]

        for ser_level in ['1','2','3','4','5']:
            data_path = os.path.join(data_root, ser_level)
            for cls_idx, cls in enumerate(dir_list):
                if cls_idx ==100:       # imagenet-100
                    break

                cls_folder = os.path.join(data_path, cls)
                for sample in os.listdir(cls_folder):
                    sample_filepath = os.path.join(cls_folder, sample)
                    self.sample_filepaths.append(sample_filepath)
                    self.test_sample_cls.append(cls_idx)

    def __len__(self):
        return len(self.test_sample_cls)

    def __getitem__(self, idx):
        img = cv2.imread(self.sample_filepaths[idx])
        img = self.transform(img)
        label = self.test_sample_cls[idx]
        return img, label

class ImageNet100AR(Dataset):
    def __init__(self, data_root, transform):
        self.transform = transform
        self.sample_filepaths = []
        self.test_sample_cls = []

        # ImageNet-100
        f=open('./data/imagenet100_s1993.txt',"r")
        lines=f.readlines()
        dir_list=[]
        for x in lines:
            #print(x.split('\n')[0])
            dir_list.append(x.split(' ')[0])

        np.random.seed(1993)
        cls_list = [i for i in range(100)]
        np.random.shuffle(cls_list)
        dir_list = [dir_list[i] for i in cls_list]

        for cls_idx, cls in enumerate(dir_list):
            if cls_idx ==100:       # imagenet-100
                break

            cls_folder = os.path.join(data_root, cls)
            if os.path.exists(cls_folder):
                for sample in os.listdir(cls_folder):
                    sample_filepath = os.path.join(cls_folder, sample)
                    self.sample_filepaths.append(sample_filepath)
                    self.test_sample_cls.append(cls_idx)

    def __len__(self):
        return len(self.test_sample_cls)

    def __getitem__(self, idx):
        img = cv2.imread(self.sample_filepaths[idx])
        img = self.transform(img)
        label = self.test_sample_cls[idx]
        return img, label

class ShapeBias(Dataset):
    def __init__(self, data_root, transform):
        self.transform = transform

        self.test_sample_cls = []
        self.sample_filepaths =[]
        for cls in ['cat','clock','dog','keyboard','truck','bird','boat']:
            cls_folder = os.path.join(data_root, cls)
            for sample in os.listdir(cls_folder):
                sample_filepath = os.path.join(cls_folder, sample)
                self.sample_filepaths.append(sample_filepath)
                self.test_sample_cls.append(sample[:-3])

    def __len__(self):
        return len(self.test_sample_cls)

    def __getitem__(self, idx):
        img = cv2.imread(self.sample_filepaths[idx])
        img = self.transform(img)
        label = self.test_sample_cls[idx]
        return img, label
    
# main
if __name__ == "__main__":
    test_set = ImageNet100AR("/data/temp_zenglin/data/imagenet-r", transform=None)
    print(len(test_set))

    cnt = 0
    for corr_type in ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness',
                      'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']:
        valdir = os.path.join("/data/temp_zenglin/data/imagenet-c", corr_type)
        test_set = ImageNet100C(valdir, transform=None)
        cnt+=len(test_set)
    print(cnt)


