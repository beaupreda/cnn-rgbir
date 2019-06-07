import argparse
import math
import os
import random
import sys
from collections import defaultdict
from enum import IntEnum
from PIL import Image


class FileType(IntEnum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class InputParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def add_arguments(self, name, default, definition):
        self.parser.add_argument(name, default=default, help=definition)

    def get_arguments(self):
        return self.parser.parse_args()


class FileLogger:
    def __init__(self, names):
        train_file = open(names[FileType.TRAIN], 'w')
        val_file = open(names[FileType.VALIDATION], 'w')
        test_file = open(names[FileType.TEST], 'w')
        self.files = [train_file, val_file, test_file]

    def get_name(self, index):
        return self.files[index]

    def write(self, message, index):
        self.files[index].write(message)


class DataLoader:
    RGB = 'rgb'
    LWIR = 'lwir'
    IMAGES_TYPE = [RGB, LWIR]
    MIRROR = '_mirror'
    SEPARATOR = '/'

    def __init__(self, path, extension):
        self.path = path
        self.extension = extension
        self.train_names = defaultdict(list)
        self.val_names = defaultdict(list)
        self.test_names = defaultdict(list)
        self.file_logger = None

    def set_file_loader(self, names):
        self.file_logger = FileLogger(names)

    def get_train_names(self):
        return self.train_names

    def get_val_names(self):
        return self.val_names

    def get_test_names(self):
        return self.test_names

    def shuffle(self, data):
        assert len(data[DataLoader.RGB]) == len(data[DataLoader.LWIR]), 'number of elements in dict not the same!'
        indices = [i for i in range(len(data[DataLoader.RGB]))]
        random.seed(12)
        random.shuffle(indices)
        shuffled_data = defaultdict(list)
        for i in range(len(indices)):
            shuffled_data[DataLoader.RGB].append(data[DataLoader.RGB][indices[i]])
            shuffled_data[DataLoader.LWIR].append(data[DataLoader.LWIR][indices[i]])
        return shuffled_data

    def mirror(self, data):
        for image_type in DataLoader.IMAGES_TYPE:
            print('Mirror images for ' + image_type)
            i = 0
            maximum = len(data[image_type])
            for image_name in data[image_type]:
                if image_name.find('mirror') < 0:
                    print(str(i) + ' / ' + str(maximum))
                    i += 1
                    image = Image.open(image_name)
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    split_name = image_name.split('.')
                    name = split_name[0]
                    extension = '.' + split_name[1]
                    filename = os.path.join(name + '_mirror' + extension)
                    image.save(filename)
                else:
                    print('Mirror image already exists')                

    def load_data(self):
        pass

    def separate_data(self, data):
        pass

    def save_image(self, index, image_name, directory, filetype):
        if image_name.endswith('png'):
            image = Image.open(image_name)
            elements = image_name.split(StCharlesLoader.SEPARATOR)
            gt = ''
            mirror = '0'
            for e in elements[:len(elements)]:
                if e == StCharlesLoader.RECTIFIED_IMAGES:
                    continue
                if e == StCharlesLoader.RGB or e == StCharlesLoader.LWIR:
                    gt += e + StCharlesLoader.GT_DISP
                    continue
                if e.endswith(StCharlesLoader.MIRROR + '.png'):
                    e = e[:len(e) - (len(StCharlesLoader.MIRROR) + len('.png'))] + StCharlesLoader.YAML_EXTENSION
                    gt += e
                    mirror = '1'
                    continue
                if e.endswith('.png'):
                    gt += e[:len(e) - len('.png')] + StCharlesLoader.YAML_EXTENSION
                    continue
                gt += e + StCharlesLoader.SEPARATOR
            unique_name = str(index) + '.png'
            unique_name = os.path.join(directory, unique_name)
            image.save(unique_name)
            width, _ = image.size
            message = str(index) + ' ' + unique_name + ' ' + image_name + ' ' + gt + ' ' + mirror + ' ' + str(
                width) + '\n'
            self.file_logger.write(message, filetype)
        elif image_name.endswith('jpg'):
            image = Image.open(image_name)
            elements = image_name.split(BilodeauLoader.SEPARATOR)
            gt = ''
            mirror = '0'
            if elements[-1].endswith(BilodeauLoader.MIRROR + '.jpg'):
                mirror = '1'
            for e in elements[:len(elements) - 2]:
                gt += e + BilodeauLoader.SEPARATOR
            files = os.listdir(gt)
            gt_file = [os.path.join(gt, f) for f in files if f.endswith(BilodeauLoader.GT_EXTENSION)]
            unique_name = str(index) + '.png'
            unique_name = os.path.join(directory, unique_name)
            image.save(unique_name)
            width, _ = image.size
            message = str(index) + ' ' + unique_name + ' ' + image_name + ' ' + gt_file[0] + ' ' + mirror + ' ' + str(width) + '\n'
            self.file_logger.write(message, filetype)


class StCharlesLoader(DataLoader):
    VID04 = 'vid04'
    VID07 = 'vid07'
    VID08 = 'vid08'
    FOLDERS = [VID04, VID07, VID08]
    RECTIFIED_IMAGES = 'rectified_images_v2'
    YAML_EXTENSION = '.yml'
    GT_DISP = '_gt_disp/'

    def __init__(self, path, extension):
        super().__init__(path, extension)

    def mirror_images(self):
        data = defaultdict(dict)
        for folder in StCharlesLoader.FOLDERS:
            for image_type in StCharlesLoader.IMAGES_TYPE:
                directory = os.path.join(self.path, folder, image_type)
                if os.path.exists(directory):
                    filenames = os.listdir(directory)
                    filenames = [os.path.join(directory, filename) for filename in filenames if filename.endswith(self.extension)]
                    filenames.sort()
                    data[folder][image_type] = filenames
            self.mirror(data[folder])

    def load_data(self):
        data = defaultdict(dict)
        for folder in StCharlesLoader.FOLDERS:
            for image_type in StCharlesLoader.IMAGES_TYPE:
                directory = os.path.join(self.path, folder, image_type)
                if os.path.exists(directory):
                    filenames = os.listdir(directory)
                    filenames = [os.path.join(directory, filename) for filename in filenames if filename.endswith(self.extension)]
                    filenames.sort()
                    data[folder][image_type] = filenames
        self.separate_data(data)

    def separate_data(self, data):
        for image_type in StCharlesLoader.IMAGES_TYPE:
            self.train_names[image_type] = []
            self.train_names[image_type] += data[StCharlesLoader.VID04][image_type]
            self.train_names[image_type] += data[StCharlesLoader.VID07][image_type]
            self.train_names[image_type] += data[StCharlesLoader.VID08][image_type]
        self.train_names = self.shuffle(self.train_names)

    def save_image(self, index, image_name, directory, filetype):
        image = Image.open(image_name)
        elements = image_name.split(StCharlesLoader.SEPARATOR)
        gt = ''
        mirror = '0'
        for e in elements[:len(elements)]:
            if e == StCharlesLoader.RECTIFIED_IMAGES:
                continue
            if e == StCharlesLoader.RGB or e == StCharlesLoader.LWIR:
                gt += e + StCharlesLoader.GT_DISP
                continue
            if e.endswith(StCharlesLoader.MIRROR + self.extension):
                e = e[:len(e) - (len(StCharlesLoader.MIRROR) + len(self.extension))] + StCharlesLoader.YAML_EXTENSION
                gt += e
                mirror = '1'
                continue
            if e.endswith(self.extension):
                gt += e[:len(e) - len(self.extension)] + StCharlesLoader.YAML_EXTENSION
                continue
            gt += e + StCharlesLoader.SEPARATOR
        unique_name = str(index) + self.extension
        unique_name = os.path.join(directory, unique_name)
        image.save(unique_name)
        width, _ = image.size
        message = str(index) + ' ' + unique_name + ' ' + image_name + ' ' + gt + ' ' + mirror + ' ' + str(width) + '\n'
        self.file_logger.write(message, filetype)


class BilodeauLoader(DataLoader):
    VID01 = 'vid1'
    VID02 = 'vid2'
    VID03 = 'vid3'
    FOLDERS = [VID01, VID02, VID03]
    CUT01 = 'cut1'
    CUT02 = 'cut2'
    CUTS = [CUT01, CUT02]
    PERSON01 = '1Person'
    PERSON02 = '2Person'
    PERSON03 = '3Person'
    PERSON04 = '4Person'
    PERSON05 = '5Person'
    PERSONS = [PERSON01, PERSON02, PERSON03, PERSON04, PERSON05]
    VIDEO_FRAMES = 'videoFrames'
    VIDEO_FRAMES_ALT = 'VideoFrame'
    VIS = 'Vis'
    IR = 'IR'
    GT_EXTENSION = 'n.txt'

    def __init__(self, path, extension, fold):
        super().__init__(path, extension)
        self.fold = int(fold)

    def mirror_images(self):
        data = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for folder in BilodeauLoader.FOLDERS:
            if folder != BilodeauLoader.VID02:
                for person in BilodeauLoader.PERSONS:
                    directory = ''
                    if folder == BilodeauLoader.VID03 and person == BilodeauLoader.PERSON02:
                        directory = os.path.join(self.path, folder, person, BilodeauLoader.VIDEO_FRAMES_ALT)
                    else:
                        directory = os.path.join(self.path, folder, person, BilodeauLoader.VIDEO_FRAMES)
                    if os.path.exists(directory):
                        filenames = os.listdir(directory)
                        filenames = [os.path.join(directory, filename) for filename in filenames if filename.endswith(self.extension)]
                        filenames.sort()
                        rgb, lwir = self.split(filenames)
                        data[folder][person][BilodeauLoader.RGB] = rgb
                        data[folder][person][BilodeauLoader.LWIR] = lwir
                    self.mirror(data[folder][person])
            else:
                for cut in BilodeauLoader.CUTS:
                    for person in BilodeauLoader.PERSONS:
                        directory = os.path.join(self.path, folder, cut, person, BilodeauLoader.VIDEO_FRAMES)
                        if os.path.exists(directory):
                            filenames = os.listdir(directory)
                            filenames = [os.path.join(directory, filename) for filename in filenames if filename.endswith(self.extension)]
                            filenames.sort()
                            rgb, lwir = self.split(filenames)
                            data[folder][person][BilodeauLoader.RGB] += rgb
                            data[folder][person][BilodeauLoader.LWIR] += lwir
                        self.mirror(data[folder][person])

    def load_data(self):
        data = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for folder in BilodeauLoader.FOLDERS:
            if folder != BilodeauLoader.VID02:
                for person in BilodeauLoader.PERSONS:
                    directory = ''
                    if folder == BilodeauLoader.VID03 and person == BilodeauLoader.PERSON02:
                        directory = os.path.join(self.path, folder, person, BilodeauLoader.VIDEO_FRAMES_ALT)
                    else:
                        directory = os.path.join(self.path, folder, person, BilodeauLoader.VIDEO_FRAMES)
                    if os.path.exists(directory):
                        filenames = os.listdir(directory)
                        filenames = [os.path.join(directory, filename) for filename in filenames if filename.endswith(self.extension)]
                        filenames.sort()
                        rgb, lwir = self.split(filenames)
                        data[folder][person][BilodeauLoader.RGB] = rgb
                        data[folder][person][BilodeauLoader.LWIR] = lwir
            else:
                for cut in BilodeauLoader.CUTS:
                    for person in BilodeauLoader.PERSONS:
                        directory = os.path.join(self.path, folder, cut, person, BilodeauLoader.VIDEO_FRAMES)
                        if os.path.exists(directory):
                            filenames = os.listdir(directory)
                            filenames = [os.path.join(directory, filename) for filename in filenames if filename.endswith(self.extension)]
                            filenames.sort()
                            rgb, lwir = self.split(filenames)
                            data[folder][person][BilodeauLoader.RGB] += rgb
                            data[folder][person][BilodeauLoader.LWIR] += lwir
        self.separate_data(data)

    def separate_data(self, data):
        for image_type in BilodeauLoader.IMAGES_TYPE:
            self.train_names[image_type] = []
            self.val_names[image_type] = []
            self.test_names[image_type] = []
            if self.fold == 1:
                train_val = []
                for person in BilodeauLoader.PERSONS:
                    train_val += data[BilodeauLoader.VID01][person][image_type]
                    train_val += data[BilodeauLoader.VID02][person][image_type]
                    self.test_names[image_type] += data[BilodeauLoader.VID03][person][image_type]
                self.val_names[image_type] += train_val[:60]
                self.train_names[image_type] += train_val[60:]
            elif self.fold == 2:
                train_val = []
                for person in BilodeauLoader.PERSONS:
                    train_val += data[BilodeauLoader.VID02][person][image_type]
                    train_val += data[BilodeauLoader.VID03][person][image_type]
                    self.test_names[image_type] += data[BilodeauLoader.VID01][person][image_type]
                self.val_names[image_type] += train_val[:60]
                self.train_names[image_type] += train_val[60:]
            elif self.fold == 3:
                train_val = []
                for person in BilodeauLoader.PERSONS:
                    train_val += data[BilodeauLoader.VID01][person][image_type]
                    train_val += data[BilodeauLoader.VID03][person][image_type]
                    self.test_names[image_type] += data[BilodeauLoader.VID02][person][image_type]
                self.val_names[image_type] += train_val[:90]
                self.train_names[image_type] += train_val[90:]
        self.train_names = self.shuffle(self.train_names)
        self.val_names = self.shuffle(self.val_names)
        self.test_names = self.shuffle(self.test_names)

    def split(self, filenames):
        rgb, lwir = ([] for i in range(2))
        for filename in filenames:
            name = filename.split(BilodeauLoader.SEPARATOR)[-1]
            if name.find(BilodeauLoader.VIS) >= 0:
                rgb.append(filename)
            elif name.find(BilodeauLoader.IR) >= 0:
                lwir.append(filename)
        return rgb, lwir

    def save_image(self, index, image_name, directory, filetype):
        image = Image.open(image_name)
        elements = image_name.split(BilodeauLoader.SEPARATOR)
        gt = ''
        mirror = '0'
        if elements[-1].endswith(BilodeauLoader.MIRROR + self.extension):
            mirror = '1'
        for e in elements[:len(elements) - 2]:
            gt += e + BilodeauLoader.SEPARATOR
        files = os.listdir(gt)
        gt_file = [os.path.join(gt, f) for f in files if f.endswith(BilodeauLoader.GT_EXTENSION)]
        unique_name = str(index) + self.extension
        unique_name = os.path.join(directory, unique_name)
        image.save(unique_name)
        width, _ = image.size
        message = str(index) + ' ' + unique_name + ' ' + image_name + ' ' + gt_file[0] + ' ' + mirror + ' ' + str(width) + '\n'
        self.file_logger.write(message, filetype)


def main():
    PNG_EXTENSION = '.png'
    JPG_EXTENSION = '.jpg'

    input_parser = InputParser()
    input_parser.add_arguments('--stcharles_dir', '/store/dabeaq/datasets/litiv/stcharles2018-v04/rectified_images_v2/', 'Directory containing rectified images for all videos (execute rectification)')
    input_parser.add_arguments('--bilodeau_dir', '/store/dabeaq/datasets/bilodeauIR/Dataset/', 'Directory containing rectified images for all videos (already rectified)')
    input_parser.add_arguments('--out_dir', '/home/travail/dabeaq/litiv/masters/pbvs2019/cnn-rgbir/dataset', 'Location of the output images')
    input_parser.add_arguments('--fold', '1', 'Fold to generate the data')
    args = input_parser.get_arguments()

    stcharles = StCharlesLoader(args.stcharles_dir, PNG_EXTENSION)
    bilodeau = BilodeauLoader(args.bilodeau_dir, JPG_EXTENSION, args.fold)
    stcharles.mirror_images()
    bilodeau.mirror_images()
    stcharles.load_data()
    bilodeau.load_data()
    
    output = args.out_dir + args.fold
    if not os.path.exists(output):
        os.mkdir(output)
    
    data_loader = DataLoader(None, None)
    train = {}
    val = {}
    test = {}
    for image_type in DataLoader.IMAGES_TYPE:
        train[image_type] = stcharles.train_names[image_type] + bilodeau.train_names[image_type]
        val[image_type] = stcharles.val_names[image_type] + bilodeau.val_names[image_type]
        test[image_type] = stcharles.test_names[image_type] + bilodeau.test_names[image_type]
    train = stcharles.shuffle(train)
    for image_type in DataLoader.IMAGES_TYPE:
        intersect_tr_val = list(set(train[image_type]).intersection(set(val[image_type])))
        intersect_tr_test = list(set(train[image_type]).intersection(set(test[image_type])))
        intersect_val_test = list(set(val[image_type]).intersection(set(test[image_type])))
        if intersect_tr_val or intersect_tr_test or intersect_val_test:
            print('PROBLEM')
        else:
            print('Data separation is OK (no intersections)')
    folders = ['train', 'validation', 'test']
    maps = []
    folders_both = []
    for folder in folders:
        current_dir = os.path.join(output, '{}'.format(folder))
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        maps.append(current_dir + '/map.txt')
        folders_both.append(current_dir)
    data_loader.set_file_loader(maps)
    for image_type in StCharlesLoader.IMAGES_TYPE:
        counter = 0
        directory = os.path.join(folders_both[FileType.TRAIN], image_type)
        if not os.path.exists(directory):
            os.mkdir(directory)
        print('Creating train folder ({})...'.format(image_type))
        for filename in train[image_type]:
            data_loader.save_image(counter, filename, directory, FileType.TRAIN)
            counter += 1
        directory = os.path.join(folders_both[FileType.VALIDATION], image_type)
        if not os.path.exists(directory):
            os.mkdir(directory)
        print('Creating validation folder ({})...'.format(image_type))
        for filename in val[image_type]:
            data_loader.save_image(counter, filename, directory, FileType.VALIDATION)
            counter += 1
        directory = os.path.join(folders_both[FileType.TEST], image_type)
        if not os.path.exists(directory):
            os.mkdir(directory)
        print('Creating test folder ({})...'.format(image_type))
        for filename in test[image_type]:
            data_loader.save_image(counter, filename, directory, FileType.TEST)
            counter += 1


if __name__ == '__main__':
    main()
