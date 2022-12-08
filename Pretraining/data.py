import torch
from tiger.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import pandas as pd
import torchvision.transforms as T
from utils import box_xyxy_to_cxcywh

def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.shape[1:], size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.shape[1:], image.shape[1:]))
    #print(ratios)
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        #print('orig:', boxes)
        #print(torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height]))
        scaled_boxes = torch.as_tensor(boxes) * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        #print('scaled: ', scaled_boxes)
        #print('rescaled im:', rescaled_image.shape[1:])
        target["boxes"] = scaled_boxes
        img_h, img_w = rescaled_image.shape[1:]
        scaled_boxes = torch.as_tensor(boxes) / torch.as_tensor([img_w, img_h, img_w, img_h])
        target['scaled boxes'] = box_xyxy_to_cxcywh(scaled_boxes)
        #print(target['scaled boxes'])

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


class DeepLesion(Dataset):
    def __init__(self, annotations_dir, img_dir, split='train'):
        groups = pd.read_csv(annotations_dir+'DL_info_v4.csv', sep=';').groupby('Train_Val_Test')
        if split == 'train':
            self.img_labels = groups.get_group(1)
        elif split == 'val':
            self.img_labels = groups.get_group(2)
        elif split == 'test':
            self.img_labels = groups.get_group(3)
        else:
            print("Error: split not found!")

        #self.img_labels = pd.read_csv(annotations_dir + 'DL_info_v3.csv', sep=';')

        self.img_dir = img_dir
        #self.transform = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = T.Normalize([-2682.8042, -2739.8506, -2725.4387], [2187.5068, 2235.9980, 2225.6160])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_filename = self.img_labels.iloc[idx]['File_name']
        img_path = self.img_dir + image_filename.replace('png', 'mha')
        image, header = read_image(img_path)
        boxes = eval(self.img_labels.iloc[idx]['Bounding_boxes'])
        boxes = [eval(x) for x in boxes]
        #append box in coner to make loss calculation easier
        if len(boxes) == 1:
            boxes.append([0, 0, 0, 0])
        if len(boxes) == 3:
            # TODO: better solution?
            boxes = boxes[:2]
            
        targets = {'image_id': image_filename, 'boxes': boxes}
        
        #print("shape: ", image.shape, " idx: ", idx)
        targets['orig_size'] = image.shape[1:]
        image = torch.from_numpy(image)
        image, targets = resize(image, targets, 512)
        image = self.transform(image.float())

        return image, targets
        
        
class DeepLesion_eval(Dataset):
    def __init__(self, annotations_dir, img_dir, split='train'):
        groups = pd.read_csv(annotations_dir + 'DL_info_v4.csv', sep=';').groupby('Train_Val_Test')
        if split == 'train':
            self.img_labels = groups.get_group(1)
        elif split == 'val':
            self.img_labels = groups.get_group(2)
        elif split == 'test':
            self.img_labels = groups.get_group(3)
        else:
            print("Error: split not found!")

        self.img_dir = img_dir
        self.transform = T.Normalize([-2682.8042, -2739.8506, -2725.4387], [2187.5068, 2235.9980, 2225.6160])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_filename = self.img_labels.iloc[idx]['File_name']
        img_path = self.img_dir + image_filename.replace('png', 'mha')
        image, header = read_image(img_path)
        boxes = eval(self.img_labels.iloc[idx]['Bounding_boxes'])
        boxes = [eval(x) for x in boxes]

        targets = {'boxes': boxes, 'image_id': idx, 'sentence_id': idx}

        # print("shape: ", image.shape, " idx: ", idx)
        targets['orig_size'] = image.shape[1:]
        image = torch.from_numpy(image)
        image, targets = resize(image, targets, 512)
        image = self.transform(image.float())

        return image, targets


class LesionData(Dataset):
    def __init__(self, annotations_dir, path_chansey, split='train'):
        if split == 'train':
            self.img_labels = pd.read_csv(annotations_dir+'Train_slice_list.csv', sep=';', decimal=',')
            
        elif split == 'val':
            self.img_labels = pd.read_csv(annotations_dir+'Val_slice_list.csv', sep=';', decimal=',')
        elif split == 'test':
            self.img_labels = pd.read_csv(annotations_dir+'Test_slice_list.csv', sep=';', decimal=',')
        elif split == 'all':
            img_labels_train = pd.read_csv(annotations_dir+'Train_slice_list.csv', sep=';', decimal=',')
            img_labels_val = pd.read_csv(annotations_dir+'Val_slice_list.csv', sep=';', decimal=',')
            img_labels_test = pd.read_csv(annotations_dir+'Test_slice_list.csv', sep=';', decimal=',')
            self.img_labels = pd.concat([img_labels_train, img_labels_val, img_labels_test])
            
        self.img_labels.dropna(inplace=True, how='all')

        self.transform = T.Normalize([0.9212, 0.9267, 0.9243], [0.3182, 0.3113, 0.3128])
        self.path_chansey = path_chansey

    def get_img_path(self, data_origin):
        if data_origin == 'Data/Final_dataset/matched_lesions_with_types_NEW_2.csv':
            return 'Data/FromReports/mha/'
        elif data_origin == 'Data/Trac_data_new/matched_lesions_with_types.csv':
            return 'radboudumc_data_v2/mha/trac/'
        elif data_origin == 'Data/GSPS/matched_lesions_with_types.csv':
            return 'radboudumc_data_v2/mha/gsps/'
        else:
            print("error: ", data_origin, " path not found")
            return None
    
    def __len__(self):
        return len(self.img_labels)

    #tartget is a dictionary containging the keys 'positive map', 'boxes', 'orig_size'
    # 'positive map': matrix of bbox x tokens, true if token belongs to box
    # 'boxes': tensor of bounding box coordinates
    # 'orig_size': original image size
    def __getitem__(self, idx):
        image_id = self.img_labels.iloc[idx]['serie']
        image_slice = self.img_labels.iloc[idx]['ima']
        image_filename = str(self.img_labels.iloc[idx]['filename'])
        #img_dir = self.img_labels.iloc[idx]['path']
        img_origin = str(self.img_labels.iloc[idx]['data_origin'])
        img_path = self.path_chansey + self.get_img_path(img_origin) + image_filename
        image, header = read_image(img_path)
        #image = []
        nr_boxes = int(self.img_labels.iloc[idx]['number of boxes'])
        boxes = eval(str(self.img_labels.iloc[idx]['bboxes']).replace('(','').replace(')',''))
        #boxes = [[b[0][0], b[0][1], b[1][0], b[1][1]] for b in boxes]
        #print(boxes)
        boxes = [[b[0], b[3], b[2], b[1]] if b[1] > b[3] else b for b in boxes] 
        boxes = [[b[2], b[1], b[0], b[3]] if b[0] > b[2] else b for b in boxes]#switch coordinates if y1 greater than y2 (due to some unknown error)
        type_label = eval(self.img_labels.iloc[idx]['targets'])
        #print(type_label, type(type_label), type(nr_boxes))

        sentence = self.img_labels.iloc[idx]['sentence']
        #print(type_label)

        type_label_words = [sentence[x:y] for x, y in type_label]
        #print(type_label_words)
        type_label = [type_label for _ in range(nr_boxes)]

        targets = {'image_id': image_id, 'sentence_id': idx, 'sentence': sentence, 'boxes': boxes, 'target': type_label, 'labels':type_label_words}
        targets['orig_size'] = image.shape[:-1]
        #print('orig_size: ', image.shape[:-1])

        image = image.transpose()
        image = torch.from_numpy(image)
        
        if image.shape[0] != 3:
            print('image: ', idx, image.shape)
            print(img_path)
        image, targets = resize(image, targets, 512)
        image = self.transform(image.float())

        return image, targets
        
    """    
    def __init__(self, annotations_dir, img_dir):
        self.img_labels = pd.read_csv(annotations_dir+'slice_file_list.csv')
        #self.slice_data = pd.read_csv(img_dir + 'slice_data.csv', index_col=0)

        #self.transform = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = T.Normalize([-2682.8042, -2739.8506, -2725.4387], [2187.5068, 2235.9980, 2225.6160])
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    #tartget is a dictionary containging the keys 'positive map', 'boxes', 'orig_size'
    # 'positive map': matrix of bbox x tokens, true if token belongs to box
    # 'boxes': tensor of bounding box coordinates
    # 'orig_size': original image size
    def __getitem__(self, idx):
        image_id = self.img_labels.iloc[idx]['serie']
        image_filename = self.img_labels.iloc[idx]['filename']
        img_path = self.img_dir + image_filename + '.mha'
        image, header = read_image(img_path)

        boxes = eval(self.img_labels.iloc[idx]['bboxes'])
        

        targets = {'image_id': image_id, 'boxes': boxes}
        targets['orig_size'] = image.shape[:-1]
        image = image.transpose()
        image = torch.from_numpy(image)
        image, targets = resize(image, targets, 512)
        image = self.transform(image.float())

        return image, targets
"""

class UnmatchedLesionData(Dataset):
    def __init__(self, annotations_dir, path_chansey, split='train', portion=1):
        if split == 'train':
            self.img_labels = pd.read_csv(annotations_dir+'Train_slice_list.csv', sep=';', decimal=',')
            
        elif split == 'val':
            self.img_labels = pd.read_csv(annotations_dir+'Val_slice_list.csv', sep=';', decimal=',')
        elif split == 'test':
            self.img_labels = pd.read_csv(annotations_dir+'Test_slice_list.csv', sep=';', decimal=',')
        elif split == 'all':
            img_labels_train = pd.read_csv(annotations_dir+'Train_slice_list.csv', sep=';', decimal=',')
            img_labels_val = pd.read_csv(annotations_dir+'Val_slice_list.csv', sep=';', decimal=',')
            img_labels_test = pd.read_csv(annotations_dir+'Test_slice_list.csv', sep=';', decimal=',')
            self.img_labels = pd.concat([img_labels_train, img_labels_val, img_labels_test])
            
            
        self.img_labels.dropna(inplace=True, how='all')
        
        if split=='train' and portion < 1:
            self.img_labels = self.img_labels.sample(frac = portion) #randomly select part of the data

        self.transform = T.Normalize([-668.5143, -668.2419, -668.0019], [696.9041, 697.3984, 697.8586])
        self.path_chansey = path_chansey
        self.img_path = 'Data/UnmatchedGSPS/unmatched/'
        self.spilt = split
    
    def __len__(self):
        return len(self.img_labels)

    #tartget is a dictionary containging the keys 'positive map', 'boxes', 'orig_size'
    # 'positive map': matrix of bbox x tokens, true if token belongs to box
    # 'boxes': tensor of bounding box coordinates
    # 'orig_size': original image size
    def __getitem__(self, idx):
        image_id = self.img_labels.iloc[idx]['serie']
        image_slice = self.img_labels.iloc[idx]['ima']
        image_filename = str(self.img_labels.iloc[idx]['filename'])
        img_path = self.path_chansey + self.img_path + image_filename
        image, header = read_image(img_path)
        #image = []
        nr_boxes = int(self.img_labels.iloc[idx]['number of boxes'])
        boxes = eval(str(self.img_labels.iloc[idx]['bboxes']).replace('(','').replace(')',''))
        #boxes = [[b[0][0], b[0][1], b[1][0], b[1][1]] for b in boxes]
        #print(boxes)
        boxes = [[b[0], b[3], b[2], b[1]] if b[1] > b[3] else b for b in boxes] 
        boxes = [[b[2], b[1], b[0], b[3]] if b[0] > b[2] else b for b in boxes]#switch coordinates if y1 greater than y2 (due to some unknown error)
        if self.spilt == 'train' or self.spilt == 'val':
            if len(boxes) == 1:
                boxes.append([0, 0, 0, 0])
            if len(boxes) == 3:
                 # TODO: better solution?
                boxes = boxes[:2]

        targets = {'image_id': image_id, 'sentence_id': idx, 'boxes': boxes}
        targets['orig_size'] = image.shape[:-1]
        #print('orig_size: ', image.shape[:-1])

        image = image.transpose()
        image = torch.from_numpy(image)
        
        if image.shape[0] != 3:
            print('image: ', idx, image.shape)
            print(img_path)
        image, targets = resize(image, targets, 512)
        image = self.transform(image.float())

        return image, targets


