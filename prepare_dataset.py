import torch
from torch.utils.data import Dataset
import pandas as pd
from tiger.io import read_image, write_image, read_dicom, ImageMetadata
import utils
import torchvision.transforms.functional as F
from PIL import Image
from utils import box_xyxy_to_cxcywh
from torchvision import transforms
import torchvision.transforms as T
import json


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
        #print(type(boxes[0][0]))
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

def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for beg, end in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)
    return positive_map #/ (positive_map.sum(-1)[:, None] + 1e-6)


"""
load a ct scan
"""
def load_file(file, path=None):
    if path:
        image, header = read_dicom(path + file + '/dicom.zip')
    else:
        image, header = read_dicom('Data/CT_scans/' + file + '/dicom.zip')
    voxel_spacing = header["spacing"]

    return image, header, voxel_spacing


class LesionData(Dataset):
    def __init__(self, annotations_dir, path_chansey, tokenizer, split='train'):
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
        #self.slice_data = pd.read_csv(img_dir + 'slice_data.csv', index_col=0)

        self.tokenizer = tokenizer
        #self.transform = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #self.transform = T.Normalize([-2682.8042, -2739.8506, -2725.4387], [2187.5068, 2235.9980, 2225.6160])
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

        targets = {'image_id': image_id, 'sentence_id': idx, 'sentence': sentence, 'tokenized': self.tokenizer(sentence, return_tensors="pt"), 'boxes': boxes, 'target': type_label, 'labels':type_label_words}
        targets['positive map'] = create_positive_map(targets['tokenized'], targets['target'])
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

def load_test_image(tokenizer):
    img = Image.open("Data/000000001503.jpg")
    convert_tensor = transforms.ToTensor()
    image = convert_tensor(img)
    targets = {}
    sentence = "A computer with a keyboard next to a laptop."
    tokenized = tokenizer(sentence, return_tensors="pt")
    targets['sentence'] = sentence
    targets['tokenized'] = tokenized
    #targets['boxes'] = [[0.54,99.78,125.66,135.91], [120.96,177.97,37.77,22.06], [161.13,152.44,154.31,45.52], [125.66,11.46,111.64,89.53], [305.55,154.39,14.4,9.66]]
    targets['boxes'] = [[0.54, 99.78, 126.2, 235.69],
                        [161.13, 152.44, 315.44, 197.96],
                        [125.66, 11.46, 237.3, 100.99]]
    targets['target'] = [[(37,43)],[(18,26)],[(2,10)]]
    type_label_words = [[sentence[x:y] for x, y in b] for b in targets['target']]
    targets['labels'] = type_label_words
    targets['positive map'] = create_positive_map(targets['tokenized'], targets['target'])
    targets['orig_size'] = image.shape[1:]

    img_h, img_w = image.shape[1:]
    scaled_boxes = torch.as_tensor(targets['boxes']) / torch.as_tensor([img_w, img_h, img_w, img_h])
    targets['scaled boxes'] = box_xyxy_to_cxcywh(scaled_boxes)

    return image, targets


class CocoDetection(Dataset):
    def __init__(self, img_folder, ann_file, tokenizer=None):
        self.img_folder = img_folder
        with open(ann_file, "r") as f:
            self.annotations = json.load(f)
        self.prepare = ConvertCocoPolysToMask()
        self.convert_tensor = transforms.ToTensor()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_meta = self.annotations['images'][idx]
        target = self.annotations['annotations'][idx]
        #print(img_meta)
        #print(target)
        img = Image.open(self.img_folder+img_meta['file_name'])
        img_meta['annotations'] = [target]
        img_meta['image_id'] = target['image_id']
        img, targets = self.prepare(img, img_meta)
        img = self.convert_tensor(img)
        img, targets = resize(img, targets, (512, 512))
        #targets['tokenized'] = self.tokenizer(targets['sentence'], return_tensors="pt")
        targets['positive map'] = create_positive_map(targets['tokenized'], [target['tokens_positive']])
        targets['target'] = [target['tokens_positive']]
        targets['sentence_id'] = idx

        if img.shape[0] != 3:
            #print("Error in image", idx, " with shape", img.shape)
            img = img.squeeze().repeat(3, 1, 1)

        return img, targets


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, return_tokens=False, tokenizer=None):
        self.return_masks = return_masks
        self.return_tokens = return_tokens
        self.tokenizer = tokenizer

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        isfinal = None
        if anno and "isfinal" in anno[0]:
            isfinal = torch.as_tensor([obj["isfinal"] for obj in anno], dtype=torch.float)

        tokens_positive = [] if self.return_tokens else None
        if self.return_tokens and anno and "tokens" in anno[0]:
            tokens_positive = [obj["tokens"] for obj in anno]
        elif self.return_tokens and anno and "tokens_positive" in anno[0]:
            tokens_positive = [obj["tokens_positive"] for obj in anno]

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if caption is not None:
            target["sentence"] = caption

        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        if tokens_positive is not None:
            target["tokens_positive"] = []

            for i, k in enumerate(keep):
                if k:
                    target["tokens_positive"].append(tokens_positive[i])

        if isfinal is not None:
            target["isfinal"] = isfinal

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self.return_tokens and self.tokenizer is not None:
            assert len(target["boxes"]) == len(target["tokens_positive"])
            tokenized = self.tokenizer(caption, return_tensors="pt")
            target["positive_map"] = create_positive_map(tokenized, target["tokens_positive"])
        return image, target

        
class DeepLesion(Dataset):
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

        # self.img_labels = pd.read_csv(annotations_dir + 'DL_info_v3.csv', sep=';')

        self.img_dir = img_dir
        # self.transform = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = T.Normalize([-2682.8042, -2739.8506, -2725.4387], [2187.5068, 2235.9980, 2225.6160])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_filename = self.img_labels.iloc[idx]['File_name']
        img_path = self.img_dir + image_filename.replace('png', 'mha')
        image, header = read_image(img_path)
        boxes = eval(self.img_labels.iloc[idx]['Bounding_boxes'])
        boxes = [eval(x) for x in boxes]

        targets = {'image_id': image_filename, 'boxes': boxes, 'image_id': idx, 'sentence_id': idx}

        # print("shape: ", image.shape, " idx: ", idx)
        targets['orig_size'] = image.shape[1:]
        image = torch.from_numpy(image)
        image, targets = resize(image, targets, 512)
        image = self.transform(image.float())

        return image, targets
