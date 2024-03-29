import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
from colormap import rgb2hex


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.sub_sub_folder = 'left_frames'
        self.transforms = transforms
        

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))


    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)




class EndovisTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        self.sub_folders = list(sorted(os.listdir(root)))
        self.sub_sub_folder = 'left_frames'
        self.sub_sub_folder_masks = 'ground_truth/TypeSegmentationRescaled'
        self.imgs = []
        self.masks = []
        for sf in self.sub_folders[:-1]:

            list_files = list(sorted(os.listdir(osp.join(root, sf, self.sub_sub_folder))))
            list_path = [osp.join(root, sf, self.sub_sub_folder, file) for file in list_files]

            list_masks = list(sorted(os.listdir(osp.join(root, sf, self.sub_sub_folder_masks))))
            list_path_masks = [osp.join(root, sf, self.sub_sub_folder_masks, file) for file in list_masks]

            self.imgs += list_path
            self.masks += list_path_masks

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        mask_hex = np.apply_along_axis(lambda x : rgb2hex(x[0], x[1], x[2]), axis=-1, arr = mask.astype(np.int64) )  
        # instances are encoded as different colors
        obj_ids = np.unique(mask_hex)

        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # generate as puch masks as objects cf notebook
        masks = mask_hex == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin >= xmax or ymin >= ymax:
                continue
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)





class EndovisTunedDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, num_classes, train, val_frac):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.num_classes = num_classes
        self.sub_folders = list(sorted(os.listdir(root)))
        self.sub_sub_folder = 'left_frames'
        self.sub_sub_folder_masks = 'ground_truth'
        self.imgs = []
        self.imgs_val = []

        self.all_cat = {}
        self.cat_id = 0 # at the end of the __init__, self.cat_id is the number of categories

        for sf in self.sub_folders[:-1]:
            

            list_img = list(sorted(os.listdir(osp.join(root, sf, self.sub_sub_folder)))) # depends on this sf
            list_cat = list(sorted(os.listdir(osp.join(root, sf, self.sub_sub_folder_masks))))

            val_size = int(val_frac * len(list_img))

            for cat in list_cat:
                if cat not in self.all_cat:
                    self.all_cat[cat] = self.cat_id
                    self.cat_id += 1

            if train:
                for img in list_img[:-val_size]:
                    img = {'img_path': osp.join(root, sf, self.sub_sub_folder, img),
                        'masks_path': [osp.join(root, sf, self.sub_sub_folder_masks, cat ,img) for cat in list_cat], # the masks are ordered as the labels
                        'labels': list_cat}

                    self.imgs.append(img)
            
            if not train:
                for img in list_img[-val_size:]:
                    img = {'img_path': osp.join(root, sf, self.sub_sub_folder, img),
                        'masks_path': [osp.join(root, sf, self.sub_sub_folder_masks, cat ,img) for cat in list_cat], # the masks are ordered as the labels
                        'labels': list_cat}

                    self.imgs.append(img)


    def __getitem__(self, idx):
        img_dict = self.imgs[idx]

        # load images and masks
        img_path = img_dict['img_path']
        img = Image.open(img_path).convert("RGB")

        masks_paths = img_dict['masks_path']
        masks_list = [Image.open(mask_path) for mask_path in masks_paths]

        masks_list_bool = []
        for mask in masks_list:
            # parts are encoded as different colors
            mask = np.array(mask)
            mask_bool = mask != 0
            if (mask_bool == False).all():
                continue
            masks_list_bool.append(mask_bool)

        if masks_list_bool == []:
            return self.__getitem__(idx - 1)

        masks = np.stack(masks_list_bool, axis=0) # shape [N, h, w], bool


        # get bounding box coordinates for each mask
        num_objs = len(img_dict['labels'])
        boxes = []
        for i in range(len(masks_list_bool)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin >= xmax or ymin >= ymax:
                continue
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class

        if self.num_classes > 2:
            labels_list = [self.all_cat[cat] for cat in img_dict['labels']]
            labels = torch.as_tensor(labels_list, dtype=torch.int64)
        else:
            labels = torch.ones((num_objs,), dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)






class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "val2017"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)