import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np 
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch

data_transforms = {
    "train": A.Compose([
        A.Resize(512,512),
#         A.Normalize(
#                 mean=[0.485, 0.456, 0.406], 
#                 std=[0.229, 0.224, 0.225], 
#                 max_pixel_value=255.0, 
#                 p=1.0,
#             ),
        A.CLAHE(p=0.35),
        A.ColorJitter(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=90, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
#             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
        A.CoarseDropout(max_holes=8, max_height=512//20, max_width=512//20,
                         min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ToTensorV2()], p=1.0),
    
    "valid": A.Compose([
        A.Resize(512,512),
#         A.Normalize(
#                 mean=[0.485, 0.456, 0.406], 
#                 std=[0.229, 0.224, 0.225], 
#                 max_pixel_value=255.0, 
#                 p=1.0
#             ),
        ToTensorV2()], p=1.0)
}

def make_df(df,id_list,args):
    
    image_name = np.unique([i.name.replace('.png','') for i in id_list])
    df = df[df['id'].isin(image_name)]
    df['image_path'] = args.base_path + '/train/' + df['id'] + '.png'
    tmp_df           = df.drop_duplicates(subset=["id", "image_path"]).reset_index(drop=True)
    tmp_df["annotation"] = df.groupby("id")["annotation"].agg(list).reset_index(drop=True)
    df               = tmp_df.copy()
    df['mask_path']  = id_list

    return df

def make_dataset(args): 
    df = pd.read_csv(f'{args.base_path}/train.csv')
    
    mask_path = Path(f'{args.base_path}/label_info/binary_mask/fold{args.Kfold}')
    
    image_name = list(mask_path.glob('train/*.png'))
    
    train_df = make_df(df,image_name,args)
    image_name = list(mask_path.glob('valid/*.png'))
    valid_df = make_df(df,image_name,args)
    return train_df,valid_df


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.img_paths  = df['image_path'].values
        try: # if there is no mask then only send images --> test data
            self.msk_paths  = df['mask_path'].values
        except:
            self.msk_paths  = None
        self.transforms = data_transforms[transforms]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = str(self.img_paths[index])
        img      = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        

        if self.msk_paths is not None:
            msk_path = str(self.msk_paths[index])
            msk      = (cv2.imread(msk_path,cv2.IMREAD_GRAYSCALE) > 0 ) * 1
            
            # print(msk.shape)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            msk      = np.expand_dims(msk, axis=0) # output_shape: (batch_size, 1, img_size, img_size)
            return img, msk
        else:
            ## chnage binary image
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            return img