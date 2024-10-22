import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import imagesize
import numpy as np
import json
import torch

from myproject.util.pdutil import split_col_to_rows


class CustomDataset(Dataset):

    def __init__(self, root, transforms=None):
        pd_data = pd.read_csv(root, low_memory=False)
        self.data = self.split_x_y(pd_data).head(10000)
        self.transforms = transforms

    def split_x_y(self, pd_data):
        filtered_pdb = pd_data['Path'].apply(lambda x: x.split('/')[0]) == '0'
        pdb_filtered = pd_data[filtered_pdb].reset_index(drop=True)
        final_result = split_col_to_rows(pdb_filtered)
        return final_result



    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        rel_path = self.data.iloc[idx]['Path']

        abs_path = f"/Users/liuxiaobo/Downloads/adownload/competition/mmdet_bisai/{rel_path}"

        bbox_axis = self.data.iloc[idx]['Polygons']

        if len(json.loads(bbox_axis)) > 1:
            print("Multiple bboxes found for one image")

        bbox = [np.array(x, dtype=np.int64) for x in json.loads(bbox_axis)]

        b = np.array([(bb[:, 0].min(), bb[:, 1].min(), bb[:, 0].max(), bb[:, 1].max()) for bb in bbox],
                     dtype=np.int64)
        x_min,y_min,x_max,y_max = b[:,0],b[:,1],b[:,2],b[:,3]
        bbox_width = x_max - x_min
        bbox_height= y_max- y_min

        annotations = [{'bbox':[x_min,y_min, bbox_width, bbox_height],'category_id':torch.tensor(0)}]

        img = Image.open(abs_path).convert('RGB')
        img = self.transforms(img)

        return img, annotations;
