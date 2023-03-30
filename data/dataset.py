import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# from .utils import compute_rotation_inv, process_camera_inv
from .utils import compute_rotation, process_camera_inv

def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')

class HDTFDataset(Dataset):
    def __init__(self, opt, is_inference):
        path = opt.path
        self.env = lmdb.open(
            os.path.join(path, str(opt.resolution)),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            map_size=1024 ** 4,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        list_file = "test_list.txt" if is_inference else "train_list.txt"
        list_file = os.path.join(path, list_file)
        with open(list_file, 'r') as f:
            lines = f.readlines()
            videos = [line.replace('\n', '') for line in lines]

        self.resolution = opt.resolution
        self.semantic_radius = opt.semantic_radius
        self.video_items, self.person_ids = self.get_video_index(videos)
        self.idx_by_person_id = self.group_by_key(self.video_items, key='person_id')

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])

        self.opt = opt 

        # just for debug
        self.person_ids = self.person_ids * 100

        # data validity check
        # bad_ids = []
        # print(len(self.person_ids))
        # for person_id in self.person_ids:
        #     idxs = self.idx_by_person_id[person_id]
        #     video_items = [self.video_items[idx] for idx in idxs]
        #     from tqdm import tqdm
        #     for video_item in tqdm(video_items, desc=person_id):
        #         with self.env.begin(write=False) as txn:
        #             try:
        #                 semantics_key = format_for_lmdb(video_item['video_name'], 'coeff_3dmm')
        #                 semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
        #                 semantics_numpy = semantics_numpy.reshape((video_item['num_frame'],-1))        
                    
        #                 condition_key = format_for_lmdb(video_item['video_name'], 'condition')
        #                 condition_numpy = np.frombuffer(txn.get(condition_key), dtype=np.float32)
        #                 condition_numpy = condition_numpy.reshape((video_item['num_frame'], -1))            
        #                 conditions_numpy = self.unfold_conditions(condition_numpy)

        #                 keypoint_key = format_for_lmdb(video_item['video_name'], 'keypoint')
        #                 keypoint_numpy = np.frombuffer(txn.get(keypoint_key), dtype = np.float32).copy()
        #                 keypoint_numpy = keypoint_numpy.reshape((video_item['num_frame'], 68, 2))
                    
        #             except:
        #                 if person_id not in bad_ids:
        #                     print(person_id)
        #                     bad_ids.append(person_id)

    def get_video_index(self, videos):
        video_items = []
        for video in videos:
            video_items.append(self.Video_Item(video))

        person_ids = sorted(list({video.split('#')[0] for video in videos}))

        return video_items, person_ids            

    def group_by_key(self, video_list, key):
        return_dict = collections.defaultdict(list)
        for index, video_item in enumerate(video_list):
            return_dict[video_item[key]].append(index)
        return return_dict  
    
    def Video_Item(self, video_name):
        video_item = {}
        video_item['video_name'] = video_name
        video_item['person_id'] = video_name.split('#')[0]
        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], 'length')
            length = int(txn.get(key).decode('utf-8'))
        video_item['num_frame'] = length
        
        return video_item

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, index):
        person_id = self.person_ids[index]
        video_item = self.video_items[random.choices(self.idx_by_person_id[person_id], k=1)[0]]

        data_dict = {
            'images': [],
            'semantics': [],
            'conditions': [],
            'keypoint': [],
            'id': []
        }        

        with self.env.begin(write=False) as txn:
            # loading video relasted attributes
            try:
                semantics_key = format_for_lmdb(video_item['video_name'], 'coeff_3dmm')
                semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
                semantics_numpy = semantics_numpy.reshape((video_item['num_frame'],-1))

                id_key = format_for_lmdb(video_item['video_name'], 'id')
                id_numpy = np.frombuffer(txn.get(id_key), dtype=np.float32).copy()

                condition_key = format_for_lmdb(video_item['video_name'], 'condition')
                condition_numpy = np.frombuffer(txn.get(condition_key), dtype=np.float32)
                condition_numpy = condition_numpy.reshape((video_item['num_frame'], -1))            
                conditions_numpy = self.unfold_conditions(condition_numpy)

                keypoint_key = format_for_lmdb(video_item['video_name'], 'keypoint')
                keypoint_numpy = np.frombuffer(txn.get(keypoint_key), dtype = np.float32).copy()
                keypoint_numpy = keypoint_numpy.reshape((video_item['num_frame'], 68, 2))

                # the eye-related weights
                weights = np.linalg.norm(semantics_numpy[:, 80:144], ord = 1, axis = 1) + 1e-3  #np.linalg.norm(1 - semantics_numpy[:, 96:103], ord = 2, axis = 1) + 1e-3
                # select both frames
                frame_list = list(range(video_item['num_frame']))
                frame_source = random.choices(frame_list, weights = weights, k = 1)[0]
                frame_target = random.choices(frame_list,  weights = weights, k = self.opt.frames_each_video - 1)


                for idx in [frame_source] + frame_target:
                    key = format_for_lmdb(video_item['video_name'], idx)    
                    img_bytes = txn.get(key)

                    img = Image.open(BytesIO(img_bytes))
                    data_dict['images'].append(self.transform(img))
                    data_dict['semantics'].append(self.transform_semantic(semantics_numpy, idx))
                    data_dict['conditions'].append(torch.from_numpy(conditions_numpy[idx, ...]))
                    data_dict['keypoint'].append(torch.from_numpy(keypoint_numpy[idx, ...]))
                    data_dict['id'].append(torch.from_numpy(id_numpy))
            except:
                print('!!!!!!', video_item['video_name'])

        for k,v in data_dict.items():
            data_dict[k] = torch.stack(v, dim = 0)

        return data_dict

    def unfold_conditions(self, conditions):
        angles, translations, focals = conditions[:, 0:3], conditions[:,3:6], conditions[:, -1]    

        angles = angles.copy()
        angles[:, 0] *= -1
        angles[:, 1] *= -1

        Rs = compute_rotation(torch.from_numpy(angles.copy())).numpy()
        c_list = process_camera_inv(translations.copy(), Rs, focals)
        return np.vstack(c_list)

    def transform_semantic(self, semantic, frame_index):
        index = self.obtain_seq_index(frame_index, semantic.shape[0])
        coeff_3dmm = semantic[index,...]
        # id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:,80:144] #expression
        # tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:,224:227] #euler angles for pose
        # gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:,254:257] #translation
        # crop = coeff_3dmm[:,257:260] / self.opt.resolution #crop param

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation], 1)
        return torch.Tensor(coeff_3dmm).permute(1,0)

    def obtain_seq_index(self, index, num_frames):
        seq = list(range(index-self.semantic_radius, index+self.semantic_radius+1))
        seq = [ min(max(item, 0), num_frames-1) for item in seq ]
        return seq



