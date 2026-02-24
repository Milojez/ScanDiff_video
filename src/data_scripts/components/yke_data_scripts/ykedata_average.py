#this script should let the dataloader know how does a singular sample looks like and it the script is needed for the Dataloader
#so far in the code only the json files names were changed for the good ones. 
# The problem still remains with assigning multiple frames to one sample which should be fed to the model

# also the task.embeddigns should be reconsidered

import torch
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
import json



class YkeDataset:
    def __init__(self, name: str, root_path: str, task: str, split: str, num_subjects: int, threshold: float = -1., threshold_no_dur: float = -1.,
                time_in_ms: bool = False, use_abs_coords: bool = True, truncate_seconds: int = 3,
                task_embeddings_file: str = 'task_embeddings.npy', img_features_dir: str = 'dinov2_base_timm_image_features') -> None:
        self.name = name
        self.root_path = root_path
        self.task = task
        self.threshold = threshold
        self.threshold_no_dur = threshold_no_dur

        if split == 'valid':
            split = 'validation'
        
        with open(Path(self.root_path, f'ykedata_2s_fixations_{split}.json'), 'rb') as f:
            data = json.load(f)
            
        
        self.samples = {'sequences': []}
        self.samples['sequences'] = [s for s in data if s['split'] == split]
        
        self.num_subjects = num_subjects
        self.time_in_ms = time_in_ms
        self.use_abs_coords = use_abs_coords
        self.truncate_seconds = truncate_seconds
        self.img_features_dir = img_features_dir
        
        self.subject_per_img = {}

        for s in self.samples['sequences']:
            names = s['name']

            if isinstance(names, list):
                key = "||".join(names)  # make list hashable
            else:
                key = names

            self.subject_per_img[key] = self.subject_per_img.get(key, 0) + 1
                
        self.embeddings = np.load(
            open(
                Path('./data', task_embeddings_file),
                mode="rb",
            ),
            allow_pickle=True,
        ).item()
        
        self.task_embedding = torch.from_numpy(self.embeddings[""])

    #NEW ADDED FUNCTION - keeps shapes consistent and avoids squeeze mistakes
    def _load_feats(self, frame_name: str) -> torch.Tensor: 
        p = Path(self.root_path, self.img_features_dir, Path(frame_name).stem + '.pth')
        feats = torch.load(p)

        # (1, N, D) -> (N, D)
        if feats.ndim == 3 and feats.shape[0] == 1:
            feats = feats.squeeze(0)

        # (D,) -> (1, D)
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)

        return feats  # (N, D)

    def __getitem__(self, index: int):
        sample = self.samples['sequences'][index]
        names = sample['name']

        if isinstance(names, list):
            frame_names = names
            clip_key = "||".join(frame_names)
            img_filename = frame_names[0]  # first frame as representative
        else:
            frame_names = [names]
            clip_key = names
            img_filename = names

        num_viewers = self.subject_per_img[clip_key]
        original_width = sample['width']
        original_height = sample['height']

        # --- NEW: load all frames and average ---
        frame_feats = [self._load_feats(fn) for fn in frame_names]
        img_feats = torch.stack(frame_feats, dim=0).mean(dim=0)  # (N, D)
                
        x_coords = sample['X'] # x, y coords in osie are already in absolute terms
        y_coords = sample['Y']
        durations = sample['T'] #durations are already in ms
        
        scanpath = np.column_stack((x_coords, y_coords, durations))
        
        # convert coords in [0, 1]
        scanpath[:,0] /= original_width
        scanpath[:,1] /= original_height
    
        # put time in seconds
        scanpath[:,2] /= 1000.0
        
        scanpath = torch.from_numpy(scanpath).float()
        
        if len(scanpath) == 0:
            print('Found scanpath of zero len, discarding it from the training size...')
            scanpath = None 
        
        return {'img_filename': img_filename, 'original_img_size': (original_width, original_height), 'img': img_feats, 'scanpath': scanpath,
                'task': "", 'task_embedding': self.task_embedding, 'num_viewers': num_viewers, 'dataset': self.name, 'question_id': ""}

    def __len__(self) -> int:
        return len(self.samples['sequences'])
    