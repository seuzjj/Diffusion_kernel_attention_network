
import pickle
from torch.utils.data import DataLoader, Dataset
import numpy as np
#from dataset.video_data import *
from dataset.skeleton import Skeleton, vis, Skeleton_val


class adni(Skeleton):
    def __init__(self, data_path, label_path, window_size, final_size,mode='train', decouple_spatial=False,
                 num_skip_frame=None, random_choose=False, center_choose=False, random_noise=False, random_scale=False):
        super().__init__(data_path, label_path, window_size, final_size, mode, decouple_spatial, num_skip_frame,
                         random_choose, center_choose, random_noise, random_scale)
        #self.edge = edge

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        # load data
        self.data = np.load(self.data_path, mmap_mode='r')  # NCTVM



class adni_val(Skeleton_val):
    def __init__(self, data_path, label_path, window_size, final_size, augtimes=1,mode='train', decouple_spatial=False,
                 num_skip_frame=None, random_choose=False, center_choose=False, random_noise=False, random_scale=False):
        super().__init__(data_path, label_path, window_size, final_size, augtimes, mode, decouple_spatial, num_skip_frame,
                         random_choose, center_choose, random_noise, random_scale)
        #self.edge = edge

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        self.data = np.load(self.data_path, mmap_mode='r')  # NCTVM



if __name__ == '__main__':
    data_path = "/your/path/to/ntu/xsub/val_data_joint.npy"
    label_path = "/your/path/to/ntu/xsub/val_label.pkl"
    test(data_path, label_path, vid='S004C001P003R001A032', edge=edge, is_3d=True, mode='train')
    # data_path = "/your/path/to/ntu/xsub/val_data_joint.npy"
    # label_path = "/your/path/to/ntu/xsub/val_label.pkl"
    # test(data_path, label_path, vid='S004C001P003R001A032', edge=edge, is_3d=True, mode='train')
