"""
Adapted from https://github.com/uzh-rpg/DSEC/blob/main/scripts/dataset/sequence.py
"""
from pathlib import Path
import weakref

import cv2
# import tables
import h5py
import hdf5plugin
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from joblib import Parallel, delayed

from dataset.event.DSEC.dataset.representations import VoxelGrid
from dataset.event.DSEC.utils.eventslicer import EventSlicer
import albumentations as A
import dataset.event.data_util as data_util
from dataset.event.data_util import gen_edge
import random
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as Rot

def correct_image(img_tensor, mapping):
    # 将 PyTorch 张量转换为 NumPy 数组
    img_np = img_tensor.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    img_np = (img_np * 255).astype(np.uint8)  # 转换为 uint8 格式

    # 对图像进行校正和映射
    img_corrected = cv2.remap(img_np, mapping, None, interpolation=cv2.INTER_CUBIC)

    # 将 NumPy 数组转换回 PyTorch 张量
    img_corrected = img_corrected.astype(np.float32) / 255.0
    img_corrected = torch.from_numpy(img_corrected).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return img_corrected

class Transform:
    def __init__(self, translation: np.ndarray, rotation: Rot):
        if translation.ndim > 1:
            self._translation = translation.flatten()
        else:
            self._translation = translation
        assert self._translation.size == 3
        self._rotation = rotation

    @staticmethod
    def from_transform_matrix(transform_matrix: np.ndarray):
        translation = transform_matrix[:3, 3]
        rotation = Rot.from_matrix(transform_matrix[:3, :3])
        return Transform(translation, rotation)

    @staticmethod
    def from_rotation(rotation: Rot):
        return Transform(np.zeros(3), rotation)

    def R_matrix(self):
        return self._rotation.as_matrix()

    def R(self):
        return self._rotation

    def t(self):
        return self._translation

    def T_matrix(self) -> np.ndarray:
        return self._T_matrix_from_tR(self._translation, self._rotation.as_matrix())

    def q(self):
        # returns (x, y, z, w)
        return self._rotation.as_quat()

    def euler(self):
        return self._rotation.as_euler('xyz', degrees=True)

    def __matmul__(self, other):
        # a (self), b (other)
        # returns a @ b
        #
        # R_A | t_A   R_B | t_B   R_A @ R_B | R_A @ t_B + t_A
        # --------- @ --------- = ---------------------------
        # 0   | 1     0   | 1     0         | 1
        #
        rotation = self._rotation * other._rotation
        translation = self._rotation.apply(other._translation) + self._translation
        return Transform(translation, rotation)

    def inverse(self):
        #           R_AB  | A_t_AB
        # T_AB =    ------|-------
        #           0     | 1
        #
        # to be converted to
        #
        #           R_BA  | B_t_BA    R_AB.T | -R_AB.T @ A_t_AB
        # T_BA =    ------|------- =  -------|-----------------
        #           0     | 1         0      | 1
        #
        # This is numerically more stable than matrix inversion of T_AB
        rotation = self._rotation.inv()
        translation = - rotation.apply(self._translation)
        return Transform(translation, rotation)

class Sequence(Dataset):
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_00_a)
    # ├── semantic
    # │   ├── left
    # │   │   ├── 11classes
    # │   │   │   └──data
    # │   │   │       ├── 000000.png
    # │   │   │       └── ...
    # │   │   └── 19classes
    # │   │       └──data
    # │   │           ├── 000000.png
    # │   │           └── ...
    # │   └── timestamps.txt
    # └── events
    #     └── left
    #         ├── events.h5
    #         └── rectify_map.h5

    def __init__(self, seq_path: Path, mode: str='train', event_representation: str = 'voxel_grid',
                 nr_events_data: int = 5, delta_t_per_data: int = 20, nr_events_per_data: int = 100000,
                 nr_bins_per_data: int = 5, require_paired_data=False, normalize_event=True, separate_pol=False,
                 semseg_num_classes: int = 11, augmentation=False, fixed_duration=False, remove_time_window: int = 250,
                 resize=False):
        assert nr_bins_per_data >= 1
        assert seq_path.is_dir()
        self.sequence_name = seq_path.name
        self.mode = mode

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.resize = resize
        self.shape_resize = None
        if self.resize: # False
            self.shape_resize = [448, 640]

        # Set event representation
        self.nr_events_data = nr_events_data
        self.num_bins = nr_bins_per_data
        assert nr_events_per_data > 0
        self.nr_events_per_data = nr_events_per_data
        self.event_representation = event_representation
        self.separate_pol = separate_pol
        self.normalize_event = normalize_event
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=self.normalize_event)

        self.locations = ['left']
        self.semseg_num_classes = semseg_num_classes
        self.augmentation = augmentation    # False

        # Save delta timestamp
        self.fixed_duration = fixed_duration    # False
        if self.fixed_duration:
            delta_t_ms = nr_events_data * delta_t_per_data
            self.delta_t_us = delta_t_ms * 1000
        self.remove_time_window = remove_time_window

        self.require_paired_data = require_paired_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load timestamps
        self.timestamps = np.loadtxt(str(seq_path / 'semantic' / 'timestamps.txt'), dtype='int64')

        # load label paths
        if self.semseg_num_classes == 11:
            label_dir = seq_path / 'semantic' / '11classes' / 'data'
        elif self.semseg_num_classes == 19:
            label_dir = seq_path / 'semantic' / '19classes' / 'data'
        else:
            raise ValueError
        assert label_dir.is_dir()
        label_pathstrings = list()
        for entry in label_dir.iterdir():
            assert str(entry.name).endswith('.png')
            label_pathstrings.append(str(entry))
        label_pathstrings.sort()
        self.label_pathstrings = label_pathstrings

        assert len(self.label_pathstrings) == self.timestamps.size

        # load images paths
        if self.require_paired_data:
            img_dir = seq_path / 'images'
            img_left_dir = img_dir / 'left' / 'ev_inf'
            assert img_left_dir.is_dir()
            img_left_pathstrings = list()
            for entry in img_left_dir.iterdir():
                assert str(entry.name).endswith('.png')
                img_left_pathstrings.append(str(entry))
            img_left_pathstrings.sort()
            self.img_left_pathstrings = img_left_pathstrings

            assert len(self.img_left_pathstrings) == self.timestamps.size

        # Remove several label paths and corresponding timestamps in the remove_time_window.
        # This is necessary because we do not have enough events before the first label.
        self.timestamps = self.timestamps[(self.remove_time_window // 100 + 1) * 2:]
        del self.label_pathstrings[:(self.remove_time_window // 100 + 1) * 2]
        assert len(self.label_pathstrings) == self.timestamps.size
        if self.require_paired_data:
            del self.img_left_pathstrings[:(self.remove_time_window // 100 + 1) * 2]
            assert len(self.img_left_pathstrings) == self.timestamps.size

        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()

        ev_dir = seq_path / 'events'
        for location in self.locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]

    def events_to_voxel_grid(self, x, y, p, t):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32')/256

    @staticmethod
    def get_img(filepath: Path, shape_resize=None):
        assert filepath.is_file()
        # filepath = Path('/home/longxl/share_container/Datasets/zhuxx/DSEC_events/test/zurich_city_14_c/images/left/ev_inf/000958.png')
        img = Image.open(str(filepath)) # image,size(1440,1080)
        if shape_resize is not None:
            img = img.resize((shape_resize[1], shape_resize[0]))
        img_transform = transforms.Compose([
            # transforms.Grayscale(),   # todo 试试灰度化
            transforms.ToTensor(),  # 直接转换为 (C, H, W) 格式的张量
        ])
        # img_tensor = img_transform(img).repeat(3, 1, 1)
        img_tensor = img_transform(img) # 不处理的为tensor:(3,1080,1440)(C,H,W)
        # #  note 可视化原始图像
        # plt.figure(figsize=(6, 6))
        # plt.imshow(img_tensor.permute(1, 2, 0))  # 转换为 (H, W, C) 以适应imshow
        # plt.show()
        filepath1 = filepath.parents[3]
        confpath = filepath1 / 'calibration' / 'cam_to_cam.yaml'
        assert confpath.exists()
        conf = OmegaConf.load(confpath)
        # 加载内参矩阵
        K_r0 = np.eye(3)
        K_r0[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect0']['camera_matrix']
        K_r1 = np.eye(3)
        K_r1[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect1']['camera_matrix']

        # 加载外参矩阵
        R_r0_0 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect0']))
        R_r1_1 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect1']))
        T_r0_0 = Transform.from_rotation(R_r0_0)
        T_r1_1 = Transform.from_rotation(R_r1_1)
        T_1_0 = Transform.from_transform_matrix(np.array(conf['extrinsics']['T_10']))

        # 计算相机之间的变换
        T_r1_r0 = T_r1_1 @ T_1_0 @ T_r0_0.inverse()
        R_r1_r0_matrix = T_r1_r0.R().as_matrix()
        P_r1_r0 = K_r1 @ R_r1_r0_matrix @ np.linalg.inv(K_r0)
        ht = 480
        wd = 640
        # coords: ht, wd, 2
        coords = np.stack(np.meshgrid(np.arange(wd), np.arange(ht)), axis=-1)
        # coords_hom: ht, wd, 3
        coords_hom = np.concatenate((coords, np.ones((ht, wd, 1))), axis=-1)
        # mapping: ht, wd, 3
        mapping = (P_r1_r0 @ coords_hom[..., None]).squeeze()
        # mapping: ht, wd, 2
        mapping = (mapping / mapping[..., -1][..., None])[..., :2]
        mapping = mapping.astype('float32')
        img_corrected = correct_image(img_tensor, mapping)
        img_corrected = img_corrected[:, :-40, :]
        # #  note 可视化矫正图像
        # plt.figure(figsize=(6, 6))
        # plt.imshow(img_corrected.permute(1, 2, 0))  # 转换为 (H, W, C) 以适应imshow
        # plt.show()




        # img_tensor = img_tensor[:, 120:-40, :-80] # (3,1080,1440) ->(3,880,1400)(C,纵，横)
        #
        # img_tensor = img_tensor.unsqueeze(0)  # (3,1080,1440)(C,H,W) -> [1, 3, 1080, 1440](1,C,H,W)
        # img_tensor = f.interpolate(img_tensor,
        #                                   size=(440,640),  # 高度和宽度
        #                                   mode='bilinear',
        #                                   align_corners=True)
        # img_tensor = img_tensor.squeeze(0)  # 去掉batch维度
        # # note 可视化调整后的图像 (3, 440, 640)，注意image在该数据集是删掉头顶的40行
        # plt.figure(figsize=(6, 6))
        # plt.imshow(img_tensor.permute(1, 2, 0))  # 转换为 (H, W, C) 以适应imshow
        # plt.show()

        # remove 40 bottom rows
        # img_tensor = img_tensor[:, 40:, :]  # (3,480,640) -> (3,440,640)
        # todo 归一化
        # 对每个通道计算均值和标准差
        # means = img_tensor.mean(dim=(1, 2), keepdim=True)
        # stddevs = img_tensor.std(dim=(1, 2), keepdim=True)

        # 对每个通道进行标准化
        # img_tensor = (img_tensor - means) / stddevs
        return img_corrected



    @staticmethod
    def get_label(filepath: Path):
        assert filepath.is_file()
        label = Image.open(str(filepath))
        label = np.array(label)
        return label

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    def __len__(self):
        return (self.timestamps.size + 1) // 2

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def generate_event_tensor(self, job_id, events, event_tensor, nr_events_per_data):
        id_start = job_id * nr_events_per_data
        id_end = (job_id + 1) * nr_events_per_data
        events_temp = events[id_start:id_end]
        event_representation = self.events_to_voxel_grid(events_temp[:, 0], events_temp[:, 1], events_temp[:, 3],
                                                         events_temp[:, 2])
        event_tensor[(job_id * self.num_bins):((job_id+1) * self.num_bins), :, :] = event_representation

    def __getitem__(self, index):
        label_path = Path(self.label_pathstrings[index * 2])
        # index = 476
        # label_path = Path('/home/longxl/share_container/Datasets/zhuxx/DSEC_events/test/zurich_city_14_c/semantic/11classes/data/000958.png')
        if self.resize:
            segmentation_mask = cv2.imread(str(label_path), 0)
            segmentation_mask = cv2.resize(segmentation_mask, (self.shape_resize[1], self.shape_resize[0]),
                                           interpolation=cv2.INTER_NEAREST)
            label = np.array(segmentation_mask)
        else:
            label = self.get_label(label_path)  # 形状(440,640)
        # # note 可视化标签
        # colors = [(0,  0,  0),
        #          (70 ,70, 70),
        #          (190,153,153),
        #          (220, 20,60),
        #          (153,153,153),
        #          (128, 64,128),
        #          (244, 35,232),
        #          (107,142, 35),
        #          (0,  0,  142),
        #          (102,102,156),
        #          (220,220,  0)]
        # # 创建一个自定义的颜色映射
        # cmap = mcolors.ListedColormap(colors)
        # # note 可视化标签图像
        # plt.figure(figsize=(6, 6))
        # plt.imshow(label, cmap=cmap)
        # plt.show()

        ts_end = self.timestamps[index * 2]

        output = {}
        for location in self.locations:
            if self.fixed_duration:
                ts_start = ts_end - self.delta_t_us
                event_tensor = None
                self.delta_t_per_data_us = self.delta_t_us / self.nr_events_data
                for i in range(self.nr_events_data):
                    t_s = ts_start + i * self.delta_t_per_data_us
                    t_end = ts_start + (i+1) * self.delta_t_per_data_us
                    event_data = self.event_slicers[location].get_events(t_s, t_end)

                    p = event_data['p']
                    t = event_data['t']
                    x = event_data['x']
                    y = event_data['y']

                    xy_rect = self.rectify_events(x, y, location)
                    x_rect = xy_rect[:, 0]
                    y_rect = xy_rect[:, 1]

                    if self.event_representation == 'voxel_grid':
                        event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
                    else:
                        events = np.stack([x_rect, y_rect, t, p], axis=1)
                        event_representation = data_util.generate_input_representation(events, self.event_representation,
                                                                  (self.height, self.width))
                        event_representation = torch.from_numpy(event_representation).type(torch.FloatTensor)

                    if event_tensor is None:
                        event_tensor = event_representation
                    else:
                        event_tensor = torch.cat([event_tensor, event_representation], dim=0)

            else:
                num_bins_total = self.nr_events_data * self.num_bins
                event_tensor = torch.zeros((num_bins_total, self.height, self.width))   # (5,480,640)
                self.nr_events = self.nr_events_data * self.nr_events_per_data  # 100000
                event_data = self.event_slicers[location].get_events_fixed_num(ts_end, self.nr_events)

                if self.nr_events >= event_data['t'].size:
                    start_index = 0
                else:
                    start_index = -self.nr_events

                p = event_data['p'][start_index:]
                t = event_data['t'][start_index:]
                x = event_data['x'][start_index:]
                y = event_data['y'][start_index:]
                nr_events_loaded = t.size

                xy_rect = self.rectify_events(x, y, location)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                nr_events_temp = nr_events_loaded // self.nr_events_data
                events = np.stack([x_rect, y_rect, t, p], axis=-1)
                # # note 可视化event数据
                # EVENT = np.ones((480, 640, 3))
                # # 将p=1的坐标用红色显示，将p=0的坐标用蓝色显示
                # for i in range(len(x)):
                #     if p[i] == 1:
                #         EVENT[y[i], x[i]] = [1, 0, 0]  # 红色
                #     else:
                #         EVENT[y[i], x[i]] = [0, 0, 1]  # 蓝色
                # EVENT = EVENT[:-40, :, :] # (480, 640, 3) -> (440, 640, 3)
                # plt.figure(figsize=(6, 6))
                # plt.imshow(EVENT)
                # plt.show()

                Parallel(n_jobs=8, backend="threading")(
                    delayed(self.generate_event_tensor)(i, events, event_tensor, nr_events_temp) for i in range(self.nr_events_data))

            # remove 40 bottom rows
            event_tensor = event_tensor[:, :-40, :] # (5,480,640) -> (5,440,640)
            # # note 可视化体素事件
            # Events_voxelgrid = event_tensor[3]
            # plt.figure(figsize=(6, 6))
            # plt.imshow(Events_voxelgrid, cmap='gray')  # 转换为 (H, W, C) 以适应imshow
            # plt.show()

            if self.resize:
                event_tensor = f.interpolate(event_tensor.unsqueeze(0),
                                             size=(self.shape_resize[0], self.shape_resize[1]),
                                             mode='bilinear', align_corners=True).squeeze(0)

            label_tensor = torch.from_numpy(label).long()   # (440,640)

            if self.augmentation:
                value_flip = round(random.random())
                if value_flip > 0.5:
                    event_tensor = torch.flip(event_tensor, [2])
                    label_tensor = torch.flip(label_tensor, [1])
                    
        # 变换维度
        # voxel grid的每个通道作为SNN的一个时间步长(1 channel = 10ms)
        if self.event_representation == 'voxel_grid':
            # x:[c(t),h,w]->[t(=c),c(=1),h,w]
            # event_tensor = event_tensor.unsqueeze(1)
            event_tensor = event_tensor
        # voxel grid的所有通道当作一个时间步长t(1 channel = 50/5/5 = 2ms)
        # elif self.event_representation == 'voxel_grid':
        #     # x:[c',h,w]=[t*c,h,w]->[t,c,h,w]
        #     C, H, W = event_tensor.shape
        #     event_tensor = event_tensor.view(C//5, 2, H, W)
        # elif self.event_representation == 'MDOE':
        #     # x:[t*4,h,w]->[t,4,h,w]->[t(=5),c(=4),h,w]
        #     C, H, W = event_tensor.shape
        #     event_tensor = event_tensor.view(C//4, 4, H, W)
        # # SBT的每个通道(2)作为SNN的一个时间步长
        # elif self.event_representation in ['SBT_1', 'SBE_1']:
        #     # x:[c',h,w]=[t*c,h,w]->[t,c,h,w]->[t(=5),c(=2),h,w]
        #     C, H, W = event_tensor.shape
        #     event_tensor = event_tensor.view(C//2, 2, H, W)
        # # SBT的所有通道(bins*2)当作一个时间步长t
        # elif self.event_representation in ['SBT_2', 'SBE_2']:
        #     # x:[c'(c*t),h,w]->[t,c,h,w]->[t(=5),c(=10),h,w]
        #     C, H, W = event_tensor.shape
        #     event_tensor = event_tensor.view(C//10, 10, H, W)
            
        edge = gen_edge(label)  # (440,640)
        edge_tensor = torch.from_numpy(edge).long()  # (440,640)

        if 'representation' not in output:
            output['representation'] = dict()
        output['representation'][location] = event_tensor

        if self.require_paired_data:
            img_left_path = Path(self.img_left_pathstrings[index * 2])
            output['img_left'] = self.get_img(img_left_path, self.shape_resize)
            return output['representation']['left'], label_tensor, edge_tensor, output['img_left']
        return output['representation']['left'], label_tensor, edge_tensor