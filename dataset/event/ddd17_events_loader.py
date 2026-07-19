import glob
from os.path import join, exists, dirname, basename
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as f
import torchvision.transforms as transforms

from dataset.event.extract_data_tools.example_loader_ddd17 import load_files_in_directory, extract_events_from_memmap
import dataset.event.data_util as data_util
import albumentations as A
from PIL import Image
from dataset.event.labels import shiftUpId, shiftDownId
from dataset.event.data_util import gen_edge
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
def get_split(dirs, split):
    return {
        "train": [dirs[0], dirs[2], dirs[3], dirs[5], dirs[6]],
        "test": [dirs[4]],
        "valid": [dirs[1]]
        # "train": [dirs[1]],
        # "test": [dirs[1]],
        # "valid": [dirs[1]]
    }[split]


def unzip_segmentation_masks(dirs):
    for d in dirs:
        assert exists(join(d, "segmentation_masks.zip"))
        if not exists(join(d, "segmentation_masks")):
            print("Unzipping segmentation mask in %s" % d)
            os.system("unzip %s -d %s" % (join(d, "segmentation_masks"), d))


class DDD17Events(Dataset):
    def __init__(self, root, split="train", event_representation='voxel_grid',
                 nr_events_data=5, delta_t_per_data=50, nr_bins_per_data=5, require_paired_data=True,
                 separate_pol=False, normalize_event=False, augmentation=False, fixed_duration=False,
                 nr_events_per_data=32000, resize=True, random_crop=False):
        data_dirs = sorted(glob.glob(join(root, "dir*")))   # 从指定的根目录root中查找所有以"dir"开头的文件夹，并将这些文件夹路径按字母顺序排序后存储在data_dirs列表中
        print(data_dirs)
        assert len(data_dirs) > 0
        assert split in ["train", "valid", "test"]

        self.split = split
        self.augmentation = augmentation
        self.fixed_duration = fixed_duration
        self.nr_events_per_data = nr_events_per_data

        self.nr_events_data = nr_events_data
        self.delta_t_per_data = delta_t_per_data
        if self.fixed_duration:
            self.t_interval = nr_events_data * delta_t_per_data
        else:
            self.t_interval = -1
            self.nr_events = self.nr_events_data * self.nr_events_per_data
        assert self.t_interval in [10, 50, 250, -1]
        self.nr_temporal_bins = nr_bins_per_data
        self.require_paired_data = require_paired_data
        self.event_representation = event_representation
        self.shape = [260, 346]
        self.resize = resize
        self.shape_resize = [260, 352]  # note 高是352可能是因为是8和16的倍数，宽到后面是260-60=200，是8的倍数，所以要调整图像尺寸
        self.random_crop = random_crop
        self.shape_crop = [120, 216]
        self.separate_pol = separate_pol
        self.normalize_event = normalize_event
        self.dirs = get_split(data_dirs, split)
        # unzip_segmentation_masks(self.dirs)

        self.files = []
        for d in self.dirs:
            self.files += glob.glob(join(d, "segmentation_masks", "*.png"))

        print("[DDD17Segmentation]: Found %s segmentation masks for split %s" % (len(self.files), split))

        # load events and image_idx -> event index mapping
        self.img_timestamp_event_idx = {}
        self.event_data = {}
        self.rgb_data = {}   # note ==================

        print("[DDD17Segmentation]: Loading real events.")
        self.event_dirs = self.dirs

        for d in self.event_dirs:
            img_timestamp_event_idx, t_events, xyp_events, _, rgb_images = load_files_in_directory(d, self.t_interval)
            self.img_timestamp_event_idx[d] = img_timestamp_event_idx
            self.event_data[d] = [t_events, xyp_events]
            self.rgb_data[d] = rgb_images   # note ==================

        if self.augmentation:
            self.transform_a = A.ReplayCompose([
                A.HorizontalFlip(p=0.5)
            ])
            self.transform_a_random_crop = A.ReplayCompose([
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True)])
        self.transform_a_center_crop = A.ReplayCompose([
            A.CenterCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
        ])

    def __len__(self):
        return len(self.files)

    def apply_augmentation(self, transform_a, events, label):
        label = shiftUpId(label)
        A_data = transform_a(image=events[0, :, :].numpy(), mask=label)
        label = A_data['mask']
        label = shiftDownId(label)
        if self.random_crop and self.split == 'train':
            events_tensor = torch.zeros((events.shape[0], self.shape_crop[0], self.shape_crop[1]))
        else:
            events_tensor = events
        for k in range(events.shape[0]):
            events_tensor[k, :, :] = torch.from_numpy(
                A.ReplayCompose.replay(A_data['replay'], image=events[k, :, :].numpy())['image'])
        return events_tensor, label

    def __getitem__(self, idx):
        segmentation_mask_file = self.files[idx]
        # todo 测试
        # segmentation_mask_file = '/home/longxl/share_container/Datasets/zhuxx/DDD17_events/dir1/segmentation_masks/segmentation_00004007.png'
        segmentation_mask = cv2.imread(segmentation_mask_file, 0)   # 第二个参数为0，所以是读取灰度图
        label_original = np.array(segmentation_mask)
        # 如果需要调整尺寸(self.resize为True),则过cv2.resize将分割掩码调整为指定的尺寸
        if self.resize:
            segmentation_mask = cv2.resize(segmentation_mask, (self.shape_resize[1], self.shape_resize[0] - 60),
                                           interpolation=cv2.INTER_NEAREST)
        label = np.array(segmentation_mask)

        directory = dirname(dirname(segmentation_mask_file))

        img_idx = int(basename(segmentation_mask_file).split("_")[-1].split(".")[0]) - 1
        img_timestamp_event_idx = self.img_timestamp_event_idx[directory]   # 加载directory目录下事件的所有时间戳索引
        t_events, xyp_events = self.event_data[directory]   # 加载directory目录下所有事件数据
        # note 提取对应的RGB图像数据
        rgb_images = self.rgb_data[directory]  # 从目录中提取 RGB 数据列表
        rgb_image = rgb_images[img_idx]  # 根据图像索引提取对应的 RGB 图像
        # 👇 可视化image数据
        # plt.figure(figsize=(6, 6))
        # plt.imshow(rgb_image)
        # plt.show()
        # 👆
        # 将 numpy.ndarray 转换为张量 (注意 PyTorch 的维度顺序是 [C, H, W])
        rgb_tensor = torch.from_numpy(rgb_image)    # [260, 346, 3][H,W,C]
        if self.resize:
            rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)  # [260, 346, 3] -> [1, 3, 260, 346]
            rgb_tensor_resize = f.interpolate(rgb_tensor,
                                               size=(self.shape_resize[0], self.shape_resize[1]),  # 高度和宽度
                                               mode='bilinear',
                                               align_corners=True)
            # 去掉第0维并转换为 numpy
            # rgb_tensor = rgb_tensor_resize.squeeze(0).permute(1, 2, 0)  # [1, 3, 260, 352] -> [260, 352, 3]
            rgb_tensor = rgb_tensor_resize.squeeze(0)  #  [1, 3, 260, 352] -> [3, 260, 352]
            if self.normalize_event:
                # 目前宽高就是数据集本来的原始图像尺寸，所以(3，260，352)是RGB数据(C,H,W)
                rgb_tensor = data_util.normalize_rgb(rgb_tensor)    # todo 对RGB做归一化
        # events has form x, y, t_ns, p (in [0,1]) x(横坐标)、y(纵坐标)、t_ns(时间戳)和p(极性)
        # 根据是否固定时间窗口(self.fixed_duration),调用extract_events_from_memmap提取事件数据
        if self.fixed_duration:
            events = extract_events_from_memmap(t_events, xyp_events, img_idx, img_timestamp_event_idx, 
                                                self.fixed_duration)
        else:
            events = extract_events_from_memmap(t_events, xyp_events, img_idx, img_timestamp_event_idx, 
                                                self.fixed_duration, self.nr_events)
        # 👇 可视化事件数据
        # x = events[:, 0]
        # y = events[:, 1]
        # p = events[:, 3]
        # # 初始化帧 (高度 260, 宽度 346)
        # frame_combined = np.zeros((260, 346), dtype=np.float32)
        # # 累积事件到帧中
        # for xi, yi, pi in zip(x, y, p):
        #     frame_combined[yi, xi] += 1 if pi == 1 else -1  # 正极性 +1，负极性 -1
        # 可视化
        # plt.figure(figsize=(6, 6))
        # plt.imshow(frame_combined, cmap='seismic', interpolation='nearest')  # 红蓝表示正负极性
        # plt.colorbar(label="Polarity Count")
        # plt.show()
        # 👆
        # 此时的events就是对应于img_idx这张图片的事件数据
        t_ns = events[:, 2]  # 获取事件的时间戳列(获取所有行的第三列)
        delta_t_ns = int((t_ns[-1] - t_ns[0]) / self.nr_events_data)    # 根据事件时间范围,将时间窗口均分为self.nr_events_data个部分
        nr_events_loaded = events.shape[0]  # 加载的事件总数
        nr_events_temp = nr_events_loaded // self.nr_events_data    #  根据事件总数和数据分块数计算每个块的事件数量

        id_end = 0
        event_tensor = None
        for i in range(self.nr_events_data):
            id_start = id_end
            if self.fixed_duration:
                id_end = np.searchsorted(t_ns, t_ns[0] + (i + 1) * delta_t_ns)
            else:
                id_end += nr_events_temp

            if id_end > nr_events_loaded:
                id_end = nr_events_loaded
            # nr_temporal_bins:表示每个时间窗口的时间分辨率;separate_pol:是否区分事件极性（正极性或负极性）
            # event_representation此时的事件数据表示已经通过generate_input_representation中的generate_voxel_grid函数生成了以voxel_grid的方式保存的体素网格
            event_representation = data_util.generate_input_representation(events[id_start:id_end],
                                                                           self.event_representation,
                                                                           self.shape,
                                                                           nr_temporal_bins=self.nr_temporal_bins,
                                                                           separate_pol=self.separate_pol)
            # 👇 可视化voxel grid处理后的事件数据
            # plt.figure(figsize=(15, 10))  # 设置画布大小
            # for t in range(5):
            #     plt.imshow(event_representation[t], cmap='gray')  # 用灰度图显示第 t 个时间段
            #     plt.title(f"Time Bin {t + 1}")
            #     plt.axis("off")  # 关闭坐标轴
            #     plt.show()
            # 👆
            event_representation = torch.from_numpy(event_representation)
            if self.normalize_event:
                # 5是nr_bins_per_data超参数定义的，应该是时间分辨率，所以可以认为是t
                # 目前宽高就是数据集本来的原始图像尺寸，所以(5，260，346)是事件数据(t,H,W)
                event_representation = data_util.normalize_voxel_grid(event_representation)
            # 处理事件数据，调整事件数据尺寸
            if self.resize:
                event_representation_resize = f.interpolate(event_representation.unsqueeze(0),
                                                            size=(self.shape_resize[0], self.shape_resize[1]),
                                                            mode='bilinear', align_corners=True)
                event_representation = event_representation_resize.squeeze(0)
            
            if event_tensor is None:
                event_tensor = event_representation
            else:
                event_tensor = torch.cat([event_tensor, event_representation], dim=0)
            
        event_tensor = event_tensor[:, :-60, :]  # remove 60 bottom rows (5,260,352) ->(5,200,352)
        # rgb_tensor = rgb_tensor[:-60, :, :]  # note  同理，将RGB也移除最下面的H的行 (260,352,3) ->(200,352,3)，移除最下面60行的目的是将数据采集车的摄像头拍到自己的引擎盖的部分移除
        rgb_tensor = rgb_tensor[:, :-60, :]  #  (3,260,352) ->(3,200,352)

        if self.random_crop and self.split == 'train':
            event_tensor = event_tensor[:, -self.shape_crop[0]:, :]
            rgb_tensor = rgb_tensor[:, -self.shape_crop[0]:, :]
            label = label[-self.shape_crop[0]:, :]  # 标签随机裁剪
            if self.augmentation:   # 数据增强
                event_tensor, label = self.apply_augmentation(self.transform_a_random_crop, event_tensor, label)
        else:
            if self.augmentation:
                event_tensor, label = self.apply_augmentation(self.transform_a, event_tensor, label)
        
        # 变换维度
        # voxel grid的每个通道作为SNN的一个时间步长(1 channel = 10ms)
        if self.event_representation == 'voxel_grid':
            # x:[c(t),h,w]->[t(=c),c(=1),h,w]
            # event_tensor = event_tensor.unsqueeze(1)
            event_tensor = event_tensor
        elif self.event_representation == 'MDOE':
            # x:[t*4,h,w]->[t,4,h,w]->[t(=5),c(=4),h,w]
            C, H, W = event_tensor.shape
            event_tensor = event_tensor.view(C//4, 4, H, W)
        # SBT的每个通道(2)作为SNN的一个时间步长
        elif self.event_representation in ['SBT_1', 'SBE_1']: 
            # x:[c',h,w]=[t*c,h,w]->[t,c,h,w]->[t(=5),c(=2),h,w]
            C, H, W = event_tensor.shape
            event_tensor = event_tensor.view(C//2, 2, H, W)
        # SBT的所有通道(bins*2)当作一个时间步长t
        elif self.event_representation in ['SBT_2', 'SBE_2']: 
            # x:[c'(c*t),h,w]->[t,c,h,w]->[t(=5),c(=10),h,w]
            C, H, W = event_tensor.shape
            event_tensor = event_tensor.view(C//10, 10, H, W)
        elif self.event_representation in ['histogram', 'ev_segnet']:
            event_tensor = event_tensor.tile([self.nr_temporal_bins, 1, 1, 1])
        
        edge = gen_edge(label)
        label_tensor = torch.from_numpy(label).long()
        edge_tensor = torch.from_numpy(edge).long()
        
        if self.split == 'valid' and self.require_paired_data:
            segmentation_mask_filepath_list = str(segmentation_mask_file).split('/')
            segmentation_mask_filename = segmentation_mask_filepath_list[-1]
            dir_name = segmentation_mask_filepath_list[-3]
            filename_id = segmentation_mask_filename.split('_')[-1]
            img_filename = '_'.join(['img', filename_id])
            img_filepath_list = segmentation_mask_filepath_list
            img_filepath_list[-2] = 'imgs'
            img_filepath_list[-1] = img_filename
            img_file = '/'.join(img_filepath_list)
            if not os.path.exists(img_file):
                img_filename = filename_id.zfill(14)
                img_filepath_list[-1] = img_filename
                img_file = '/'.join(img_filepath_list)
            img = Image.open(img_file)

            if self.resize:
                img = img.resize((self.shape_resize[1], self.shape_resize[0]))
            img_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ])
            img_tensor = img_transform(img)
            img_tensor = img_tensor[:, :-60, :]

            label_original_tensor = torch.from_numpy(label_original).long()
            return event_tensor, img_tensor, label_tensor, label_original_tensor, edge_tensor
        # return event_tensor, label_tensor, edge_tensor
        return event_tensor, label_tensor, edge_tensor, rgb_tensor
        
