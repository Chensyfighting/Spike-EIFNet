import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from utils.metric.metric import get_iou
from dataset.event.base_trainer import BaseTrainer
from spikingjelly.activation_based import functional
from tqdm import tqdm
from torch.nn import functional as F
from PIL import Image
from torchvision.utils import save_image
from utils.losses.loss import  CrossEntropyLoss2d, ProbOhemCrossEntropy2d, FocalLoss2d
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser(description='Spike-EIFNet: Lightweight Spike-driven Event-Image Fusion Network for Accurate and Efficient Semantic Segmentation')
    parser.add_argument('--model', default="SpikeEIFNet", help="model name")
    parser.add_argument('--dataset', default="DSEC_events", help="dataset: DDD17_events or DSEC_events")
    parser.add_argument('--input_size', type=str, default="480,640", help="DDD17_events:200,346,DSEC_events:480,640")
    parser.add_argument('--dataset_path', type=str, default="/nfs/share_hdd_6t/Datasets/DSEC_events",
                        help='Please enter the path to your DDD17/DSEC-Semantic dataset directory')
    parser.add_argument('--workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when testing")
    parser.add_argument('--checkpoint', type=str,default="/home/SpikeEIFNet/checkpoint/DSEC_events/SpikeEIFNetDSEC.pth",
                        help='Load the pretrained .pth checkpoint.')
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--save', type=bool, default=True, help="Save the predicted image")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="1", type=str, help="gpu ids (default: 0)")

    parser.add_argument('--split', type=str, default="test", help="spilt in ['train', 'test', 'valid']"
                                                                   "ddd17 valid"
                                                                    'dsec test')
    parser.add_argument('--nr_events_data', type=int, default=1)
    parser.add_argument('--delta_t_per_data', type=int, default=50)
    parser.add_argument('--nr_events_window', type=int, default=100000, help='DDD17:32000,DSEC:100000')
    parser.add_argument('--data_augmentation_train', type=bool, default=False)
    parser.add_argument('--event_representation', type=str, default="voxel_grid")
    parser.add_argument('--nr_temporal_bins', type=int, default=5)
    parser.add_argument('--require_paired_data_train', type=bool, default=True)
    parser.add_argument('--require_paired_data_val', type=bool, default=True)
    parser.add_argument('--separate_pol', type=bool, default=False)
    parser.add_argument('--normalize_event', type=bool, default=True)
    parser.add_argument('--fixed_duration', type=bool, default=False)

    # event datasets
    parser.add_argument('--use_ohem', type=bool, default=True, help='OhemCrossEntropy2d Loss for event dataset')
    parser.add_argument("--use_earlyloss", type=bool, default=True, help='Use early-surpervised training for event dataset')
    parser.add_argument("--balance_weights", type=list, default=[1.0, 0.4], help='balance between out and early_out')

    args = parser.parse_args()

    return args


def test(args, val_loader, model, criterion, device):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # evaluation mode
    model.eval()
    total_batches = len(val_loader)

    epoch_loss = []
    data_list = []
    for i, (event, label, _, RGB) in enumerate(val_loader):

        with torch.no_grad():
            event_var = event.cuda(device)
            label_var = label.long().cuda(device)
            RGB_var = RGB.cuda(device)
            functional.reset_net(model)
            start_time = time.time()
            output = model(event_var, RGB_var)


        time_taken = time.time() - start_time
        print("[%d/%d]  time: %.2f" % (i + 1, total_batches, time_taken))
        loss = criterion(output, label_var)
        epoch_loss.append(loss[0].item())

        if len(output) == 2:
            output = output[0]
        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        # DDD17:['flat','background','object','vegetation','human','vehicle']
        colors = [(128, 64,128),
                 (70 , 70, 70),
                 (220,220,  0),
                 (107,142, 35),
                 (220, 20, 60),
                 (0  ,  0,142)]
        # DSEC
        # colors = [(0,  0,  0),
        #              (70 ,70, 70),
        #              (190,153,153),
        #              (220, 20,60),
        #              (153,153,153),
        #              (128, 64,128),
        #              (244, 35,232),
        #              (107,142, 35),
        #              (0,  0,  142),
        #              (102,102,156),
        #              (220,220,  0)]
        colors = np.array(colors) / 255.0

        plt.figure(figsize=(6, 6))
        plt.imshow(RGB.squeeze(0).permute(1, 2, 0))
        plt.show()
        plt.figure(figsize=(6, 6))
        plt.imshow(event[0, 1].squeeze(), cmap='gray', vmin=-2, vmax=2)
        plt.show()
        colored_gt = np.zeros((gt.shape[0], gt.shape[1], 3))
        for j in range(6):  # 6 or 11
            colored_gt[gt == j] = colors[j]
        plt.figure(figsize=(6, 6))
        plt.imshow(colored_gt)
        plt.axis('off')
        plt.show()
        colored_image = np.zeros((output.shape[0], output.shape[1], 3))
        for k in range(6):  # 6 or 11
            colored_image[output == k] = colors[k]
        plt.figure(figsize=(6, 6))
        plt.imshow(colored_image)
        plt.axis('off')
        plt.show()

        data_list.append([gt.flatten(), output.flatten()])

    meanIoU, per_class_iu, acc = get_iou(data_list, args.classes)
    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    return meanIoU, per_class_iu, acc, average_epoch_loss_val


def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    h, w = map(int, args.input_size.split(','))
    print(args)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        if args.cuda:
            print("=====> use gpu id: '{}'".format(args.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            if not torch.cuda.is_available():
                raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(args.model, num_classes=args.classes, ohem=args.use_ohem, early_loss=args.use_earlyloss)
    functional.set_step_mode(model, step_mode='m')


    if args.dataset in ['DDD17_events', 'DSEC_events']:
        if args.use_ohem:
            min_kept = int(args.batch_size // len(args.gpus) * h * w // 16)
            criteria = ProbOhemCrossEntropy2d(use_weight=False, ignore_label=ignore_label, thresh=0.7, min_kept=min_kept, balance_weights=args.balance_weights)
        elif args.use_focal:
            criteria = FocalLoss2d(weight=None, ignore_index=ignore_label, balance_weights=args.balance_weights)
        else:
            criteria = CrossEntropyLoss2d(weight=None, ignore_label=ignore_label)



    if args.cuda:
        model = model.cuda(device)  # using GPU for inference
        criteria = criteria.cuda(device)
        cudnn.benchmark = True

    if args.save:
        if not os.path.exists(args.save_seg_dir):
            os.makedirs(args.save_seg_dir)


    # DDD17/DSEC datasets
    base_trainer_instance = BaseTrainer()
    trainLoader, testLoader = base_trainer_instance.createDataLoaders(args)


    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=====> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model'])
        else:
            print("=====> no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    print("=====> beginning validation")
    print("validation set length: ", len(testLoader))

    mIOU_val, per_class_iu, _, _ = test(args, testLoader, model, criteria, device)
    print("mIOU_val:",mIOU_val)
    print("per_class_iu:",per_class_iu)



if __name__ == '__main__':

    args = parse_args()
    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)

    if args.dataset == 'DDD17_events':
        args.classes = 6
    elif args.dataset == 'DSEC_events':
        args.classes = 11
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    ignore_label = 255
    test_model(args)
