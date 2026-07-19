"""
Adapted from: https://github.com/uzh-rpg/rpg_ev-transfer
"""
from __future__ import division

import torch
import numpy as np


class BaseTrainer(object):
    """BaseTrainer class to be inherited"""

    def __init__(self):
        # override this function to define your model, optimizers etc.
        super(BaseTrainer, self).__init__()

    
    def getDataloader(self, dataset_name):
        """Returns the dataset loader specified in the settings file"""
        if dataset_name == 'DSEC_events':
            from dataset.event.DSEC_events_loader import DSECEvents
            return DSECEvents
        # elif dataset_name == 'Cityscapes_gray':
        #     from dataset.event.cityscapes_loader import CityscapesGray
        #     return CityscapesGray
        elif dataset_name == 'DDD17_events':
            from dataset.event.ddd17_events_loader import DDD17Events
            return DDD17Events

    def createDataLoaders(self, args):
        if args.dataset == 'DSEC_events':
            out = self.createDSECDataset(args.dataset,
                                         args.dataset_path,
                                         args.batch_size,
                                         args.workers,
                                         args.nr_events_data,
                                         args.delta_t_per_data,
                                         args.nr_events_window,
                                         args.data_augmentation_train,
                                         args.event_representation,
                                         args.nr_temporal_bins,
                                         args.require_paired_data_train,
                                         args.require_paired_data_val,
                                         args.separate_pol,
                                         args.normalize_event,
                                         args.classes,
                                         args.fixed_duration)
        elif args.dataset == 'DDD17_events':
            out = self.createDDD17EventsDataset(args.dataset,
                                                args.dataset_path,
                                                args.split,
                                                args.batch_size,
                                                args.workers,
                                                args.nr_events_data,
                                                args.delta_t_per_data,
                                                args.nr_events_window,
                                                args.data_augmentation_train,
                                                args.event_representation,
                                                args.nr_temporal_bins,
                                                args.require_paired_data_train,
                                                args.require_paired_data_val,
                                                args.separate_pol,
                                                args.normalize_event,
                                                args.fixed_duration)
        train_loader, val_loader = out
        return train_loader, val_loader


    def createDSECDataset(self, dataset_name, dsec_dir, batch_size, workers, nr_events_data, delta_t_per_data, nr_events_window,
                          augmentation, event_representation, nr_bins_per_data, require_paired_data_train,
                          require_paired_data_val, separate_pol, normalize_event, semseg_num_classes, fixed_duration):
        """
        Creates the validation and the training data based on the provided paths and parameters.
        """
        dataset_builder = self.getDataloader(dataset_name)

        train_dataset = dataset_builder(dsec_dir=dsec_dir,
                                        nr_events_data=nr_events_data,
                                        delta_t_per_data=delta_t_per_data,
                                        nr_events_window=nr_events_window,
                                        augmentation=augmentation,
                                        mode='train',
                                        event_representation=event_representation,
                                        nr_bins_per_data=nr_bins_per_data,
                                        require_paired_data=require_paired_data_train,
                                        separate_pol=separate_pol,
                                        normalize_event=normalize_event,
                                        semseg_num_classes=semseg_num_classes,
                                        fixed_duration=fixed_duration)
        val_dataset = dataset_builder(dsec_dir=dsec_dir,
                                      nr_events_data=nr_events_data,
                                      delta_t_per_data=delta_t_per_data,
                                      nr_events_window=nr_events_window,
                                      augmentation=False,
                                      mode='val',
                                      event_representation=event_representation,
                                      nr_bins_per_data=nr_bins_per_data,
                                      require_paired_data=require_paired_data_val,
                                      separate_pol=separate_pol,
                                      normalize_event=normalize_event,
                                      semseg_num_classes=semseg_num_classes,
                                      fixed_duration=fixed_duration)

        dataset_loader = torch.utils.data.DataLoader
        train_loader = dataset_loader(train_dataset, 
                                      batch_size=batch_size,
                                      num_workers=workers,
                                      pin_memory=False, 
                                      shuffle=True, 
                                      drop_last=True)
        val_loader = dataset_loader(val_dataset, 
                                    batch_size=batch_size,
                                    num_workers=workers,
                                    pin_memory=False, 
                                    shuffle=False, 
                                    drop_last=True)
        print('DSEC num of batches: ', len(train_loader), len(val_loader))

        return train_loader, val_loader

    def createDDD17EventsDataset(self, dataset_name, root, split_train, batch_size, workers, nr_events_data, delta_t_per_data,
                                 nr_events_per_data,
                                 augmentation, event_representation,
                                 nr_bins_per_data, require_paired_data_train, require_paired_data_val, separate_pol,
                                 normalize_event, fixed_duration):
        """
        Creates the validation and the training data based on the provided paths and parameters.
        """
        dataset_builder = self.getDataloader(dataset_name)

        train_dataset = dataset_builder(root=root,
                                        split=split_train,
                                        event_representation=event_representation,
                                        nr_events_data=nr_events_data,
                                        delta_t_per_data=delta_t_per_data,
                                        nr_bins_per_data=nr_bins_per_data,
                                        require_paired_data=require_paired_data_train,
                                        separate_pol=separate_pol,
                                        normalize_event=normalize_event,
                                        augmentation=augmentation,
                                        fixed_duration=fixed_duration,
                                        nr_events_per_data=nr_events_per_data)
        val_dataset = dataset_builder(root=root,
                                      split='valid',
                                      event_representation=event_representation,
                                      nr_events_data=nr_events_data,
                                      delta_t_per_data=delta_t_per_data,
                                      nr_bins_per_data=nr_bins_per_data,
                                      require_paired_data=require_paired_data_val,
                                      separate_pol=separate_pol,
                                      normalize_event=normalize_event,
                                      augmentation=False,
                                      fixed_duration=fixed_duration,
                                      nr_events_per_data=nr_events_per_data)
        
        dataset_loader = torch.utils.data.DataLoader
        train_loader = dataset_loader(train_dataset, 
                                      batch_size=batch_size,
                                      num_workers=workers,
                                      pin_memory=False, 
                                      shuffle=True, 
                                      drop_last=True)
        val_loader = dataset_loader(val_dataset, 
                                    batch_size=batch_size,
                                    num_workers=workers,
                                    pin_memory=False, 
                                    shuffle=False, 
                                    drop_last=True)
        print('DDD17Events num of batches: ', len(train_loader), len(val_loader))

        return train_loader, val_loader
