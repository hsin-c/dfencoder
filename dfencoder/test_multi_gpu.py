import os
import time
import numpy as np
import itertools
from datetime import datetime
import argparse
import pandas as pd
import tqdm
import json
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader

from dfencoder.autoencoder import AutoEncoder, EncoderDataFrame
from collections import defaultdict


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
FEATURE_COLUMNS = [
    'app_name',
    'client_app',
    'device_name',
    'browser_type',
    'os',
    'error_reason',
    'risk_event_types',
    'country',
    'city',
    'log_count',
    'location_incr',
    'app_incr'
]
INFO_COLUMNS = ['user', 'time', 'comb_risk_level_med_high', 'signin_risk_level_med_high', 'agg_risk_level_med_high', 'individual_model']
OUTPUT_DIR = 'train_result_0221'
TRAIN_FOLDER = 'train_data'
VALIDATION_FOLDER = 'validation_data'
INFERENCE_FOLDER = 'inference_data'

class MyDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __iter__(self):
        # unbatch to get rid of the first dimention of 1 intorduced by DataLoaders batching (batch size is always set to 1)
        for data_d in super().__iter__():
            data_d['data'] = {k: v[0] if type(v) != list else [_v[0] for _v in v] for k, v in data_d['data'].items()}
            yield data_d


class MyDataset(Dataset):
    def __init__(
        self, data_folder, batch_size, model, 
        shuffle_rows_in_batch=True, preload_data_into_memory=False, 
        include_original_input_tensor=False, 
        # ^ normally want swapped input only for training. user may want to get both swapped and orignal for validation
    ):
        self.model = model # to help with preprocessing
        self.data_folder = data_folder
        self.filenames = sorted(os.listdir(data_folder))

        self.preloaded_data = None
        if preload_data_into_memory:
            self.preloaded_data = {
                fn: pd.read_csv(f'{self.data_folder}/{fn}', index_col=0, dtype={'risk_event_types': 'str'})
                for fn in self.filenames
            }

        self.file_sizes = {
            fn: self._get_file_len(fn) - 1 if not self.preloaded_data else len(self.preloaded_data[fn])
            for fn in self.filenames
        }
        self.len = sum(v for v in self.file_sizes.values())
        self.batch_size = batch_size
        self.shuffle_rows_in_batch = shuffle_rows_in_batch
        self.include_original_input_tensor = include_original_input_tensor
        
    def _get_file_len(self, fn):
        with open(f'{self.data_folder}/{fn}') as f:
            count = sum(1 for line in f)
        return count
    
    def __len__(self):
        return int(np.ceil(self.len / self.batch_size))

    def __iter__(self):
        # iterate through the whole dataset by batch without any shuffling
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        # print(idx)
        # Assuming batch size << row count in each file
        start = idx*self.batch_size
        end = (idx+1)*self.batch_size
        
        data = []
        curr_cnt = 0
        for fn in self.filenames:
            f_count = self.file_sizes[fn]
            curr_cnt = f_count
            
            if start < curr_cnt and end <= curr_cnt:
                data.append(self._get_data_from_filename(fn)[start: end])
                return self._preprocess(pd.concat(data), batch_index=idx)
            
            if start < curr_cnt and end > curr_cnt:
                data.append(self._get_data_from_filename(fn)[start:])
            start = max(0, start - curr_cnt)
            end = end - curr_cnt

        # clear out last batch
        return self._preprocess(pd.concat(data), batch_index=idx)

    def _get_data_from_filename(self, filename):
        if self.preloaded_data:
            return self.preloaded_data[filename]
        return pd.read_csv(f'{self.data_folder}/{filename}', index_col=0, dtype={'risk_event_types': 'str'})

    def _preprocess(self, df, batch_index):
        df = self.model.prepare_df(df)
        if self.shuffle_rows_in_batch:
            df = df.sample(frac=1.0)
        df = EncoderDataFrame(df)
        input_df = df.swap(likelihood=self.model.swap_p)
        in_sample_tensor = self.model.build_input_tensor(input_df)
        num_target, bin_target, codes = self.model.compute_targets(df)
        data_d = {
            'input_swapped': in_sample_tensor, 
            'num_target': num_target, 
            'bin_target': bin_target, 
            'cat_target': codes
        }
        if self.include_original_input_tensor:
            data_d['input_original'] = self.model.build_input_tensor(df)
        return {'batch_index': batch_index, 'data': data_d}

    def get_preloaded_data(self):
        if self.preloaded_data is None:
            return None
        return pd.concat(pdf for pdf in self.preloaded_data.values())

def get_training_dataloader(
    model, data_folder, data_load_batch_size, rank, world_size, shuffle=True, pin_memory=False, num_workers=0
):
    dataset = MyDataset(data_folder, data_load_batch_size, model=model, 
    shuffle_rows_in_batch=True, preload_data_into_memory=False, include_original_input_tensor=False)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
    dataloader = MyDataLoader(dataset, batch_size=1, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    return dataloader


def get_validation_dataset(data_folder, model):
    # loading the whole set into memory assuming host memory is big enough
    dataset = MyDataset(
        data_folder, 
        model.eval_batch_size, 
        model, 
        shuffle_rows_in_batch=False, 
        preload_data_into_memory=True,
        include_original_input_tensor=True,
    )
    return dataset

def get_inference_dataset(data_folder, model, inf_batch_size):
    # loading the whole set into memory assuming host memory is big enough
    dataset = MyDataset(
        data_folder, 
        inf_batch_size, 
        model, 
        shuffle_rows_in_batch=False, 
        preload_data_into_memory=True,
        include_original_input_tensor=True,
    )
    return dataset


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    torch.cuda.set_device(rank)
    print(f"Running basic DDP example on rank {rank}.")

    setup(rank, world_size)

    preset_cats = json.load(open('azure_202302040659_preset_cats_0207.json', 'r'))
    preset_numerical_scaler_params = {
        'log_count': {
            'scaler_type': 'standard', 
            'scaler_attr_dict': {'mean': 49.781, 'std': 84.306},
            'mean': 49.781,
            'std': 84.306,
        },
        'location_incr': {
            'scaler_type': 'standard', 
            'scaler_attr_dict': {'mean': 1.410, 'std': 1.360},
            'mean': 1.410,
            'std': 1.360,
        },
        'app_incr': {
            'scaler_type': 'standard', 
            'scaler_attr_dict': {'mean': 2.788, 'std': 2.493},
            'mean': 2.788,
            'std': 2.493,
        },
    }
    
    model = AutoEncoder(
        encoder_layers = [512, 500], #layers of the encoding part
        decoder_layers = [512], #layers of the decoding part
        activation='relu', #activation function
        swap_p=0.2, #noise parameter
        lr = 0.01, # learning rate
        lr_decay=0.99, # learning decay
        batch_size=2048,
        logger='basic', 
        verbose=True,
        progress_bar=False,
        optimizer='adam', #SGD optimizer is selected(Stochastic gradient descent)
        scaler='standard', #feature scaling method
        min_cats=1, #cut off for minority categories
        device=rank,
        preset_numerical_scaler_params=preset_numerical_scaler_params,
        binary_feature_list=[],
        preset_cats=preset_cats,
        eval_batch_size=100000, # as big as possible to save time
        patience=5,
    )
    model.build_model()

    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    
    # get optimizer
    model.optim = model.build_optimizer(distributed_model=ddp_model)
    if model.lr_decay is not None:
        model.lr_decay = torch.optim.lr_scheduler.ExponentialLR(model.optim, model.lr_decay)

    # prepare the dataloader
    dataloader = get_training_dataloader(model, data_folder=TRAIN_FOLDER, data_load_batch_size=1024, rank=rank, world_size=world_size)
    # load validation set
    val_dataset = get_validation_dataset(VALIDATION_FOLDER, model)

    model.train()
    # early stopping
    count_es = 0
    last_net_loss = float('inf')
    rank_stats = defaultdict(list)
    start_time = time.time()
    for epoch in range(30):
        epoch_start = time.time()
        if model.verbose:
            print(f'\nR{rank} training epoch {epoch + 1}... ({len(dataloader.dataset)}) batches)')
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader.sampler.set_epoch(epoch)       
        
        train_loss_sum = 0
        train_loss_count = 0
        for step, data_d in enumerate(dataloader):
            loss = model.fit_batch(**data_d['data'])

            train_loss_count += 1
            train_loss_sum += loss

            if step % 10 == 0:
                print(f'\tR{rank} {epoch}-th epoch, processed {step} batches...')
        
        if model.lr_decay is not None:
            model.lr_decay.step()

        train_done_time = time.time()

        # run validation
        curr_net_loss = model.run_validation(val_dataset, rank)
        print(f'R{rank} Loss: {round(last_net_loss,4)}->{round(curr_net_loss,4)}')
        if curr_net_loss > last_net_loss:
            count_es += 1
            if model.verbose:
                print(f'R{rank} Loss went up. Early stop count: {count_es}')

            if count_es >= model.patience:
                if model.verbose:
                    print('Early stopping: early stop count({}) >= patience({})'.format(count_es, model.patience))
                break
        else:
            if model.verbose:
                print(f'R{rank} Loss went down :) Reset count for earlystop to 0')
            count_es = 0
        last_net_loss = curr_net_loss

        validation_done_time = time.time()

        model.logger.end_epoch()
        rank_stats[f'rank_{rank}'].append({
            'epoch': epoch,
            'train_loss': train_loss_sum/train_loss_count,
            'val_loss': curr_net_loss,
            'train_time_sec': train_done_time - epoch_start,
            'validation_time_sec': validation_done_time - train_done_time,
        })

    all_training_end_time = time.time()
    rank_stats[f'rank_{rank}'].append({
            'epoch': 'final_stats',
            'processing_time_sec': all_training_end_time - start_time,
    })
    # we have to create enough room to store the collected objects
    stats = [None for _ in range(world_size)]
    # the first argument is the collected lists, the second argument is the data unique in each process
    dist.all_gather_object(stats, rank_stats)

    if rank == 0:
        print(json.dumps(stats, indent=4)) 
        model.populate_loss_stats_from_dataset(val_dataset)
        
        # Inference
        result = run_inference(model, INFERENCE_FOLDER)
        inference_done_time = time.time()

        json.dump(stats, open(f'{OUTPUT_DIR}/statistics_by_rank.json', 'w'))
        result.to_csv(f'{OUTPUT_DIR}/inference_result.csv', index=False)
        write_done_time = time.time()

        json.dump(
            {
                'inference_time_sec': inference_done_time - all_training_end_time,
                'write_time_sec': write_done_time - inference_done_time,
            },
            open(f'{OUTPUT_DIR}/inference_time_stats.json', 'w')
        )
        
    cleanup()
    return

def run_inference(model, inference_folder):
    start = time.time()
    inf_dataset = get_inference_dataset(inference_folder, model, inf_batch_size=2**17)
    inference = inf_dataset.get_preloaded_data()
    load_time = time.time()
    print('\t\tInference data loading done ({} sec)'.format(round(load_time - start, 2)))
    
    result = (
        model
        .get_results_from_dataset(inf_dataset, preloaded_df=inference, return_abs=True)
        .join(
            inference[INFO_COLUMNS], 
            how='inner', 
            rsuffix='_'
        )
    )
    inference_time = time.time()
    print('\t\tInference done ({} sec)'.format(round(inference_time - load_time, 2)))

    result.fillna('nan', inplace=True)
    result['date'] = result.time.str[:10]
    result['anomaly_score'] = result['mean_abs_z']
    return result


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    world_size = 2

    print("We have available ", torch.cuda.device_count(), "GPUs! but using ",world_size," GPUs")

    #########################################################
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)    
    #########################################################