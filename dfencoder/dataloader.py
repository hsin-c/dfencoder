import os
import numpy as np
import pandas as pd
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader


class DFEncoderDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for data_d in super().__iter__():
            if self.batch_size == 1:
                # unbatch to get rid of the first dimention of 1 intorduced by DataLoaders batching 
                # (if batch size is set to 1)
                data_d["data"] = {
                    k: v[0] if type(v) != list else [_v[0] for _v in v]
                    for k, v in data_d["data"].items()
                }
            yield data_d


class DFEncoderDataset(Dataset):
    def __init__(
        self,
        data_folder,
        batch_size,
        preprocess_fn,
        load_data_fn=pd.read_csv,
        shuffle_rows_in_batch=True,
        preload_data_into_memory=False,
    ):
        """
        A dataset class that reads data in batches from a folder and applies preprocessing to each batch.
        * This class assumes that the data is saved in small csv files in one folder.

        Args:
            data_folder (str): the path to the folder containing the data files
            batch_size (int): the size of the batches to read the data in
            preprocess_fn (function): a function to preprocess the data, which should take a pandas dataframe and
                a boolean indicating whether to shuffle rows in batch or not, and return a dictionary containing
                the preprocessed data
            load_data_fn (function): a function for loading data from a provided file path into a pandas dataframe
            shuffle_rows_in_batch (bool): whether to shuffle the rows within each batch
            preload_data_into_memory (bool): whether to preload all the data into memory
                (can speed up data loading if the data can fit into memory)
        """
        self.data_folder = data_folder
        self.filenames = sorted(os.listdir(data_folder))
        self.preprocess_fn = preprocess_fn
        self.load_data_fn = load_data_fn

        self.preloaded_data = None
        if preload_data_into_memory:
            self.preloaded_data = {
                fn: self.load_data_fn(f"{self.data_folder}/{fn}")
                for fn in self.filenames
            }

        self.file_sizes = {
            fn: self._get_file_len(fn) - 1
            if not self.preloaded_data
            else len(self.preloaded_data[fn])
            for fn in self.filenames
        }
        self.len = sum(v for v in self.file_sizes.values())
        self.batch_size = batch_size
        self.shuffle_rows_in_batch = shuffle_rows_in_batch

    def _get_file_len(self, fn, file_include_header_line=True):
        """
        Private method for getting the number of lines in a file.

        Args:
            fn (str): The name of the file to get the length of
            file_include_header_line (bool): Whether the file includes a header line

        Returns:
            int: The number of lines in the file
        """
        with open(f"{self.data_folder}/{fn}") as f:
            count = sum(1 for _ in f)
        return count - 1 if file_include_header_line else count

    @property
    def num_samples(self):
        """Returns the number of samples in the dataset."""
        return sum(self.file_sizes.values())

    def __len__(self):
        """Returns the number of batches in the dataset.
        Under normal circumstances, `Dataset` loads/returns one sample at a time. However, to optimize the loading of
        high-volume csv files, this class loads a batch of csv rows at a time. So this built-in len funciton needs to 
        return the batch count instead of sample count.
        """
        return int(np.ceil(self.len / self.batch_size))

    def __iter__(self):
        """Iterates through the whole dataset by batch in order, without any shuffling."""
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        """
        Gets the item at the given index in the dataset.

        Args:
            idx (int): the index of the item to get

        Returns:
            dict: a dictionary containing the preprocessed data for the current batch
        """
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        data = []
        curr_cnt = 0
        for fn in self.filenames:
            f_count = self.file_sizes[fn]
            curr_cnt = f_count

            if start < curr_cnt and end <= curr_cnt:
                data.append(self._get_data_from_filename(fn)[start:end])
                return self._preprocess(pd.concat(data), batch_index=idx)

            if start < curr_cnt and end > curr_cnt:
                data.append(self._get_data_from_filename(fn)[start:])
            start = max(0, start - curr_cnt)
            end = end - curr_cnt

        # clear out last batch
        return self._preprocess(pd.concat(data), batch_index=idx)

    def _get_data_from_filename(self, filename):
        """
        Returns the data from the given file as a pandas dataframe.

        Args:
            filename (str): The filename of the file to load

        Returns:
            pandas dataframe
        """
        if self.preloaded_data:
            return self.preloaded_data[filename]
        return self.load_data_fn(f"{self.data_folder}/{filename}")

    def _preprocess(self, df, batch_index):
        """
        Preprocesses the given dataframe and returns a dictionary containing the preprocessed data.

        Args:
            df (pandas dataframe): the dataframe to preprocess.
            batch_index (int): the index of the current batch.

        Returns:
            dict: a dictionary containing the preprocessed data for the current batch.
        """
        data = self.preprocess_fn(
            df,
            shuffle_rows_in_batch=self.shuffle_rows_in_batch,
        )
        return {"batch_index": batch_index, "data": data}

    def get_preloaded_data(self):
        """
        Loads all data from the files into memory and returns it as a pandas dataframe.

        Returns:
            pandas dataframe
        """
        if self.preloaded_data is None:
            self.preloaded_data = {
                fn: self.load_data_fn(f"{self.data_folder}/{fn}")
                for fn in self.filenames
            }
        return pd.concat(pdf for pdf in self.preloaded_data.values())


def get_distributed_training_dataloader(
    model, data_folder, load_data_fn, rank, world_size, pin_memory=False, num_workers=0
):
    dataset = DFEncoderDataset(
        data_folder,
        model.batch_size,
        model.preprocess_train_data,
        load_data_fn=load_data_fn,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )
    dataloader = DFEncoderDataLoader(
        dataset,
        batch_size=1,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )
    return dataloader

def get_validation_dataset(model, data_folder, load_data_fn, preload_data_into_memory=True):
    dataset = DFEncoderDataset(
        data_folder, 
        model.eval_batch_size, 
        model.preprocess_validation_data, 
        load_data_fn=load_data_fn,
        shuffle_rows_in_batch=False, 
        preload_data_into_memory=preload_data_into_memory,
    )
    return dataset