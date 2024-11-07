from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from typing import List
import logging
# this is the mostly correct dataset

class Dataset_Custom(Dataset):
    def __init__(self, list_IDs : List[str], base_dir: str):
        '''
        self.list_IDs : list of strings (each string: utt key),
        '''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600 # length of each audio chunk (~4 seconds)
        self.audio_segments = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        '''
        Prepares the dataset by splitting longer audios into multiple chunks and 
        storing their corresponding utt_id and segment number.
        '''

        # print('PREPARE DATASSET')
        for utt_id in self.list_IDs:
            audio_path = os.path.join(self.base_dir, utt_id)


            if not os.path.exists(audio_path):
                logging.info(f"File {audio_path} doesn't exist.")

                # Get the number of files in the base directory and print
                num_files = len([f for f in os.listdir(self.base_dir) if os.path.isfile(os.path.join(self.base_dir, f))])
                logging.info(f"Number of files in the folder '{self.base_dir}': {num_files}")
                continue 

            X, fs = librosa.load(audio_path, sr=16000)

            # Calculate how many full chunks we can make from the audio
            
            num_full_chunks = len(X) // self.cut
            remainder = len(X) % self.cut

            # Create tuples of (utt_id, segment number) for each chunk
            for i in range(num_full_chunks):
                self.audio_segments.append((utt_id, i))


            # Add the remainder as an additional padded chunk if needed
            if remainder > 0:
                self.audio_segments.append((utt_id, num_full_chunks))

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, index):
        utt_id, segment_num = self.audio_segments[index]
        audio_path = os.path.join(self.base_dir, utt_id)
        X, fs = librosa.load(audio_path, sr=16000)

        # Extract the relevant segment of the audio
        start = segment_num * self.cut
        end = min(start + self.cut, len(X))  # Ensure we don't go out of bounds
        X_segment = X[start:end]

        # If the segment is shorter than 'cut', apply padding
        if len(X_segment) < self.cut:
            X_segment = pad(X_segment, self.cut)

        x_inp = Tensor(X_segment)
        return x_inp, (utt_id, segment_num)

def pad(x, max_len=64600):
    x_len = len(x)
    if x_len >= max_len:
        return x[:max_len]
    # Repeat x until reaching max_len, then slice to the exact length
    padded_x = np.tile(x, (max_len // x_len) + 1)[:max_len]
    return padded_x


if __name__ == '__main__':
    pass