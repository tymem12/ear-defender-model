from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa

# this is the mostly correct dataset

class Dataset_Custom(Dataset):
    def __init__(self, list_IDs, base_dir=os.getenv('AUDIO_STORAGE')):
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
            X, fs = librosa.load(audio_path, sr=16000)

            # Calculate how many full chunks we can make from the audio
            
            num_full_chunks = len(X) // self.cut
            remainder = len(X) % self.cut

            # Create tuples of (utt_id, segment number) for each chunk
            for i in range(num_full_chunks):
                self.audio_segments.append((utt_id, i))

            # print(f'audio: {utt_id}  len(X): {len(X)},  num_chunks: {num_full_chunks}, remainder: {remainder}')

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
    # Pad the audio if it is shorter than max_len
    padded_x = np.pad(x, (0, max_len - x_len), 'constant')
    return padded_x



if __name__ == '__main__':
    print('works')
    # Sample test data for list_IDs
    list_IDs = ["audio_1.m4a", "audio_2.m4a"]
    base_dir = '../datasets/test_dataset'

    # Instantiate the Dataset_Custom class
    dataset = Dataset_Custom(list_IDs, base_dir=base_dir)

    # Test 1: Print basic attributes
    print("Testing basic attributes:")
    print("list_IDs:", dataset.list_IDs)
    print("base_dir:", dataset.base_dir)
    print("cut (audio chunk size):", dataset.cut)
    print("Number of audio segments prepared:", len(dataset.audio_segments))
    print("First few audio segments:", dataset.audio_segments[:3])  # print only the first 3 segments for brevity
    print()

    # Test 2: Check the length of the dataset
    print("Testing length of dataset:")
    print("Total number of audio segments (len):", len(dataset))
    print()

    # Test 3: Test __getitem__ method
    print("Testing __getitem__:")
    sample_index = 0
    x_inp, (utt_id, segment_num) = dataset[sample_index]
    print(f"Sample input from __getitem__ at index {sample_index}:")
    print("Audio tensor shape:", x_inp.shape)
    print("Utt_id:", utt_id)
    print("Segment number:", segment_num)
    print()

    # Test 4: Iterating over the dataset
    print("Testing iteration over dataset:")
    for idx, (x_inp, (utt_id, segment_num)) in enumerate(dataset):
        print(f"Sample {idx}: utt_id = {utt_id}, segment_num = {segment_num}, tensor shape = {x_inp.shape}")
        if idx == 5:  # Break after 3 samples to avoid too much printing
            break
    print()

    # Test 5: Padding functionality test
    print("Testing padding:")
    short_audio = np.array([0.1, 0.2, 0.3])  # A short audio array for padding test
    padded_audio = pad(short_audio)
    print("Original short audio:", short_audio)
    print("Padded audio length:", len(padded_audio))
    print("Padded audio array:", padded_audio)
    print()
