#import config
import os
import hashlib
import json
from pathlib import Path
import requests
import time
import logging
import numpy as np
import audio2numpy
import datasets
import openwakeword.data
import openwakeword.utils

from tqdm import tqdm
from scipy.signal import resample
from numpy.lib.format import open_memmap


def main(args):

    # Create audio pre-processing object to get openWakeWord audio embeddings
    F = openwakeword.utils.AudioFeatures()
    
    folder = Path(args.folder)
    # folder = Path(r"F:\data\fma_large")

    clips = [str(i) for i in folder.glob("**/*.wav")]
    directories = list(set(str(file.parent) for file in folder.glob("**/*.wav")))
    user_input = input(f"{len(clips)} .wavs to process. Continue (y/n)? ")
    if user_input != "y":
        exit

    # STEP 1: Get negative example paths, filtering out clips that are too long or too short

    negative_clips, negative_durations = openwakeword.data.filter_audio_paths(
        directories,
        min_length_secs = 1.0, # minimum clip length in seconds
        max_length_secs = 60*30, # maximum clip length in seconds
        duration_method = "header" # use the file header to calculate duration
    )

    print(f"{len(negative_clips)} negative clips after filtering, representing ~{sum(negative_durations)//3600} hours")

    # STEP 2: Use HuggingFace datasets to load files from disk by batches

    audio_dataset = datasets.Dataset.from_dict({"audio": negative_clips})
    audio_dataset = audio_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

    # STEP 3: Get audio embeddings (features) for negative clips and save to .npy file
    # Process files by batch and save to Numpy memory mapped file so that
    # an array larger than the available system memory can be created

    batch_size = 64 # number of files to load, compute features, and write to mmap at a time
    clip_size = 2  # the desired window size (in seconds) for the trained openWakeWord model
    N_total = int(sum(negative_durations)//clip_size) # maximum number of rows in mmap file
    n_feature_cols = F.get_embedding_shape(clip_size)

    output_file = os.path.join(folder.parent, f"{folder.name}.npy")
    output_array_shape = (N_total, n_feature_cols[0], n_feature_cols[1])
    fp = open_memmap(output_file, mode='w+', dtype=np.float32, shape=output_array_shape)

    row_counter = 0
    for i in tqdm(np.arange(0, audio_dataset.num_rows, batch_size)):
        # Load data in batches and shape into rectangular array
        wav_data = [(j["array"]*32767).astype(np.int16) for j in audio_dataset[i:i+batch_size]["audio"]]
        wav_data = openwakeword.data.stack_clips(wav_data, clip_size=16000*clip_size).astype(np.int16)

        # Compute features (increase ncpu argument for faster processing)
        features = F.embed_clips(x=wav_data, batch_size=1024, ncpu=12)
        
        # Save computed features to mmap array file (stopping once the desired size is reached)
        if row_counter + features.shape[0] > N_total:
            fp[row_counter:min(row_counter+features.shape[0], N_total), :, :] = features[0:N_total - row_counter, :, :]
            fp.flush()
            break
        else:
            fp[row_counter:row_counter+features.shape[0], :, :] = features
            row_counter += features.shape[0]
            fp.flush()

    # Release mmap files
    del fp

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description=f"Script to generate openwakeword features from podcasts. Generates file output.npy .")
    
    parser.add_argument(
        '--folder', 
        type=str, 
        help="Absolute path to folder containing .wav audios",
        default="current",
        required=True)
    
    args = parser.parse_args()

    main(args)
    