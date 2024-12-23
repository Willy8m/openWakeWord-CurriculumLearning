#import config
import os
import hashlib
import json
import requests
import time
import logging
import numpy as np
import audio2numpy
import openwakeword.data
import openwakeword.utils

from tqdm import tqdm
from scipy.signal import resample
from numpy.lib.format import open_memmap

class DirectorySpace():

    def __init__(self):
        self.OUTPUT_DIR = "generate_podcasts_features"
        self.LOGS_DIR = os.path.join(self.OUTPUT_DIR, 'logs')
        self.OUTPUT_DIR = os.path.join(self.OUTPUT_DIR, 'output')

class ProcessedIdsList(list):

    def __init__(self, path):
        self.path = path
        super().__init__(self.initialize())

    def initialize(self):
        if os.path.exists(self.path):
            try:
                return np.load(self.path, allow_pickle=True).tolist()
            except Exception as e:
                print(f"Error loading file '{self.path}': {e}")
                return []
        else:
            return []
        
    def update(self, new_id):
        self.append(new_id)
        np_array = np.array(self)
        np.save(self.path, np_array)

def get_logger(log_level: int = logging.DEBUG):

    # Set up the logger
    logger = logging.getLogger('MyLogger')
    logger.setLevel(log_level)  # Set the logging level (e.g., DEBUG, INFO, WARNING)

    # Create a file handler to save logs to a file
    file_handler = logging.FileHandler('generate_podcasts_features.log')
    file_handler.setLevel(log_level)

    # Create a stream handler to display logs in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)

    # Define a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
            
def podcast_api_call(lang: str, max_results: int):

    api_key = "FDP6MLVWZUZP67QMAZG6"
    api_secret = "3X^hPB$d^LEGn7zb3kZtGCQRtftwBtsBdrR76FhH"
    url = "https://api.podcastindex.org/api/1.0/episodes/random?&lang=" + lang + "&max=" + str(max_results)
    
    epoch_time = int(time.time())  # we'll need the unix time
    data_to_hash = api_key + api_secret + str(epoch_time)  # our hash here is the api key + secret + time 
    sha_1 = hashlib.sha1(data_to_hash.encode()).hexdigest()  # which is then sha-1'd
    
    headers = {
        'X-Auth-Date': str(epoch_time),
        'X-Auth-Key': api_key,
        'Authorization': sha_1,
        'User-Agent': 'postcasting-index-python-cli'
    }

    r = requests.post(url, headers=headers)
    
    if r.status_code == 200:
        response = []
        for p in json.loads(r.text)['episodes']:
            response.append({"id": p['id'], "enclosureUrl": p['enclosureUrl']})

        return {'statusCode': 200, 'body': response}
    else:
        return {'statusCode': 500, 'body': json.dumps('Internal service error')}

def download_mp3(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP request errors
        with open(save_path, "wb") as file:
            file.write(response.content)
        success = True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the audio: {e}")
        success = False
    except Exception as e:
        print(f"Error downloading audio: {e}")
        success = False

    finally:
        return success

def read_to_numpy(file_path):
    try:
        audio, sr = audio2numpy.open_audio(file_path)
        if audio.ndim > 1:
            audio = audio[:,0]  # To mono
        step = int(np.round(sr / 16000))
        resampled_array = audio[::step]  # To 16kHz
        success = True
    except Exception as e:
        resampled_array = []
        success = False
        LOG.info(e)

    return success, resampled_array

def get_urls(response):
    urls = []
    if response['statusCode'] == 200:
        for p in response['body']:
            # Check if podcast has been already processed
            if p['id'] in PROCESSED_IDS:
                print(f"Skipping podcast with id {p['id']}: already processed")
                continue
            PROCESSED_IDS.update(p['id'])
            urls.append(p['enclosureUrl'])
    else:
        print("Podcast API not responding with 200")
    
    return urls
    

# def batch_download(urls, file_path):
#     for url in urls:
#         download_mp3(url, file_path)
#         audio = read_to_numpy(file_path)

def main(args):
    
    global LOG
    global PROCESSED_IDS
    lang = args.lang 
    hours = args.hours
    ids_path = args.ids_path

    LOG = get_logger(log_level=logging.INFO)
    F = openwakeword.utils.AudioFeatures()
    PROCESSED_IDS = ProcessedIdsList(ids_path)
    max_results = 16  # urls per request
    #batch_size = 16  # number of files to load, compute features, and write to mmap at a time
    clip_size = 2  # the desired window size (in seconds) for the trained openWakeWord model
    N_total = hours*3600 // clip_size # maximum number of rows in mmap file
    n_feature_cols = F.get_embedding_shape(clip_size)
    
    output_file = "output.npy"
    output_array_shape = (N_total, n_feature_cols[0], n_feature_cols[1])
    fp = open_memmap(output_file, mode='w+', dtype=np.float32, shape=output_array_shape)

    LOG.info(f"Saving {hours} hours worth of features to {N_total} rows of \"{output_file}\" ...")
    row_counter = 0
    end = False
    while(not end):
        
        # Get mp3 urls
        response = podcast_api_call(lang, max_results)
        urls = get_urls(response)
        for url in urls:
            LOG.debug(f"Processing: {url}")

            # Download audio data
            download_time = time.time()
            success = download_mp3(url, "tmp.mp3")
            if not success: continue
            download_time = time.time() - download_time

            # Read audio to numpy array
            read_time = time.time()
            success, audio = read_to_numpy("tmp.mp3")
            if not success: continue
            read_time = time.time() - read_time

            # Divide and stack clips
            stack_time = time.time()
            audio = [(audio*32767).astype(np.int16)]
            audio = openwakeword.data.stack_clips(audio, clip_size=16000*clip_size).astype(np.int16)
            stack_time = time.time() - stack_time

            # Compute features (increase ncpu argument for faster processing)
            compute_time = time.time()
            features = F.embed_clips(x=audio, batch_size=128, ncpu=8)
            compute_time = time.time() - compute_time
            
            # Save computed features to mmap array file (stopping once the desired size is reached)
            if row_counter + features.shape[0] > N_total:
                fp[row_counter:min(row_counter+features.shape[0], N_total), :, :] = features[0:N_total - row_counter, :, :]
                fp.flush()
                end = True
            else:
                fp[row_counter:row_counter+features.shape[0], :, :] = features
                row_counter += features.shape[0]
                fp.flush()

            LOG.info(f"{row_counter} (+{features.shape[0]}) / {N_total} | download_time: {int(download_time)}s | read_time: {int(read_time)}s | stack_time: {int(stack_time)}s | compute_time: {int(compute_time)}s")
            os.remove("tmp.mp3")
            if end: break

    # Release mmap files
    del fp


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description=f"Script to generate openwakeword features from podcasts. Generates file output.npy .")
    
    parser.add_argument(
        '--lang', 
        type=str, 
        help="Language of the podcasts in locale format. (Example: --lang \"ca,ca-es,es-es\")",
        default="ca,ca-es,es-es")
    
    parser.add_argument(
        '--hours', 
        type=int, 
        help="Hours of podcasts to process",
        default=4)
    
    parser.add_argument(
        '--ids_path', 
        type=str,
        help="Path to npy file containing processed ids",
        default="ids.npy")
    
    args = parser.parse_args()

    main(args)

    