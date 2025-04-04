import wave
import numpy as np
import os

def read_wav_file(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        params = wav_file.getparams()
        n_channels, sampwidth, framerate, n_frames = params[:4]
        audio_data = wav_file.readframes(n_frames)
        audio_signal = np.frombuffer(audio_data, dtype=np.int16)
        
        # If stereo, take only one channel
        if n_channels == 2:
            audio_signal = audio_signal[::2]

        return audio_signal, framerate

def find_start_and_end_indices(audio_signal, threshold, samplerate):
    # Duration for the moving average window (0.1 seconds)
    window_size = int(0.1 * samplerate)

    start_index = None
    end_index = None

    # Find the start index
    for i in range(window_size, len(audio_signal)):
        avg_value = np.mean(np.abs(audio_signal[i - window_size:i]))
        if avg_value >= threshold:
            start_index = max(0, i - window_size)
            break

    # Find the end index
    for i in range(len(audio_signal) - window_size, 0, -1):
        avg_value = np.mean(np.abs(audio_signal[i:i + window_size]))
        if avg_value >= threshold:
            end_index = min(len(audio_signal), i + window_size)
            break

    return start_index, end_index


def cut_audio_segment(audio_signal, start_index, duration, samplerate):
    # Ensure the segment is 1 second long
    total_length = len(audio_signal)
    end_index = start_index + int(duration * samplerate)

    # If the end_index exceeds total length, adjust the start_index accordingly
    if end_index > total_length:
        start_index = max(0, total_length - int(duration * samplerate))
        end_index = total_length

    return audio_signal[start_index:end_index]

def save_wav_file(file_path, audio_signal, samplerate):
    with wave.open(file_path, 'wb') as wav_file:
        # Parameters: nchannels, sampwidth, framerate, nframes, comptype, compname
        wav_file.setparams((1, 2, samplerate, len(audio_signal), 'NONE', 'not compressed'))
        wav_file.writeframes(audio_signal.tobytes())

def process_audio_files(input_directory, output_directory, threshold):
    samplerate = 16000  # Sample rate in Hz

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each .wav file in the input directory
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".wav"):
            input_file_path = os.path.join(input_directory, file_name)
            output_file_path = os.path.join(output_directory, f"trimmed_{file_name}")

            print(f"Processing file: {input_file_path}")

            # Read the input wav file
            audio_signal, framerate = read_wav_file(input_file_path)

            # Ensure the sample rate is correct
            if framerate != samplerate:
                print(f"Expected sample rate: {samplerate}, but got: {framerate}")
                continue

            # Find the start and end indices based on the threshold
            start_index, end_index = find_start_and_end_indices(audio_signal, threshold, samplerate)
            
            if start_index is None or end_index is None or start_index >= end_index:
                print(f"No valid audio segment found in {file_name}.")
                continue

            # Trim the audio signal
            trimmed_audio = audio_signal[start_index:end_index]

            # Save the trimmed segment to the output folder
            save_wav_file(output_file_path, trimmed_audio, samplerate)
            print(f"Trimmed audio saved to {output_file_path}")

if __name__ == "__main__":
    # Get the directory of the script
    input_directory = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(input_directory, "output")
    threshold = 500  # Adjust threshold value as needed

    process_audio_files(input_directory, output_directory, threshold)
