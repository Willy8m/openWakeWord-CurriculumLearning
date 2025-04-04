import os
import argparse
from pydub import AudioSegment

def change_volume_in_folder(folder_path, db_change):
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Check if it's an audio file (e.g., .mp3, .wav)
        if os.path.isfile(file_path) and file_name.endswith(('.mp3', '.wav', '.ogg')):
            # Load the audio file
            audio = AudioSegment.from_file(file_path)
            
            # Change the volume
            modified_audio = audio + db_change
            
            # Save the modified audio to a new file
            new_file_path = os.path.join(folder_path, f"modified_{file_name}")
            modified_audio.export(new_file_path, format=file_name.split('.')[-1])
            print(f"Modified {file_name} and saved as {new_file_path}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Change the volume of audio files in a folder.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing audio files.")
    parser.add_argument("--db_change", type=float, required=True, help="Amount of volume change in dB (positive to increase, negative to decrease).")
    
    args = parser.parse_args()
    
    # Call the function with the parsed arguments
    change_volume_in_folder(args.folder, args.db_change)

if __name__ == "__main__":
    main()