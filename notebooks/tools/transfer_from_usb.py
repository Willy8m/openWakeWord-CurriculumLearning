import os

source_path = r"D:\OpenWakeWord\openwakeword_features_podcasts_10000h_ca-es_ca_es-es.npy"
destination_path = r"C:\Users\gbartag\source\openwakeword_features_podcasts_10000h_ca-es_ca_es-es.npy"
buffer_size = 1024 * 1024 * 10  # 10 MB chunks

try:
    with open(source_path, 'rb') as src, open(destination_path, 'wb') as dst:
        total_size = os.path.getsize(source_path)
        copied = 0
        
        while chunk := src.read(buffer_size):
            dst.write(chunk)
            copied += len(chunk)
            progress = (copied / total_size) * 100
            print(f"Progress: {progress:.2f}%", end="\r")
    
    print("\nFile transfer completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")