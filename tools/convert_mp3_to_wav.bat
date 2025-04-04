@echo off
for /r %%i in (*.mp3) do (
    ffmpeg -i "%%i" -acodec pcm_s16le -ar 44100 -ac 2 "%%~dpi%%~ni.wav"
)
