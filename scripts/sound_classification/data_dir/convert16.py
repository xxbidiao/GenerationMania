import soundfile as sf
import os

for subdir, dirs, files in os.walk("./"):
    for file in files:
        if "convert16" not in file:
            print os.path.join(subdir, file)
            data, samplerate = sf.read(os.path.join(subdir, file))
            sf.write(os.path.join(subdir, file), data, 16000, subtype='PCM_16')
