
import numpy as np

import scipy.signal as signal
import scipy.io.wavfile as wav

import random

import glob

WAV_DATARATE = 44100
SKIP_SECONDS = 60

def get_wavs_from_dir(dir_path, shuffle=True):
	files = []
	for file in glob.glob(dir_path + "/*.wav"):
		files.append(file)
	if shuffle:
		random.shuffle(files)
	for file in files:
		yield WavRawData(wav.read(file)[1])

# All files are 44.1 kHz
class WavRawData:
	def __init__(self, data):
		self.data = data

	@classmethod
	def from_mono(cls, samples):
		data = np.einsum("s,c->sc", samples, np.ones((2,), dtype=np.int16))
		return cls(data)


	def subdata(self, start=0, end=-1):
		return WavRawData(self.data[start:end])

	def create_wav_file(self, path):
		wav.write(path, 44100, self.data)



# Those parameters where chosen empirically after expirementation with audio quality of output after 
WINDOW_STDS_IN_SEC = 50
WINDOW = signal.get_window(('gaussian', 2*WAV_DATARATE//WINDOW_STDS_IN_SEC), 6*WAV_DATARATE//WINDOW_STDS_IN_SEC)
STFT_PARAMS = {
	'window': WINDOW,
	'nperseg': 6*WAV_DATARATE//WINDOW_STDS_IN_SEC,
	'noverlap': 5*WAV_DATARATE//WINDOW_STDS_IN_SEC
}

class WavFFTData:
	def __init__(self, WavRawData):
		# Convert to mono
		data = WavRawData.data.astype(np.float64)/2
		data = data[:,0] + data[:,1]
		self.data = signal.stft(data, **STFT_PARAMS)[2]

	def to_raw(self):
		return WavRawData.from_mono((signal.istft(self.data, **STFT_PARAMS)[1]).astype(np.int16))


def main():
	fs = get_wavs_from_dir("Data/S_A/WAV")
	fst_file = fs.__next__()
	fst_file = fst_file.subdata(SKIP_SECONDS*WAV_DATARATE, 2*SKIP_SECONDS*WAV_DATARATE)
	print(fst_file.data)
	wfd = WavFFTData(fst_file)
	print(wfd.data)
	wav = wfd.to_raw()
	print(wav.data)
	wav.create_wav_file("out.wav")
	return wav


if __name__ == "__main__":
	main()

