
import numpy as np

import scipy.signal as signal
import scipy.io.wavfile as wav

import random

import glob

import torch
import torch.nn as nn

WAV_DATARATE = 44100

# Skip intro
SKIP_SECONDS = 60
FFT_GENERATOR_WIN_SECONDS = 20
FFT_GENERATOR_OVERLAP_SECONDS = 2
MAX_INT16 = (1 << 15) - 1

# WAV is segmented to save precious RAM...
MAX_WAV_SEGMENT_SECONDS = 1800

def get_wavs_from_dir(dir_path, shuffle=True):
	files = []
	for file in glob.glob(dir_path + "/*.wav"):
		files.append(file)
	if shuffle:
		random.shuffle(files)
	for file in files:
		data = WavRawData(wav.read(file)[1])
		data = data.subdata(SKIP_SECONDS*WAV_DATARATE)
		datum = data.subdata(end=MAX_WAV_SEGMENT_SECONDS*WAV_DATARATE)
		while(datum.data.shape[0] > 0):
			yield datum
			data = data.subdata(MAX_WAV_SEGMENT_SECONDS*WAV_DATARATE)
			datum = data.subdata(end=MAX_WAV_SEGMENT_SECONDS*WAV_DATARATE)

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

	def to_mono(self):
		data = self.data.astype(np.float64)
		data = data[:,0] + data[:,1]
		return data


class MonoRawGenerator:
	def __init__(self):
		self.ptr = 0

	def gen(self, wav_raw_data):
		monodata = wav_raw_data.to_mono()
		while self.ptr+FFT_GENERATOR_WIN_SECONDS*WAV_DATARATE <= monodata.shape[0]:
			yield monodata[self.ptr:self.ptr+FFT_GENERATOR_WIN_SECONDS*WAV_DATARATE]
			self.ptr += (FFT_GENERATOR_WIN_SECONDS-FFT_GENERATOR_OVERLAP_SECONDS) * WAV_DATARATE


WINDOWS_IN_SEC = 60
WINDOW = signal.get_window(('gaussian', 2*WAV_DATARATE//WINDOWS_IN_SEC), 6*WAV_DATARATE//WINDOWS_IN_SEC)
STFT_PARAMS = {
	'window': WINDOW,
	'nperseg': 6*WAV_DATARATE//WINDOWS_IN_SEC,
	'noverlap': 5*WAV_DATARATE//WINDOWS_IN_SEC
}

class WavFFTData:
	def __init__(self, fft_data):
		self.data = fft_data

	@classmethod
	def from_WavRawData(cls, wav_raw_data):
		# Convert to mono
		data = wav_raw_data.to_mono()
		return cls(signal.stft(data, **STFT_PARAMS)[2])

	@classmethod
	def from_mono(cls, wav_mono):
		return cls(signal.stft(data, **STFT_PARAMS)[2])

	def to_raw(self):
		data = (signal.istft(self.data, **STFT_PARAMS)[1])
		# Renormalize output
		data = data / data.max() * 0.4999 * MAX_INT16
		data = data.astype(np.int16)
		return WavRawData.from_mono(data)

class FFTGenerator:
	def __init__(self):
		pass

	def gen(self, mono_generator):
		for monodata in mono_generator:
			data = signal.stft(monodata, **STFT_PARAMS)[2]
			data = data[:,WINDOWS_IN_SEC:-WINDOWS_IN_SEC-1]
			yield data

MIN_SECONDS = 5
SEPERATORS_PER_SECOND = 6

class SentenceSeperator:
	def __init__(self, volume_threshold=400., seperator_length=WINDOWS_IN_SEC//SEPERATORS_PER_SECOND):
		self.volume_threshold = volume_threshold
		self.seperator_length = seperator_length
		self.pool = nn.MaxPool1d(seperator_length)

	def gen(self, fft_gen):
		sample = fft_gen.__next__()
		volume = self.sample_to_volume(sample)
		ptr = 0
		fst = True
		while True:
			if volume.shape[0] < WINDOWS_IN_SEC*MIN_SECONDS:
				tmp = fft_gen.__next__()
				sample = np.concatenate([sample, tmp], axis=1)
				volume = np.concatenate([volume, self.sample_to_volume(tmp)])
				continue
			pooled_volume = torch.from_numpy(volume)
			pooled_volume = torch.stack([pooled_volume])
			pooled_volume = torch.stack([pooled_volume])
			pooled_volume = self.pool(pooled_volume).numpy()
			pooled_volume = pooled_volume[0,0]
			nbreak = self.find_next_break(pooled_volume, ptr)
			if nbreak == None:
				tmp = fft_gen.__next__()
				sample = np.concatenate([sample, tmp], axis=1)
				volume = np.concatenate([volume, self.sample_to_volume(tmp)])
				continue
			ptr = nbreak * self.seperator_length
			if fst:
				fst = False
			else:
				yield sample[:,:ptr]
			sample = sample[:,ptr:]
			volume = volume[ptr:]
			ptr = MIN_SECONDS*SEPERATORS_PER_SECOND


	def sample_to_volume(self, sample):
		volume = (sample*sample.conj()).real
		volume = np.power(volume, 0.5)
		# Assuming: Volume ~ Frequency*Magnitude
		volume = np.einsum('fv,f->v', volume, np.array(range(sample.shape[0]), dtype=np.float64)/sample.shape[0])
		return volume

	def find_next_break(self, pooled_volume, ptr):
		l = pooled_volume.shape[0]
		while ptr < l-2:
			if max(pooled_volume[ptr], pooled_volume[ptr+1], pooled_volume[ptr+2]) < self.volume_threshold:
				return ptr+1
			ptr += 1
		return None


class DataSource:
	def __init__(self):
		pass

	def gen(self, folder_path, repeats=True):
		while True:
			for wav in get_wavs_from_dir(folder_path):
				rg = MonoRawGenerator().gen(wav)
				fg = FFTGenerator().gen(rg)
				sg = SentenceSeperator().gen(fg)
				for sentence in sg:
					yield sentence


