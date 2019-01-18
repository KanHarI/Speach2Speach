
import math
import random
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal

# This is a content saving transformation, generating "new sounds" 
class RandomTransformer:
	def __init__(self, phase_trans=True, volume_trans=(-3.,3.), freq_shift=0.1, gaussian_amp=(3, 100., 3.)):
		self.phase_trans = phase_trans
		self.volume_trans = volume_trans
		self.freq_shift = freq_shift
		self.gaussian_amp = gaussian_amp

	def transform(self, sample):
		shape_0 = sample.shape[0]
		if self.phase_trans:
			dphase = random.random()*2*math.pi*1j
			sample *=  np.exp(dphase)
		if self.volume_trans:
			factor = self.volume_trans[0] + random.random()*(self.volume_trans[1]-self.volume_trans[0])
			sample *= np.exp(factor)
		if self.freq_shift:
			factor = 1 + 2*(random.random()-0.5)*self.freq_shift
			sample_real = sample.real
			sample_imag = sample.imag
			sample_real = ndimage.zoom(sample_real, (factor,1))
			sample_imag = ndimage.zoom(sample_imag, (factor,1))
			sample = sample_real + sample_imag*1j
			sample = np.concatenate([sample, np.zeros((max(shape_0 - sample.shape[0], 0), sample.shape[1]))])
			sample = sample[:shape_0]
		if self.gaussian_amp:
			for i in range(random.randint(0, self.gaussian_amp[0])):
				idx = random.randint(0,shape_0)
				gaussian = signal.get_window(('gaussian', random.random()*self.gaussian_amp[1]), 1000) * random.random() * self.gaussian_amp[2]
				gaussian = np.concatenate([np.zeros((shape_0,)), gaussian])
				gaussian = gaussian[-idx:]
				gaussian = np.concatenate([gaussian, np.zeros((shape_0,))])
				gaussian = gaussian[:shape_0]
				gaussian += 1.
				sample = (sample.T * gaussian).T

		return sample


