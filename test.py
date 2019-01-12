
import numpy as np

import preprocessing


def test_filegen():
	fs = preprocessing.get_wavs_from_dir("Data/Speaker_A/WAV")
	wav = fs.__next__()
	rg = preprocessing.MonoRawGenerator().gen(wav)
	fg = preprocessing.FFTGenerator().gen(rg)
	data = fg.__next__()
	data = np.concatenate([data, fg.__next__()], axis=1)
	print(data.dtype)
	print(data.shape)
	wfd = preprocessing.WavFFTData(data)
	wav = wfd.to_raw()
	print(wav)
	print(wav.data)
	wav.create_wav_file("out.wav")

def test_2():
	fs = preprocessing.get_wavs_from_dir("Data/Speaker_A/WAV")
	wav = fs.__next__()
	rg = preprocessing.MonoRawGenerator().gen(wav)
	fg = preprocessing.FFTGenerator().gen(rg)
	sg = preprocessing.SentenceSeperator().gen(fg)
	for i in range(20):
		sample = sg.__next__()
		sfft = preprocessing.WavFFTData(sample)
		sfft.to_raw().create_wav_file("out_%d.wav" % i)

def main():
	sg = preprocessing.DataSource().gen("Data/Speaker_A/WAV")
	for i, sentence in enumerate(sg):
		preprocessing.WavFFTData(sentence).to_raw().create_wav_file("out_%d.wav" % i)
		if i > 10:
			break

if __name__ == "__main__":
	main()

