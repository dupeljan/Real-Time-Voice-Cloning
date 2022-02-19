import os
import numpy as np
import librosa
import threading

from abc import ABC, abstractmethod
from typing import Tuple, Optional
from enum import Enum
from multiprocessing.pool import Pool
from functools import partial
from itertools import chain
from pathlib import Path
from tqdm import tqdm

from synthesizer import audio
from encoder import inference as encoder
from utils import logmmse


def g2p(text):
    return text


class Dataset(Enum):
    book = "book"
    mozilla = "mozilla"
    librispeech = "librispeech"
    Ruslan = 'Ruslan'


class DatasetPreprocessor:
    def __init__(self, datasets_root: Path, datasets: str, out_dir: Path, n_processes: int, skip_existing: bool,
                 no_alignments: bool, subfolders: str, use_g_to_p: bool, hparams):
        self.datasets_root = datasets_root
        self.datasets = [Dataset[name] for name in datasets.split(',')]
        self.out_dir = out_dir
        self.n_processes = n_processes
        self.skip_existing = skip_existing
        self.hparams = hparams
        self.no_alignments = no_alignments
        self.subfolders = subfolders
        self.use_g_to_p = use_g_to_p
        if use_g_to_p:
            raise Exception('Not supported yet')

    def preprocess(self):
        for dataset_name in self.datasets:
            self.preprocess_dataset(dataset_name)

    def preprocess_dataset(self, dataset_name: Dataset):
        speaker_processor_map = {
            Dataset.book: SpeakerProcessorBook,
            Dataset.mozilla: SpeakerProcessorMozilla,
            Dataset.Ruslan: SpeakerPreprocessorRuslan,
        }
        self._preprocess(dataset_name, speaker_processor_map[dataset_name])

    @staticmethod
    def _dataset_has_several_dir_for_speaker(dataset_name: Dataset) -> bool:
        return dataset_name in [Dataset.librispeech]

    def _preprocess(self, dataset_name: Dataset, speaker_processor: 'SpeakerProcessorBase'):
        # Gather the input directories
        dataset_root = self.datasets_root.joinpath(dataset_name.value)
        input_dirs = list(dataset_root.glob("*"))

        assert all(input_dir.exists() for input_dir in input_dirs)

        # Create the output directories for each output file type
        self.out_dir.joinpath("mels").mkdir(exist_ok=True)
        self.out_dir.joinpath("audio").mkdir(exist_ok=True)

        # Create a metadata file
        metadata_fpath = self.out_dir.joinpath("train.txt")
        metadata_file = metadata_fpath.open("a" if self.skip_existing else "w", encoding="utf-8")

        # Preprocess the dataset
        speaker_dirs = speaker_processor.get_speaker_dir(input_dirs)
        print("\n    ".join(map(str, ["Using data from:"] + speaker_dirs)))
        processor = speaker_processor(
                       out_dir=self.out_dir,
                       skip_existing=self.skip_existing,
                       use_g_to_p=self.use_g_to_p,
                       no_alignments=self.no_alignments,
                       hparams=self.hparams)
        job = Pool(self.n_processes).imap(processor.preprocess_speaker, speaker_dirs)

        for speaker_metadata in tqdm(job, dataset_name.value, len(speaker_dirs), unit="speakers"):
            for metadatum in speaker_metadata:
                metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
        metadata_file.close()

        # Verify the contents of the metadata file
        with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]
        mel_frames = sum([int(m[4]) for m in metadata])
        timesteps = sum([int(m[3]) for m in metadata])
        sample_rate = self.hparams.sample_rate
        hours = (timesteps / sample_rate) / 3600
        print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
              (len(metadata), mel_frames, timesteps, hours))
        print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
        print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
        print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))


class SpeakerProcessorBase(ABC):
    lock = threading.Lock() # Thead save static var
    broken = 0

    def __init__(self, out_dir: Path, use_g_to_p: bool, no_alignments: bool, skip_existing: bool, hparams):
        self.out_dir = out_dir
        self.skip_existing = skip_existing
        self.use_g_to_p = use_g_to_p
        self.no_alignments = no_alignments
        self.hparams = hparams

    def maybe_rescale(self, wav):
        if not self.hparams.rescale:
            return wav
        return wav / np.abs(wav).max() * self.hparams.rescaling_max

    def maybe_g2p(self, text):
        if not self.use_g_to_p:
            return text
        return g2p(text)

    @staticmethod
    @abstractmethod
    def get_speaker_dir(input_dirs):
        pass

    @abstractmethod
    def preprocess_speaker(self, speaker_dir: str):
        pass


class SpeakerProcessorBookBase(SpeakerProcessorBase):
    @staticmethod
    @abstractmethod
    def encode_metadata_line(line: str) -> Optional[Tuple[str, str]]:
        """
        Should return basename, target text
        """
        pass

    def preprocess_speaker(self, speaker_dir: str):
        file_names = list(Path(speaker_dir).rglob('*'))
        uncased_file_names = [str(name).lower() for name in file_names]
        metadata = []
        lines = []
        texts = []
        with open(os.path.join(speaker_dir, 'metadata.csv'), encoding='utf-8') as f:
            for line in f:
                parts = self.encode_metadata_line(line)
                if not parts:
                    print(f'[WARNING] Wrong metadata file for speaker {speaker_dir}.'
                          f' Broken line: {parts}.\nSkip this part.')
                    with self.lock:
                        self.__class__.broken += 1
                    continue
                basename, text = parts
                lines.append(basename)
                texts.append(text)

        texts = self.maybe_g2p(texts)

        for basename, text in zip(lines, texts):
            default_ext = '.wav'
            basename, ext = os.path.splitext(basename)
            if not ext:
                ext = default_ext

            wav_path = os.path.join(speaker_dir, basename + ext)
            if os.path.isfile(wav_path):
                validated_wav_path = wav_path
            else:
                idx = uncased_file_names.index(wav_path.lower())
                validated_wav_path = file_names[idx]
                print(f'[WARNING] Typo in metadata file for speaker {speaker_dir}')
                print(f'Using {validated_wav_path} instead of {wav_path}')

            wav, _ = librosa.load(validated_wav_path, self.hparams.sample_rate)
            wav = self.maybe_rescale(wav)
            metadata.append(process_utterance(wav, text, self.out_dir, basename,
                            self.skip_existing, self.hparams))

        return [m for m in metadata if m is not None]

    @staticmethod
    def get_speaker_dir(input_dirs):
        return input_dirs


class SpeakerProcessorBook(SpeakerProcessorBookBase):
    @staticmethod
    def encode_metadata_line(line: str) -> Optional[Tuple[str, str]]:
        parts = line.strip().split('|')
        if len(parts) != 3:
            return None
        return parts[0], parts[2]


class SpeakerProcessorMozilla(SpeakerProcessorBookBase):
    @staticmethod
    def encode_metadata_line(line: str) -> Optional[Tuple[str, str]]:
        parts = line.strip().split('|')
        if len(parts) != 2:
            return None
        return parts[0], parts[1]


class SpeakerPreprocessorLibriSpeech(SpeakerProcessorBase):
    def preprocess_speaker(self, speaker_dir: str):
        metadata = []
        for book_dir in speaker_dir.glob("*"):
            if self.no_alignments:
                # Gather the utterance audios and texts
                # LibriTTS uses .wav but we will include extensions for compatibility with other datasets
                extensions = ["*.wav", "*.flac", "*.mp3"]
                for extension in extensions:
                    wav_fpaths = book_dir.glob(extension)

                    for wav_fpath in wav_fpaths:
                        # Load the audio waveform
                        wav, _ = librosa.load(str(wav_fpath), self.hparams.sample_rate)
                        wav = self.maybe_rescale(wav)
                        # Get the corresponding text
                        # Check for .txt (for compatibility with other datasets)
                        text_fpath = wav_fpath.with_suffix(".txt")
                        if not text_fpath.exists():
                            # Check for .normalized.txt (LibriTTS)
                            text_fpath = wav_fpath.with_suffix(".normalized.txt")
                            assert text_fpath.exists()
                        with text_fpath.open("r") as text_file:
                            text = "".join([line for line in text_file])
                            text = text.replace("\"", "")
                            text = text.strip()

                        # Process the utterance
                        text = self.maybe_g2p(text)
                        metadata.append(process_utterance(wav, text, self.out_dir, str(wav_fpath.with_suffix("").name),
                                                          self.skip_existing, self.hparams))
            else:
                # Process alignment file (LibriSpeech support)
                # Gather the utterance audios and texts
                try:
                    alignments_fpath = next(book_dir.glob("*.alignment.txt"))
                    with alignments_fpath.open("r") as alignments_file:
                        alignments = [line.rstrip().split(" ") for line in alignments_file]
                except StopIteration:
                    # A few alignment files will be missing
                    continue

                # Iterate over each entry in the alignments file
                for wav_fname, words, end_times in alignments:
                    wav_fpath = book_dir.joinpath(wav_fname + ".flac")
                    assert wav_fpath.exists()
                    words = words.replace("\"", "").split(",")
                    end_times = list(map(float, end_times.replace("\"", "").split(",")))

                    # Process each sub-utterance
                    wavs, texts = split_on_silences(wav_fpath, words, end_times, self.hparams)
                    texts = self.maybe_g2p(texts)
                    for i, (wav, text) in enumerate(zip(wavs, texts)):
                        sub_basename = "%s_%02d" % (wav_fname, i)
                        metadata.append(process_utterance(wav, text, self.out_dir, sub_basename,
                                                          self.skip_existing, self.hparams))

        return [m for m in metadata if m is not None]

    @staticmethod
    def get_speaker_dir(input_dirs):
        return list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))


class SpeakerPreprocessorSST(SpeakerProcessorBase):
    def preprocess_speaker(self, speaker_dir: str):
        metadata = []
        lines = []
        texts = []
        with open(os.path.join(speaker_dir, 'metadata.csv'), encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                lines.append(parts[0])
                with open(os.path.join(speaker_dir, parts[1]), encoding='utf-8') as f2:
                    for line2 in f2:
                        txt_paths = line2
                texts.append(txt_paths)
            texts = g2p(texts)
            for basename, text in zip(lines, texts):
                wav_path = os.path.join(speaker_dir, basename)
                wav, _ = librosa.load(wav_path, self.hparams.sample_rate)

                wav = self.maybe_rescale(wav)
                basename2 = basename.strip().split('/')
                basename3 = "sl_"+basename2[0]+"_"+basename2[1]+"_"+basename2[2]
                text = self.maybe_g2p(text)
                metadata.append(process_utterance(wav, text, self.out_dir, basename3 ,
                                                  self.skip_existing, self.hparams))
        return [m for m in metadata if m is not None]


class SpeakerPreprocessorRuslan(SpeakerProcessorBook):
    @staticmethod
    def get_speaker_dir(input_dirs):
        return [os.path.split(input_dirs[0])[0]]


def split_on_silences(wav_fpath, words, end_times, hparams):
    # Load the audio waveform
    wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    assert len(words) == len(end_times) == len(start_times)
    assert words[0] == "" and words[-1] == ""

    # Find pauses that are too long
    mask = (words == "") & (end_times - start_times >= hparams.silence_min_duration_split)
    mask[0] = mask[-1] = True
    breaks = np.where(mask)[0]

    # Profile the noise from the silences and perform noise reduction on the waveform
    silence_times = [[start_times[i], end_times[i]] for i in breaks]
    silence_times = (np.array(silence_times) * hparams.sample_rate).astype(np.int)
    noisy_wav = np.concatenate([wav[stime[0]:stime[1]] for stime in silence_times])
    if len(noisy_wav) > hparams.sample_rate * 0.02:
        profile = logmmse.profile_noise(noisy_wav, hparams.sample_rate)
        wav = logmmse.denoise(wav, profile, eta=0)

    # Re-attach segments that are too short
    segments = list(zip(breaks[:-1], breaks[1:]))
    segment_durations = [start_times[end] - end_times[start] for start, end in segments]
    i = 0
    while i < len(segments) and len(segments) > 1:
        if segment_durations[i] < hparams.utterance_min_duration:
            # See if the segment can be re-attached with the right or the left segment
            left_duration = float("inf") if i == 0 else segment_durations[i - 1]
            right_duration = float("inf") if i == len(segments) - 1 else segment_durations[i + 1]
            joined_duration = segment_durations[i] + min(left_duration, right_duration)

            # Do not re-attach if it causes the joined utterance to be too long
            if joined_duration > hparams.hop_size * hparams.max_mel_frames / hparams.sample_rate:
                i += 1
                continue

            # Re-attach the segment with the neighbour of shortest duration
            j = i - 1 if left_duration <= right_duration else i
            segments[j] = (segments[j][0], segments[j + 1][1])
            segment_durations[j] = joined_duration
            del segments[j + 1], segment_durations[j + 1]
        else:
            i += 1

    # Split the utterance
    segment_times = [[end_times[start], start_times[end]] for start, end in segments]
    segment_times = (np.array(segment_times) * hparams.sample_rate).astype(np.int)
    wavs = [wav[segment_time[0]:segment_time[1]] for segment_time in segment_times]
    texts = [" ".join(words[start + 1:end]).replace("  ", " ") for start, end in segments]

    # # DEBUG: play the audio segments (run with -n=1)
    # import sounddevice as sd
    # if len(wavs) > 1:
    #     print("This sentence was split in %d segments:" % len(wavs))
    # else:
    #     print("There are no silences long enough for this sentence to be split:")
    # for wav, text in zip(wavs, texts):
    #     # Pad the waveform with 1 second of silence because sounddevice tends to cut them early
    #     # when playing them. You shouldn't need to do that in your parsers.
    #     wav = np.concatenate((wav, [0] * 16000))
    #     print("\t%s" % text)
    #     sd.play(wav, 16000, blocking=True)
    # print("")

    return wavs, texts


def process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename: str,
                      skip_existing: bool, hparams):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.


    # Skip existing utterances if needed
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
    wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)
    if skip_existing and mel_fpath.exists() and wav_fpath.exists():
        return None

    # Trim silence
    if hparams.trim_silence:
        wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)

    # Skip utterances that are too short
    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
        return None

    # Compute the mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    # Skip utterances that are too long
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)

    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text


def embed_utterance(fpaths, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpaths
    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)


def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, n_processes: int):
    wav_dir = synthesizer_root.joinpath("audio")
    metadata_fpath = synthesizer_root.joinpath("train.txt")
    assert wav_dir.exists() and metadata_fpath.exists()
    embed_dir = synthesizer_root.joinpath("embeds")
    embed_dir.mkdir(exist_ok=True)

    # Gather the input wave filepath and the target output embed filepath
    with metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[2])) for m in metadata]

    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))

