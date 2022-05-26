import os

from itertools import islice
from enum import Enum
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import librosa
from tqdm import tqdm

from encoder import audio
from encoder.config import librispeech_datasets, anglophone_nationalites
from encoder.params_data import *
from encoder import inference as encoder_inference


_AUDIO_EXTENSIONS = ("wav", "flac", "m4a", "mp3")


class Dataset(Enum):
    book = "book"
    mozilla = "mozilla"
    Ruslan = 'Ruslan'
    russian_single = 'russian_single'
    m_ailabs = 'ru_RU'
    librispeech = "librispeech"
    vox_celeb1 = 'VoxCeleb1'
    vox_celeb2 = 'VoxCeleb2'


class DatasetLog:
    """
    Registers metadata about the dataset in a text file.
    """
    def __init__(self, root, name):
        self.text_file = open(Path(root, "Log_%s.txt" % name.replace("/", "_")), "w")
        self.sample_data = dict()

        start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Creating dataset %s on %s" % (name, start_time))
        self.write_line("-----")
        self._log_params()

    def _log_params(self):
        from encoder import params_data
        self.write_line("Parameter values:")
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            self.write_line("\t%s: %s" % (param_name, value))
        self.write_line("-----")

    def write_line(self, line):
        self.text_file.write("%s\n" % line)

    def add_sample(self, **kwargs):
        for param_name, value in kwargs.items():
            if not param_name in self.sample_data:
                self.sample_data[param_name] = []
            self.sample_data[param_name].append(value)

    def finalize(self):
        self.write_line("Statistics:")
        for param_name, values in self.sample_data.items():
            self.write_line("\t%s:" % param_name)
            self.write_line("\t\tmin %.3f, max %.3f" % (np.min(values), np.max(values)))
            self.write_line("\t\tmean %.3f, median %.3f" % (np.mean(values), np.median(values)))
        self.write_line("-----")
        end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Finished on %s" % end_time)
        self.text_file.close()


class DatasetPreprocessorMode(Enum):
    to_mels = 'to_mels'
    clustering = 'clustering'


class DatasetsPreprocessor:
    def __init__(self, datasets_root: Path, datasets: str, out_dir: Path,
                 n_processes: int, skip_existing: bool = False,
                 mode: str = DatasetPreprocessorMode.to_mels.value,
                 encoder_model_fpath: Path = None, speaker_limit_for_cluster: int = 5):
        self.dataset_root = datasets_root
        self.datasets = [Dataset[name] for name in datasets.split(',')]
        self.out_dir = out_dir
        self.n_processes = n_processes
        self.skip_existing = skip_existing
        # Param is valid only in clustering mode
        # means count of wavs for one speaker to embedd
        self.wavs_per_speaker = speaker_limit_for_cluster
        self.mode = DatasetPreprocessorMode(mode)
        if self.mode == DatasetPreprocessorMode.clustering:
            if not encoder_model_fpath:
                raise RuntimeError('To preprocess wavs to embeddings encoder checkpoint is needed')
            encoder_inference.load_model(encoder_model_fpath)

    def preprocess(self):
        for dataset_name in self.datasets:
            self._preprocess_dataset(dataset_name)

    def _preprocess_dataset(self, dataset_name):
        dataset_processor_map = {
            Dataset.book: BookPreprocessor,
            Dataset.mozilla: BookPreprocessor,
            Dataset.m_ailabs: AILabsPreprocessor,
            Dataset.russian_single: SingleSpeakerPreprocessor,
            Dataset.Ruslan: SingleSpeakerPreprocessor,
            Dataset.librispeech: LibriSpeechPreprocessor, # Not tested
            Dataset.vox_celeb1: VoxCeleb1Preprocessor, # Not tested
            Dataset.vox_celeb2: VoxCeleb2Preprocessor, # Not tested
        }
        preprocessor = dataset_processor_map[dataset_name](dataset_name.value, self.dataset_root,
                                                           self.out_dir, self.n_processes, mode=self.mode,
                                                           wavs_per_speaker=self.wavs_per_speaker)

        preprocessor.preprocess()


class DatasetPreprocessor(ABC):
    def __init__(self, dataset_name: str, datasets_root: Path, out_dir: Path, n_processes: int, skip_existing=False,
                 mode: DatasetPreprocessorMode = DatasetPreprocessorMode.to_mels, wavs_per_speaker: int = 10):
        self.dataset_name = dataset_name
        self.datasets_root = datasets_root
        self.out_dir = out_dir
        self.skip_existing = skip_existing
        self.n_processes = n_processes
        self.wavs_per_speaker = wavs_per_speaker
        self.mode = mode
        self.mode_to_fn_map = {
            DatasetPreprocessorMode.to_mels: self._preprocess_speaker_mels,
            DatasetPreprocessorMode.clustering: self._preprocess_speaker_embedd,
        }

    def get_logger(self):
        if self.mode == DatasetPreprocessorMode.clustering:
            return None
        return DatasetLog(self.out_dir, self.dataset_name)

    def preprocess(self):
        dataset_root = self.datasets_root.joinpath(self.dataset_name)
        if not dataset_root.exists():
            print("Couldn\'t find %s, skipping this dataset." % dataset_root)
        logger = self.get_logger()
        speaker_dirs = self._get_speakers_dirs(dataset_root)
        self._preprocess_speaker_dirs(speaker_dirs, dataset_root, logger)

    @abstractmethod
    def _get_speakers_dirs(self, dataset_root: Path):
        pass

    @staticmethod
    def _get_speaker_name(dataset_root, speaker_dir) -> str:
        return "_".join(speaker_dir.relative_to(dataset_root).parts)

    def _preprocess_speaker_mels(self, dataset_root: Path, speaker_dir: Path):
        # Give a name to the speaker that includes its dataset
        speaker_name = self._get_speaker_name(dataset_root, speaker_dir)
        # Create an output directory with that name, as well as a txt file containing a
        # reference to each source file.
        speaker_out_dir = self.out_dir.joinpath(speaker_name)
        speaker_out_dir.mkdir(exist_ok=True)
        sources_fpath = speaker_out_dir.joinpath("_sources.txt")

        # There's a possibility that the preprocessing was interrupted earlier, check if
        # there already is a sources file.
        if sources_fpath.exists():
            try:
                with sources_fpath.open("r") as sources_file:
                    existing_fnames = {line.split(",")[0] for line in sources_file}
            except:
                existing_fnames = {}
        else:
            existing_fnames = {}

        # Gather all audio files for that speaker recursively
        sources_file = sources_fpath.open("a" if self.skip_existing else "w")
        audio_durs = []
        for extension in _AUDIO_EXTENSIONS:
            for in_fpath in speaker_dir.glob("**/*.%s" % extension):
                # Check if the target output file already exists
                out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
                out_fname = out_fname.replace(".%s" % extension, ".npy")
                if self.skip_existing and out_fname in existing_fnames:
                    continue

                # Load and preprocess the waveform
                wav = audio.preprocess_wav(in_fpath)
                if len(wav) == 0:
                    continue

                # Create the mel spectrogram, discard those that are too short
                frames = audio.wav_to_mel_spectrogram(wav)
                if len(frames) < partials_n_frames:
                    continue

                out_fpath = speaker_out_dir.joinpath(out_fname)
                np.save(out_fpath, frames)
                sources_file.write("%s,%s\n" % (out_fname, in_fpath))
                audio_durs.append(len(wav) / sampling_rate)

        sources_file.close()
        return audio_durs

    def _preprocess_speaker_embedd(self, dataset_root: Path, speaker_dir: Path):
        # Give a name to the speaker that includes its dataset
        speaker_name = self._get_speaker_name(dataset_root, speaker_dir)
        embeds = []
        out_fname = f'{speaker_name}.npy'
        for extension in _AUDIO_EXTENSIONS:
            for in_fpath in islice(speaker_dir.glob("**/*.%s" % extension), self.wavs_per_speaker):
                # Check if the target output file already exists
                original_wav, sampling_rate = librosa.load(str(in_fpath))
                if not len(original_wav):
                    continue
                preprocessed_wav = encoder_inference.preprocess_wav(original_wav, sampling_rate)

                # Then we derive the embedding. There are many functions and parameters that the
                # speaker encoder interfaces. These are mostly for in-depth research. You will typically
                # only use this function (with its default parameters):
                embed = encoder_inference.embed_utterance(preprocessed_wav)
                embeds.append(embed)
            embed = np.mean(embeds, axis=0)
            out_fpath = self.out_dir.joinpath(out_fname)
            np.save(out_fpath, embed)

    def _preprocess_speaker_dirs(self, speaker_dirs, dataset_root, logger):
        print("%s: Preprocessing data for %d speakers." % (self.dataset_name, len(speaker_dirs)))
        # Process the utterances for each speaker
        work_fn = partial(self.mode_to_fn_map[self.mode], dataset_root)
        if self.mode == DatasetPreprocessorMode.clustering:
            tasks = map(work_fn, speaker_dirs)
            for _ in tqdm(tasks, self.dataset_name, len(speaker_dirs), unit="speakers"):
                pass
        else:
            with Pool(self.n_processes) as pool:
                tasks = pool.imap(work_fn, speaker_dirs)
                for sample_durs in tqdm(tasks, self.dataset_name, len(speaker_dirs), unit="speakers"):
                    for sample_dur in sample_durs:
                        logger.add_sample(duration=sample_dur)

            logger.finalize()
        print("Done preprocessing %s.\n" % self.dataset_name)


class BookPreprocessor(DatasetPreprocessor):
    def _get_speakers_dirs(self, dataset_root):
        return list(dataset_root.glob("*"))


class SingleSpeakerPreprocessor(DatasetPreprocessor):
    def _get_speakers_dirs(self, dataset_root):
        return [dataset_root]

    @staticmethod
    def _get_speaker_name(dataset_root, speaker_dir) -> str:
        return dataset_root.name


class AILabsPreprocessor(DatasetPreprocessor):
    def _get_speakers_dirs(self, dataset_root):
        retval = []
        for gender in (dataset_root / 'by_book').glob('*'):
            if os.path.isdir(gender):
                retval.extend(gender.glob('*'))
        return retval


class LibriSpeechPreprocessor(DatasetPreprocessor):
    def preprocess(self):
        for dataset_name in librispeech_datasets["train"]["other"]:
            self.dataset_name = dataset_name
            super().preprocess()

    def _get_speakers_dirs(self, dataset_root):
        return list(dataset_root.glob("*"))


class VoxCeleb1Preprocessor(DatasetPreprocessor):
    def _get_speakers_dirs(self, dataset_root):
        # Get the contents of the meta file
        with dataset_root.joinpath("vox1_meta.csv").open("r") as metafile:
            metadata = [line.split("\t") for line in metafile][1:]

        # Select the ID and the nationality, filter out non-anglophone speakers
        nationalities = {line[0]: line[3] for line in metadata}
        keep_speaker_ids = [speaker_id for speaker_id, nationality in nationalities.items() if
                            nationality.lower() in anglophone_nationalites]
        print("VoxCeleb1: using samples from %d (presumed anglophone) speakers out of %d." %
              (len(keep_speaker_ids), len(nationalities)))

        # Get the speaker directories for anglophone speakers only
        speaker_dirs = dataset_root.joinpath("wav").glob("*")
        speaker_dirs = [speaker_dir for speaker_dir in speaker_dirs if
                        speaker_dir.name in keep_speaker_ids]
        print("VoxCeleb1: found %d anglophone speakers on the disk, %d missing (this is normal)." %
              (len(speaker_dirs), len(keep_speaker_ids) - len(speaker_dirs)))


class VoxCeleb2Preprocessor(DatasetPreprocessor):
    def _get_speakers_dirs(self, dataset_root):
        return list(dataset_root.joinpath("dev", "aac").glob("*"))
