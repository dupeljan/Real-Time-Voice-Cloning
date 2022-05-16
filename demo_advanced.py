import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch

from pathlib import Path

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder
from demo_advanced.clusterization import SpeakerMixer
from demo_advanced.clusterization import Strategy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("-c", "--cluster_path", type=Path,
                        default=None,
                        help="Path to a directory with per speaker from training datasets mean embeddings."
                             "To generate this directory one should launch "
                             "`encoder_preprocess` with `--mode=clustering`")
    parser.add_argument("--mix_coef", type=float,
                        default=0.,
                        help="Coefficient in [0.,1.] for linear mixing of original speaker embedding and"
                             "closet to original speaker embedding from training dataset. Useful when original" 
                             "speaker voice is too unique and syntesizer can't perform TTS with satisfying quality." 
                             "Bigger the coefficient - fewer the original speaker features "
                             "in embedding for synthesizer")
    parser.add_argument("--adaptive_mixing", action="store_true", help= \
                    "Adapted strategy depending on cosine score is used in case this parameter is fed.")
    parser.add_argument("--texts_to_gen", type=Path,
                        default="demo_advanced/default_texts.txt",
                        help="Path to txt file with lines of text to generate")
    parser.add_argument("--voices_to_clone", type=Path,
                        #default="demo_advanced/default_texts.txt",
                        help="Path to dir with audiofiles with voices to clone."
                             "Each voice should be put in separate dir with file with meaningful name."
                             "All audio files in dir are used to generate one embedding."
                             "Encoder encode each audio and then all embeddings are averaging.")
    parser.add_argument('-o', "--output_dir", type=Path,
                        default="demo_advanced/output/",
                        help="Path to dir to save generated audios.")
    parser.add_argument("--cpu", action="store_true", help= \
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help= \
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help= \
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
              "%.1fGb total memory.\n" %
              (torch.cuda.device_count(),
               device_id,
               gpu_properties.name,
               gpu_properties.major,
               gpu_properties.minor,
               gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    #ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)


    ## Interactive speech generation
    print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
          "show how you can interface this project easily with your own. See the source code for "
          "an explanation of what is happening.\n")


    # Gather all voices to clone

    speakers = dict()
    for speaker in args.voices_to_clone.iterdir():
        if speaker.is_dir():
            speakers[speaker.name] = list(speaker.glob('*'))

    texts = [text.rstrip('\n') for text in args.texts_to_gen.open('r').readlines()]

    print(f'{len(texts)} texts in total')
    print(f'{len(speakers)} speakers:\nSpeaker : num of audiofiles\n' +
          '\n'.join([f'{k} : {len(v)}' for k, v in speakers.items()]))
    ## Computing the embedding
    # First, we load the wav using the function that the speaker encoder provides. This is
    # important: there is preprocessing that must be applied.

    # The following two methods are equivalent:
    # - Directly load from the filepath:
    aggregate_fn = lambda input: np.mean(input, axis=0)
    embed = None
    args.output_dir.mkdir(exist_ok=True)
    mixing_strategy = Strategy.linear if not args.adaptive_mixing else Strategy.adaptive
    speaker_clusterisation = SpeakerMixer(args.cluster_path, mixing_strategy, args.mix_coef)

    for name, data in speakers.items():
        print(f'Process speaker {name}')
        speaker_dir_output = args.output_dir.joinpath(name)
        speaker_dir_output.mkdir(exist_ok=True)
        if not data:
            continue

        speaker_embedds = []
        for in_fpath in data:
            #preprocessed_wav = encoder.preprocess_wav(in_fpath)
            # - If the wav is already loaded:
            original_wav, sampling_rate = librosa.load(str(in_fpath))
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)

            # Then we derive the embedding. There are many functions and parameters that the
            # speaker encoder interfaces. These are mostly for in-depth research. You will typically
            # only use this function (with its default parameters):
            embed = encoder.embed_utterance(preprocessed_wav)
            speaker_embedds.append(embed)
        embed = aggregate_fn(speaker_embedds)
        embed = speaker_clusterisation.mix_with_most_similar(embed)
        print("Created the embedding")


        # If seed is specified, reset torch seed and force synthesizer reload
        if args.seed is not None:
            torch.manual_seed(args.seed)
            synthesizer = Synthesizer(args.syn_model_fpath)

        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        for idx, text in enumerate(texts):
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = synthesizer.synthesize_spectrograms([text], [embed])
            spec = specs[0]
            # If seed is specified, reset torch seed and reload vocoder
            if args.seed is not None:
                torch.manual_seed(args.seed)
                vocoder.load_model(args.voc_model_fpath)

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = vocoder.infer_waveform(spec)

            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

            # Trim excess silences to compensate for gaps in spectrograms (issue #53)
            generated_wav = encoder.preprocess_wav(generated_wav)

            # Play the audio (non-blocking)
            if not args.no_sound:
                import sounddevice as sd
                try:
                    sd.stop()
                    sd.play(generated_wav, synthesizer.sample_rate)
                except sd.PortAudioError as e:
                    print("\nCaught exception: %s" % repr(e))
                    print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
                except:
                    raise

            # Save it on the disk
            filename = speaker_dir_output / f'{idx}_generated.wav'
            #print(generated_wav.dtype)
            sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
            #num_generated += 1
            #print("\nSaved output as %s\n\n" % filename)

    print('Done!')
