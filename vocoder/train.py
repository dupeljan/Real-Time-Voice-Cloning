import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import vocoder.hparams as hp
from vocoder.display import stream, simple_table
from vocoder.distribution import discretized_mix_logistic_loss
from vocoder.gen_wavernn import gen_testset
from vocoder.models.fatchord_version import WaveRNN
from vocoder.vocoder_dataset import VocoderDataset, collate_vocoder


def train(run_id: str, syn_dir: Path, voc_dir: Path, models_dir: Path, ground_truth: bool, save_every: int,
          backup_every: int, force_restart: bool, tensorboard_freq: int, use_dataparallel: bool):
    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    # Instantiate the model
    print("Initializing the model...")
    model = WaveRNN(
        rnn_dims=hp.voc_rnn_dims,
        fc_dims=hp.voc_fc_dims,
        bits=hp.bits,
        pad=hp.voc_pad,
        upsample_factors=hp.voc_upsample_factors,
        feat_dims=hp.num_mels,
        compute_dims=hp.voc_compute_dims,
        res_out_dims=hp.voc_res_out_dims,
        res_blocks=hp.voc_res_blocks,
        hop_length=hp.hop_length,
        sample_rate=hp.sample_rate,
        mode=hp.voc_mode
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    for p in optimizer.param_groups:
        p["lr"] = hp.voc_lr
    model_mode = model.mode
    loss_func = F.cross_entropy if model_mode == "RAW" else discretized_mix_logistic_loss

    # Load the weights
    model_dir = models_dir / run_id
    model_dir.mkdir(exist_ok=True)
    weights_fpath = model_dir / "vocoder.pt"
    trained_weights_fpath = model_dir / "vocoder_trained.pt"
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of WaveRNN from scratch\n")
        model.save(trained_weights_fpath, optimizer)
    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.load(weights_fpath, optimizer)
        print("WaveRNN weights loaded from step %d" % model.step)

    # Initialize the dataset
    metadata_fpath = syn_dir.joinpath("train.txt") if ground_truth else \
        voc_dir.joinpath("synthesized.txt")
    mel_dir = syn_dir.joinpath("mels") if ground_truth else voc_dir.joinpath("mels_gta")
    wav_dir = syn_dir.joinpath("audio")
    dataset = VocoderDataset(metadata_fpath, mel_dir, wav_dir)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Begin the training
    simple_table([('Batch size', hp.voc_batch_size),
                  ('LR', hp.voc_lr),
                  ('Sequence Len', hp.voc_seq_len)])

    if use_dataparallel:
        model = torch.nn.DataParallel(model)

    writer = SummaryWriter(model_dir)
    for epoch in range(1, 350):
        data_loader = DataLoader(dataset, hp.voc_batch_size, shuffle=True, num_workers=2, collate_fn=collate_vocoder)
        start = time.time()
        running_loss = 0.

        for i, (x, y, m) in enumerate(data_loader, 1):
            if torch.cuda.is_available():
                x, m, y = x.cuda(), m.cuda(), y.cuda()

            # Forward pass
            y_hat = model(x, m)
            if model_mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            elif model_mode == 'MOL':
                y = y.float()
            y = y.unsqueeze(-1)

            # Backward pass
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            speed = i / (time.time() - start)
            avg_loss = running_loss / i

            if use_dataparallel:
                step = model.module.get_step()
            else:
                step = model.get_step()

            k = step // 1000

            if backup_every != 0 and step % backup_every == 0 :
                if use_dataparallel:
                    model.module.checkpoint(model_dir, optimizer)
                else:
                    model.checkpoint(model_dir, optimizer)

            if save_every != 0 and step % save_every == 0 :
                if use_dataparallel:
                    model.module.save(trained_weights_fpath, optimizer)
                else:
                    model.save(trained_weights_fpath, optimizer)

            msg = f"| Epoch: {epoch} ({i}/{len(data_loader)}) | " \
                f"Loss: {avg_loss:.4f} | {speed:.1f} " \
                f"steps/s | Step: {k}k | "
            stream(msg)

            if step % tensorboard_freq == 0:
                writer.add_scalar('Train/Epoch', epoch, step)
                writer.add_scalar('Train/lr', hp.voc_lr, step)
                writer.add_scalar('Train/Loss', avg_loss, step)
                writer.add_scalar('Steps/s', speed, step)

        gen_testset(model, test_loader, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                    hp.voc_target, hp.voc_overlap, model_dir, use_dataparallel)
        print("")
