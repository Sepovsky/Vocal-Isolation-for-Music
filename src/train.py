#!/usr/bin/env python3
"""SpeechBrain training entrypoint for music source separation.

This refactored version keeps the project logic but removes debug prints,
clarifies tensor handling, and makes the script easier to publish on GitHub.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class Separation(sb.Brain):
    """SpeechBrain Brain class for source separation."""

    def _stack_targets(self, targets):
        target_tensor = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)
        return target_tensor.squeeze(0).permute(1, 0, 2)

    def _prepare_eval_mix(self, mix):
        mix = mix.squeeze(0)
        return mix.permute(1, 0)

    def compute_forward(self, mix, targets, stage, noise=None):
        del noise
        mix, mix_lens = mix
        mix = mix.to(self.device)
        mix_lens = mix_lens.to(self.device)
        targets = self._stack_targets(targets)

        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb:
                    mix, targets = self.add_speed_perturb(targets, mix_lens)
                    mix = targets.sum(-1)

                if self.hparams.use_wavedrop:
                    mix = self.hparams.drop_chunk(mix, mix_lens)
                    mix = self.hparams.drop_freq(mix)

                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)
        else:
            mix = self._prepare_eval_mix(mix)

        encoded = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(encoded)
        encoded = torch.stack([encoded] * self.hparams.num_spks)
        separated_hidden = encoded * est_mask

        est_source = torch.cat(
            [self.hparams.Decoder(separated_hidden[i]).unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        )

        t_origin = mix.size(1)
        t_est = est_source.size(1)
        if t_origin > t_est:
            est_source = F.pad(est_source, (0, 0, 0, t_origin - t_est))
        else:
            est_source = est_source[:, :t_origin, :]

        return est_source, targets

    def compute_objectives(self, predictions, targets):
        if predictions.ndim > 3:
            predictions = predictions.squeeze(0).permute(1, 0, 2)
        return self.hparams.loss(targets, predictions)

    def fit_batch(self, batch):
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig, batch.s3_sig, batch.s4_sig]

        with self.training_ctx:
            predictions, targets = self.compute_forward(mixture, targets, sb.Stage.TRAIN)
            loss = self.compute_objectives(predictions, targets)

            if self.hparams.threshold_byloss:
                threshold = self.hparams.threshold
                loss = loss[loss > threshold]
                if loss.nelement() > 0:
                    loss = loss.mean()
            else:
                loss = loss.mean()

        if loss.nelement() > 0 and loss < self.hparams.loss_upper_lim:
            self.scaler.scale(loss).backward()
            if self.hparams.clip_grad_norm >= 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.modules.parameters(), self.hparams.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.nonfinite_count += 1
            logger.info(
                "Skipping batch due to invalid or empty loss. Occurrence count: %s",
                self.nonfinite_count,
            )
            loss = torch.tensor(0.0, device=self.device)

        self.optimizer.zero_grad()
        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig, batch.s3_sig, batch.s4_sig]

        with torch.no_grad():
            predictions, targets = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives(predictions, targets)

        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save -= 1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.mean().detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif stage == sb.Stage.VALID:
            if isinstance(self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau):
                current_lr, next_lr = self.hparams.lr_scheduler([self.optimizer], epoch, stage_loss)
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"])
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


def load_hparams(hparams_path: str):
    with open(hparams_path, encoding="utf-8") as fin:
        return load_hyperpyyaml(fin)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/train.py <config.yaml>")

    hparams_path = sys.argv[1]
    if not Path(hparams_path).exists():
        raise FileNotFoundError(f"Config not found: {hparams_path}")

    hparams = load_hparams(hparams_path)
    logger.info("Starting training with config: %s", hparams_path)

    # Dataset preparation hooks should be added here if needed.
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        checkpointer=hparams["checkpointer"],
    )

    separator.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=hparams["train_data"],
        valid_set=hparams["valid_data"],
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )


if __name__ == "__main__":
    main()
