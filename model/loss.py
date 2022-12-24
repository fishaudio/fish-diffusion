import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from utils.tools import ssim
from utils.pitch_tools import cwt2f0_norm
from text import sil_phonemes_ids


class DiffSingerLoss(nn.Module):
    """ DiffSinger Loss """

    def __init__(self, args, preprocess_config, model_config, train_config):
        super(DiffSingerLoss, self).__init__()
        self.model = args.model
        self.loss_config = train_config["loss"]
        self.pitch_config = preprocess_config["preprocessing"]["pitch"]
        self.pitch_type = self.pitch_config["pitch_type"]
        self.use_pitch_embed = model_config["variance_embedding"]["use_pitch_embed"]
        self.use_energy_embed = model_config["variance_embedding"]["use_energy_embed"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.sil_ph_ids = sil_phonemes_ids()

    def forward(self, inputs, predictions):
        (
            texts,
            _,
            _,
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            mel2phs,
        ) = inputs[3:]
        (
            mel_predictions,
            _,
            noise_loss,
            _,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        self.src_masks = ~src_masks
        mel_masks = ~mel_masks
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        self.mel_masks = mel_masks[:, :mel_masks.shape[1]]
        self.mel_masks_fill = ~self.mel_masks
        self.mel2phs = mel2phs

        duration_loss = self.get_duration_loss(log_duration_predictions, duration_targets, texts)

        pitch_loss = energy_loss = torch.zeros(1).to(mel_targets.device)
        if self.use_pitch_embed:
            pitch_loss = self.get_pitch_loss(pitch_predictions, pitch_targets)
        if self.use_energy_embed:
            energy_loss = self.get_energy_loss(energy_predictions, energy_targets)

        total_loss = sum(duration_loss.values()) + sum(pitch_loss.values()) + energy_loss

        if self.model == "aux":
            noise_loss = torch.zeros(1).to(mel_targets.device)
            mel_loss = self.get_mel_loss(mel_predictions, mel_targets)
            total_loss += mel_loss
        elif self.model in ["naive", "shallow"]:
            mel_loss = torch.zeros(1).to(mel_targets.device)
            total_loss += noise_loss
        else:
            raise NotImplementedError

        return (
            total_loss,
            mel_loss,
            noise_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )

    def get_mel_loss(self, mel_predictions, mel_targets):
        mel_predictions = mel_predictions.masked_fill(self.mel_masks_fill.unsqueeze(-1), 0)
        mel_targets = mel_targets.masked_fill(self.mel_masks_fill.unsqueeze(-1), 0)
        mel_loss = self.l1_loss(mel_predictions, mel_targets)
        return mel_loss

    def l1_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        l1_loss = F.l1_loss(decoder_output, target, reduction="none")
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def ssim_loss(self, decoder_output, target, bias=6.0):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        weights = self.weights_nonzero_speech(target)
        decoder_output = decoder_output[:, None] + bias
        target = target[:, None] + bias
        ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
        ssim_loss = (ssim_loss * weights).sum() / weights.sum()
        return ssim_loss

    def weights_nonzero_speech(self, target):
        # target : B x T x mel
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

    def get_duration_loss(self, dur_pred, dur_gt, txt_tokens):
        """
        :param dur_pred: [B, T], float, log scale
        :param txt_tokens: [B, T]
        :return:
        """
        losses = {}
        B, T = txt_tokens.shape
        nonpadding = self.src_masks.float()
        dur_gt = dur_gt.float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p_id in self.sil_ph_ids:
            is_sil = is_sil | (txt_tokens == p_id)
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if self.loss_config["dur_loss"] == "mse":
            losses["pdur"] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction="none")
            losses["pdur"] = (losses["pdur"] * nonpadding).sum() / nonpadding.sum()
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        elif self.loss_config["dur_loss"] == "mog":
            return NotImplementedError
        elif self.loss_config["dur_loss"] == "crf":
            # losses["pdur"] = -self.model.dur_predictor.crf(
            #     dur_pred, dur_gt.long().clamp(min=0, max=31), mask=nonpadding > 0, reduction="mean")
            return NotImplementedError
        losses["pdur"] = losses["pdur"] * self.loss_config["lambda_ph_dur"]

        # use linear scale for sent and word duration
        if self.loss_config["lambda_word_dur"] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
            word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction="none")
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses["wdur"] = wdur_loss * self.loss_config["lambda_word_dur"]
        if self.loss_config["lambda_sent_dur"] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction="mean")
            losses["sdur"] = sdur_loss.mean() * self.loss_config["lambda_sent_dur"]
        return losses

    def get_pitch_loss(self, pitch_predictions, pitch_targets):
        losses = {}
        for _, pitch_target in pitch_targets.items():
            if pitch_target is not None:
                pitch_target.requires_grad = False
        if self.pitch_type == "ph":
            nonpadding = self.src_masks.float()
            pitch_loss_fn = F.l1_loss if self.loss_config["pitch_loss"] == "l1" else F.mse_loss
            losses["f0"] = (pitch_loss_fn(pitch_predictions["pitch_pred"][:, :, 0], pitch_targets["f0"],
                                          reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_f0"]
        else:
            mel2ph = self.mel2phs  # [B, T_s]
            f0 = pitch_targets["f0"]
            uv = pitch_targets["uv"]
            nonpadding = self.mel_masks.float()
            if self.pitch_type == "cwt":
                cwt_spec = pitch_targets[f"cwt_spec"]
                f0_mean = pitch_targets["f0_mean"]
                f0_std = pitch_targets["f0_std"]
                cwt_pred = pitch_predictions["cwt"][:, :, :10]
                f0_mean_pred = pitch_predictions["f0_mean"]
                f0_std_pred = pitch_predictions["f0_std"]
                losses["C"] = self.cwt_loss(cwt_pred, cwt_spec) * self.loss_config["lambda_f0"]
                if self.pitch_config["use_uv"]:
                    assert pitch_predictions["cwt"].shape[-1] == 11
                    uv_pred = pitch_predictions["cwt"][:, :, -1]
                    losses["uv"] = (F.binary_cross_entropy_with_logits(uv_pred, uv, reduction="none") * nonpadding) \
                                    .sum() / nonpadding.sum() * self.loss_config["lambda_uv"]
                losses["f0_mean"] = F.l1_loss(f0_mean_pred, f0_mean) * self.loss_config["lambda_f0"]
                losses["f0_std"] = F.l1_loss(f0_std_pred, f0_std) * self.loss_config["lambda_f0"]
                # if self.loss_config["cwt_add_f0_loss"]:
                #     f0_cwt_ = cwt2f0_norm(cwt_pred, f0_mean_pred, f0_std_pred, mel2ph, self.pitch_config)
                #     self.add_f0_loss(f0_cwt_[:, :, None], f0, uv, losses, nonpadding=nonpadding)
            elif self.pitch_type == "frame":
                self.add_f0_loss(pitch_predictions["pitch_pred"], f0, uv, losses, nonpadding=nonpadding)
        return losses

    def add_f0_loss(self, p_pred, f0, uv, losses, nonpadding):
        assert p_pred[..., 0].shape == f0.shape
        if self.pitch_config["use_uv"]:
            assert p_pred[..., 1].shape == uv.shape
            losses["uv"] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_uv"]
            nonpadding = nonpadding * (uv == 0).float()

        f0_pred = p_pred[:, :, 0]
        if self.loss_config["pitch_loss"] in ["l1", "l2"]:
            pitch_loss_fn = F.l1_loss if self.loss_config["pitch_loss"] == "l1" else F.mse_loss
            losses["f0"] = (pitch_loss_fn(f0_pred, f0, reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_f0"]
        elif self.loss_config["pitch_loss"] == "ssim":
            return NotImplementedError

    def cwt_loss(self, cwt_p, cwt_g):
        if self.loss_config["cwt_loss"] == "l1":
            return F.l1_loss(cwt_p, cwt_g)
        if self.loss_config["cwt_loss"] == "l2":
            return F.mse_loss(cwt_p, cwt_g)
        if self.loss_config["cwt_loss"] == "ssim":
            return self.ssim_loss(cwt_p, cwt_g, 20)

    def get_energy_loss(self, energy_predictions, energy_targets):
        energy_targets.requires_grad = False
        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(self.src_masks)
            energy_targets = energy_targets.masked_select(self.src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(self.mel_masks)
            energy_targets = energy_targets.masked_select(self.mel_masks)
        energy_loss = F.l1_loss(energy_predictions, energy_targets)
        return energy_loss
