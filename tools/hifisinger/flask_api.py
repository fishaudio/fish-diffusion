"""
This file is used to support the vst plugin.
"""

import io
import logging

import librosa
import numpy as np
import soundfile as sf
import torch
from fish_audio_preprocess.utils import separate_audio
from flask import Flask, request, send_file
from flask_cors import CORS
from loguru import logger
from mmengine import Config

from fish_diffusion.utils.audio import separate_vocals
from fish_diffusion.utils.tensor import repeat_expand
from tools.hifisinger.inference import HiFiSingerSVCInference

app = Flask(__name__)

CORS(app)

logging.getLogger("numba").setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # 获取fSafePrefixPadLength
    f_safe_prefix_pad_length = float(request_form.get("fSafePrefixPadLength", 0))
    logger.info("接收到的f_safe_prefix_pad_length是:" + str(f_safe_prefix_pad_length))
    # 变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # 获取spk_id
    int_speak_id = int(request_form.get("sSpeakId", 0))
    logger.info("接收到的说话人是:" + str(int_speak_id))
    # DAW所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    # http获得wav文件并转换
    input_wav_read = io.BytesIO(wave_file.read())
    # 模型推理
    _audio, _model_sr = svc_model.infer(
        input_wav_read, f_pitch_change, int_speak_id, f_safe_prefix_pad_length
    )
    tar_audio = librosa.resample(_audio, _model_sr, daw_sample)
    # 返回音频
    out_wav_path = io.BytesIO()
    sf.write(out_wav_path, tar_audio.T, daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


class HiFiSingerSVC:
    def __init__(self, config_flask):
        self.checkpoint_path = config_flask["checkpoint_path"]
        self.config = Config.fromfile(config_flask["config_path"])
        self.extract_vocals = config_flask["extract_vocals"]
        self.spk_id = config_flask["spk_id"]
        self.enable_spk_id_cover = config_flask["enable_spk_id_cover"]
        self.hifisinger_model = HiFiSingerSVCInference(
            self.config, self.checkpoint_path
        )
        self.device = torch.device(config_flask["device"])
        self.hifisinger_model = self.hifisinger_model.to(self.device)
        self.spk_tensor = self.hifisinger_model._parse_speaker(self.spk_id)
        self.pad_f0_model = config_flask["pad_f0_model"]
        self.pad_f0 = 0.0
        self.pad_f0_pad = config_flask["pad_f0_pad"]
        self.separate_model = None

    def infer(self, wav_path, pitch_change, speak_id, safe_prefix_pad):
        if safe_prefix_pad > self.pad_f0_pad:
            safe_prefix_pad = safe_prefix_pad - self.pad_f0_pad
        else:
            safe_prefix_pad = 0
        if self.enable_spk_id_cover:
            speak_tensor = self.spk_tensor.to(self.device)
            logger.info(f"使用了说话人覆盖，实际说话人id是{self.spk_id}")
        else:
            speak_tensor = self.hifisinger_model._parse_speaker(speak_id).to(
                self.device
            )
        # 读音频
        audio, sr = librosa.load(wav_path, sr=None, mono=True)
        logger.info(f"Loaded {wav_path} with sr={sr}")
        if self.extract_vocals:  # 提取人声
            logger.info("Extracting vocals...")
            if self.separate_model is None:
                self.separate_model = separate_audio.init_model(
                    "htdemucs", device=self.device
                )
            audio, _ = separate_vocals(audio, sr, self.device, self.separate_model)
        # audio = loudness_norm.loudness_norm(audio, sr)  # 响度归一化，实时中效果存疑

        generated_audio = np.zeros_like(audio)
        audio_torch = audio = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        mel_len = audio_torch.shape[-1] // 512

        # 跳过pad部分并提起音高
        if safe_prefix_pad != 0:
            unpadded = audio[:, int(safe_prefix_pad * sr) :]
            unpadded_mel_len = unpadded.shape[-1] // 512
            pitches = (
                self.hifisinger_model.pitch_extractor(
                    unpadded, sr, pad_to=unpadded_mel_len
                )
                .float()
                .cpu()
            )
            if self.pad_f0_model == "first":
                logger.info("f0_pad=first")
                self.pad_f0 = pitches[0]
            _pitches_pad = torch.full(
                (mel_len - unpadded_mel_len,), self.pad_f0
            )  # self.last_f0)
            pitches = torch.cat([_pitches_pad, pitches]).to(self.device)
            if self.pad_f0_model == "last":
                logger.info("f0_pad=last")
                self.pad_f0 = pitches[-1]
        else:
            pitches = self.hifisinger_model.pitch_extractor(
                audio, sr, pad_to=mel_len
            ).float()
        if (pitches == 0).all():
            return [np.zeros((audio.shape[-1],)), sr]
        pitches *= 2 ** (pitch_change / 12)
        # 提取特征
        text_features = self.hifisinger_model.text_features_extractor(audio, sr)[0]
        text_features = repeat_expand(text_features, mel_len).T
        # 不移调增强来保证合成效果
        pitch_shift = None
        if self.config.model.get("pitch_shift_encoder"):
            pitch_shift = torch.zeros((1, 1), device=self.device)
        energy = None
        # 处理energy
        if self.config.model.get("energy_encoder"):
            energy = self.hifisinger_model.energy_extractor(audio, sr, pad_to=mel_len)
            energy = energy[None, :, None]  # (1, mel_len, 1)
        # 推理
        contents_lens = torch.tensor([mel_len]).to(self.device)
        out_wav = (
            self.hifisinger_model.model.generator(
                speakers=speak_tensor.to(self.device),
                contents=text_features[None].to(self.device),
                contents_lens=contents_lens.to(self.device),
                contents_max_len=max(contents_lens).to(self.device),
                pitches=pitches[None, :, None].to(self.device),
                pitch_shift=pitch_shift.to(self.device),
                energy=energy,
            )
            .cpu()
            .detach()
            .numpy()[0]
        )
        max_wav_len = generated_audio.shape[-1] - 0
        generated_audio[0 : 0 + out_wav.shape[-1]] = out_wav[:max_wav_len]
        return generated_audio, sr


if __name__ == "__main__":
    # 对接的是串串香火锅大佬https://github.com/zhaohui8969/VST_NetProcess-。建议使用最新版本。
    config_flask = {
        # config和模型路径
        "checkpoint_path": "logs/HiFiSVC/version_1/checkpoints/epoch=597-step=43000-valid_loss=0.98.ckpt",
        "config_path": "configs/hifisinger.py",
        "extract_vocals": False,  # 提取人声
        "device": "cuda",  # cpu or cuda
        "pad_f0_model": "first",  # 'first','last',指定填充使用的值，输入别的参数则填0
        "pad_f0_pad": 0.05,  # 缩减pad用作平滑，建议0.05
        "spk_id": 0,  # 默认说话人。
        "enable_spk_id_cover": True,  # 是否优先使用默认说话人覆盖vst传入的参数。可以是混合说话人。
    }

    svc_model = HiFiSingerSVC(config_flask)

    # 此处与vst插件对应，端口必须接上。
    app.run(port=6844, host="0.0.0.0", debug=False, threaded=False)
