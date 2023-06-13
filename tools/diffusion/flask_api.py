"""
This file is used to support the vst plugin.
"""

import io
import logging

import librosa
import soundfile
import torch
from flask import Flask, request, send_file
from flask_cors import CORS
from mmengine import Config

from tools.diffusion.inference import SVCInference

app = Flask(__name__)

CORS(app)

logging.getLogger("numba").setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # 变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # 获取spkid
    int_speaker_id = int(request_form.get("sSpeakId", 0))
    if enable_spk_id_cover:
        int_speaker_id = spk_id

    print(f"Speaker: {int_speaker_id}, pitch: {f_pitch_change}")

    # DAW所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))

    # http获得wav文件并转换
    input_wav_file = io.BytesIO(wave_file.read())
    audio, sr = librosa.load(input_wav_file, sr=model.config.sampling_rate, mono=True)
    audio = torch.from_numpy(audio).unsqueeze(0).to(device)

    # 模型推理
    _audio = model.forward(
        audio,
        sr,
        pitch_adjust=f_pitch_change,
        speakers=torch.tensor([int_speaker_id]).to(device),
    )

    tar_audio = librosa.resample(_audio, orig_sr=sr, target_sr=daw_sample)

    # 返回音频
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio, daw_sample, format="wav")
    out_wav_path.seek(0)

    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == "__main__":
    # fish下只需传入下列参数，文件路径以项目根目录为准
    checkpoint_path = "logs/DiffSVC/8f6md039/checkpoints"
    config_path = "configs/exp_cn_hubert_leak.py"
    # 加速倍率，None即采用配置文件的值
    sampler_interval = None
    # 是否提取人声
    extract_vocals = False
    # 默认说话人。以及是否优先使用默认说话人覆盖vst传入的参数。
    spk_id = 0
    enable_spk_id_cover = False
    # 最大切片时长
    max_slice_duration = 30.0
    # 静音阈值
    silence_threshold = 60

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config.fromfile(config_path)
    model = SVCInference(config, checkpoint_path)
    model = model.to(device)

    # 此处与vst插件对应，不建议更改
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=True)
