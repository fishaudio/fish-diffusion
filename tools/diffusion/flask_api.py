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

    # 模型推理
    _audio, _model_sr = model.inference(
        input_path=input_wav_file,
        output_path=None,
        speaker=int_speaker_id,
        pitch_adjust=f_pitch_change,
        silence_threshold=silence_threshold,
        max_slice_duration=max_slice_duration,
        extract_vocals=extract_vocals,
        sampler_interval=sampler_interval,
    )

    tar_audio = librosa.resample(_audio, _model_sr, daw_sample)

    # 返回音频
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio, daw_sample, format="wav")
    out_wav_path.seek(0)

    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == "__main__":
    # fish下只需传入下列参数，文件路径以项目根目录为准
    checkpoint_path = (
        "logs/DiffSVC/version_0/checkpoints/epoch=123-step=300000-valid_loss=0.17.ckpt"
    )
    config_path = "configs/svc_cn_hubert_soft_ms.py"
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
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)
