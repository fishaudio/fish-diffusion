"""
This file is used to support the vst plugin.
"""

import io
import logging

import librosa
import soundfile
from flask import Flask, request, send_file
from flask_cors import CORS
from inference import inference
from mmengine import Config

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
    int_speak_Id = int(request_form.get("sSpeakId", 0))
    if enable_spk_id_cover:
        int_speak_Id = spk_id
    print("说话人:" + str(int_speak_Id))
    # DAW所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    # http获得wav文件并转换
    input_wav_file = io.BytesIO(wave_file.read())
    # 模型推理
    _audio, _model_sr = svc_model.infer(input_wav_file, f_pitch_change, int_speak_Id)
    tar_audio = librosa.resample(_audio, _model_sr, daw_sample)
    # 返回音频
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio, daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


class SvcFish:
    def __init__(
        self,
        checkpoint_path,
        config_path,
        sampler_interval=None,
        extract_vocals=True,
        merge_non_vocals=True,
        vocals_loudness_gain=0.0,
        silence_threshold=60,
        max_slice_duration=30.0,
    ):
        self.config = Config.fromfile(config_path)
        self.checkpoint_path = checkpoint_path
        self.sampler_interval = sampler_interval
        self.silence_threshold = silence_threshold
        self.max_slice_duration = max_slice_duration
        self.extract_vocals = extract_vocals
        self.merge_non_vocals = merge_non_vocals
        self.vocals_loudness_gain = vocals_loudness_gain

    def infer(self, input_path, pitch_adjust, speaker_id):
        return inference(
            config=self.config,
            checkpoint=self.checkpoint_path,
            input_path=input_path,
            output_path=None,
            speaker_id=speaker_id,
            pitch_adjust=pitch_adjust,
            silence_threshold=self.silence_threshold,
            max_slice_duration=self.max_slice_duration,
            extract_vocals=self.extract_vocals,
            merge_non_vocals=self.merge_non_vocals,
            vocals_loudness_gain=self.vocals_loudness_gain,
            sampler_interval=self.sampler_interval,
            sampler_progress=True,
            device="cuda",
            gradio_progress=None,
        )


if __name__ == "__main__":
    # fish下只需传入下列参数，文件路径以项目根目录为准
    checkpoint_path = (
        "logs/DiffSVC/version_0/checkpoints/epoch=123-step=300000-valid_loss=0.17.ckpt"
    )
    config_path = "configs/svc_cn_hubert_soft_ms.py"
    # 加速倍率，None即采用配置文件的值
    sampler_interval = None
    # 是否提取人声，是否合成非人声，以及人声响度增益
    extract_vocals = False
    merge_non_vocals = False
    vocals_loudness_gain = 0.0
    # 默认说话人。以及是否优先使用默认说话人覆盖vst传入的参数。
    spk_id = 0
    enable_spk_id_cover = False
    # 最大切片时长
    max_slice_duration = 30.0
    # 静音阈值
    silence_threshold = 60

    svc_model = SvcFish(
        checkpoint_path,
        config_path,
        sampler_interval=sampler_interval,
        extract_vocals=extract_vocals,
        merge_non_vocals=merge_non_vocals,
        vocals_loudness_gain=vocals_loudness_gain,
        silence_threshold=silence_threshold,
        max_slice_duration=max_slice_duration,
    )

    # 此处与vst插件对应，不建议更改
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)
