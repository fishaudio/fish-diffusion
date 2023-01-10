import librosa
import torch
from utils.tools import load_cn_model
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

path = "data/diff-svc-clean/aria/call of silence 干音_0000/0000.wav"
con_model = load_cn_model()

audio, sampling_rate = librosa.load(path)
if len(audio.shape) > 1:
    audio = librosa.to_mono(audio.transpose(1, 0))
if sampling_rate != 16000:
    audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
feats = torch.from_numpy(audio).float()
feats = feats.view(1, -1)
padding_mask = torch.BoolTensor(feats.shape).fill_(False)
inputs = {
    "source": feats,
    "padding_mask": padding_mask,
    "output_layer": 9,  # layer 9
}
print(con_model)
with torch.no_grad():
    logits = con_model.extract_features(**inputs)
    print(logits[0])
    feats = con_model.final_proj(logits[0])
r0 = feats.transpose(1, 2)

model_path = "TencentGameMate/chinese-hubert-base"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = HubertModel.from_pretrained(model_path)

# for pretrain: Wav2Vec2ForPreTraining
# model = Wav2Vec2ForPreTraining.from_pretrained(model_path)

# model = model.half()
model.eval()

# wav, sr = sf.read(wav_path)
input_values = feature_extractor(
    audio, sampling_rate=16000, return_tensors="pt"
).input_values
# input_values = input_values.half()
# input_values = input_values.to(device)

with torch.no_grad():
    outputs = model(input_values, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states

print(last_hidden_state)
