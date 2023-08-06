from transformers import Wav2Vec2ForPreTraining, Wav2Vec2FeatureExtractor
import torch
import librosa
from pathlib import Path
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "facebook/mms-1b"
source_path = Path("/home/fish/fish-diffusion/dataset/train/aria")
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForPreTraining.from_pretrained(model_id).to(device)

for path in tqdm(list(source_path.rglob("*.wav"))[:10]):
    sample, _ = librosa.load(str(path), sr=16_000)
    inputs = processor(sample, sampling_rate=16_000, return_tensors="pt").to(device)
    print(inputs.attention_mask.shape)

    with torch.no_grad():
        outputs = model(**inputs)

    # print(outputs.projected_quantized_states.shape)
    # cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)
    # print(cosine_sim.mean())
