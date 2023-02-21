import argparse
import os

parser = argparse.ArgumentParser(description="Fish-SVC batch inferencing tool")
parser.add_argument(
    "--config", type=str, required=True, help="Path of the config you want to use"
)
parser.add_argument(
    "--sampler_interval",
    type=int,
    default=10,
    help="Speedup value for the inference process. Higher values will decrease render quality but increase inference speed.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to the checkpoint file to use for rendering.",
)
parser.add_argument(
    "--input_audio_folder",
    type=str,
    required=True,
    help="Path to the folder containing input audio WAV files.",
)
parser.add_argument(
    "--output_folder",
    type=str,
    required=True,
    help="Path to the folder where rendered audio WAV files will be saved.",
)
parser.add_argument(
    "--pitch_adjust",
    type=int,
    default=0,
    help="Pitch adjustment value for the rendered audio, in semitones. Positive values increase pitch, negative values decrease pitch.",
)
args = parser.parse_args()

config = args.config
sampler_interval = args.sampler_interval
checkpoint = args.checkpoint
input_audio_folder = args.input_audio_folder
output_folder = args.output_folder
pitch_adjust = args.pitch_adjust

for name in os.listdir(input_audio_folder):
    if name.endswith(".wav"):
        input_file = os.path.join(input_audio_folder, name)
        print(input_file)
        render = os.path.join(output_folder, name)
        command = f"python inference.py --config {config} --checkpoint {checkpoint} --sampler_interval {sampler_interval} --pitch_adjust {pitch_adjust} --input {input_audio_folder} --output {output_folder}"
        os.system(command)
