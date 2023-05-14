import argparse
import subprocess as sp
import tempfile

import gradio as gr
import librosa
import yaml
from mmengine import Config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", action="store_true", default=False)
args = parser.parse_args()

# Parse MODELS.yaml
with open(args.config, "r") as f:
    GLOBAL_CONFIG = yaml.safe_load(f)

README = GLOBAL_CONFIG["readme"]
MAX_MIXING_SPEAKERS = GLOBAL_CONFIG["max_mixing_speakers"]
MODELS = GLOBAL_CONFIG["models"]


# Speaker mixing component
def get_speakers_layout(init_speakers=[], default_speaker=None):
    with gr.Blocks():
        # Define a component for speaker mixing
        n_speakers = gr.Slider(
            label="Number of Speakers",
            minimum=1,
            maximum=MAX_MIXING_SPEAKERS,
            value=1,
            step=1,
        )

        speakers = []
        speaker_weights = []

        for i in range(MAX_MIXING_SPEAKERS):
            with gr.Row():
                speaker = gr.Dropdown(
                    choices=init_speakers,
                    label=f"Speaker Name {i}",
                    type="value",
                    value=default_speaker
                    or (init_speakers[0] if len(init_speakers) > 0 else None),
                    visible=False if i > 0 else True,
                )
                weight = gr.Slider(
                    label="Weight",
                    minimum=0,
                    maximum=1,
                    value=1,
                    visible=False if i > 0 else True,
                )

                speakers.append(speaker)
                speaker_weights.append(weight)

    n_speakers.change(
        lambda n: [
            gr.Number.update(visible=i < int(n)) for i in range(MAX_MIXING_SPEAKERS)
        ],
        n_speakers,
        speakers,
    )

    n_speakers.change(
        lambda n: [
            gr.Slider.update(visible=i < int(n)) for i in range(MAX_MIXING_SPEAKERS)
        ],
        n_speakers,
        speaker_weights,
    )

    return speakers, speaker_weights, n_speakers


# Audios Component
def audios_layout():
    input_method = gr.Radio(
        ["Upload", "Microphone"], label="Input Method", value="Upload"
    )
    input_audio = gr.Audio(
        label="Input Audio",
        type="filepath",
        source="upload",
    )

    output_audio = gr.Audio(label="Output Audio")
    input_method.change(
        lambda m: gr.Audio.update(source=m.lower()), input_method, input_audio
    )

    return input_audio, output_audio


# Update speakers and state on model change
def get_speakers_from_model(m):
    config = Config.fromfile(MODELS[m]["config"])
    return list(config.speaker_mapping.keys())


def on_select_model(m):
    all_speakers = get_speakers_from_model(m)

    return [
        gr.Dropdown.update(
            choices=all_speakers,
            value=all_speakers[0] if len(all_speakers) > 0 else None,
        )
    ] * MAX_MIXING_SPEAKERS


def run_model(
    input_audio,
    model,
    pitch_shift,
    n_speakers,
    pitch_extractor,
    silence_threshold,
    max_slice_duration,
    min_silence_duration,
    *args,
):
    if input_audio is None:
        return [
            "Please upload an audio file or record an audio file using the microphone.",
            None,
        ]

    speakers = args[:MAX_MIXING_SPEAKERS]
    speaker_weights = args[MAX_MIXING_SPEAKERS:]
    assert len(speakers) == len(speaker_weights) == MAX_MIXING_SPEAKERS

    speaker_mixing = {}
    for i in range(n_speakers):
        speaker_mixing[speakers[i]] = (
            speaker_mixing.get(speakers[i], 0) + speaker_weights[i]
        )

    speaker_mixing = ",".join([f"{k}:{v}" for k, v in speaker_mixing.items()])

    # Call shell instead of python
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        command = [
            "python3",
            "tools/hifisinger/inference.py",
            "--config",
            MODELS[model]["config"],
            "--checkpoint",
            MODELS[model]["checkpoint"],
            "--input",
            input_audio,
            "--output",
            f.name,
            "--pitch_adjust",
            f"{pitch_shift}",
            "--speaker",
            speaker_mixing,
            "--pitch_extractor",
            pitch_extractor,
            "--silence_threshold",
            f"{silence_threshold}",
            "--max_slice_duration",
            f"{max_slice_duration}",
            "--min_silence_duration",
            f"{min_silence_duration}",
        ]

        log = "Executing: " + " ".join(command) + "\n"

        try:
            x = sp.check_output(command)
            log += x.decode("utf-8")
        except sp.CalledProcessError as e:
            log += "Failed: \n" + e.output.decode("utf-8")

            return log, None

        audio, sr = librosa.load(f.name, sr=None)

    return log, (sr, audio)


# Main app
def main():
    with gr.Blocks(title="HiFiSinger Demo") as app:
        gr.Markdown(README)

        with gr.Row():
            with gr.Column():
                input_audio, output_audio = audios_layout()
                log = gr.Textbox(label="Log")

            with gr.Column():
                model = gr.Dropdown(
                    label="Model",
                    choices=[k["name"] for k in MODELS],
                    value=MODELS[0]["name"],
                    type="index",
                )

                readme = gr.Markdown(MODELS[0]["readme"])
                model.change(
                    lambda m: gr.Textbox.update(MODELS[m]["readme"]), model, readme
                )

                pitch_shift = gr.Slider(
                    label="Pitch Shift",
                    minimum=-12,
                    maximum=12,
                    value=0,
                    step=1,
                )

                with gr.Accordion("Advanced", open=False):
                    pitch_extractor = gr.Dropdown(
                        label="Pitch Extractor",
                        choices=[
                            "ParselMouthPitchExtractor",
                            "DioPitchExtractor",
                            "PyinPitchExtractor",
                            "CrepePitchExtractor",
                        ],
                        value="ParselMouthPitchExtractor",
                    )

                    silence_threshold = gr.Slider(
                        label="Silence Threshold",
                        minimum=40,
                        maximum=100,
                        value=60,
                        step=1,
                    )

                    max_slice_duration = gr.Slider(
                        label="Max Slice Duration",
                        minimum=10,
                        maximum=120,
                        value=120,
                        step=1,
                    )

                    min_silence_duration = gr.Slider(
                        label="Min Silence Duration",
                        minimum=0,
                        maximum=10,
                        value=2,
                    )

                speakers, speaker_weights, n_speakers = get_speakers_layout(
                    init_speakers=get_speakers_from_model(0),
                    default_speaker=MODELS[0].get("default_speaker", None),
                )
                model.change(on_select_model, model, speakers)

                with gr.Row():
                    run = gr.Button(value="⚡️ Run")

                run.click(
                    run_model,
                    [input_audio, model, pitch_shift, n_speakers]
                    + [
                        pitch_extractor,
                        silence_threshold,
                        max_slice_duration,
                        min_silence_duration,
                    ]
                    + speakers
                    + speaker_weights,
                    [log, output_audio],
                )

    return app


demo = main().queue()

if __name__ == "__main__":
    demo.launch(
        server_port=args.port,
        share=args.share,
    )
