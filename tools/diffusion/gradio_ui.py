from functools import partial
from typing import Union

import gradio as gr


def run_inference(
    inference_fn,
    input_path: str,
    speaker: Union[int, str, float],
    pitch_adjust: int,
    sampler_interval: int,
    extract_vocals: bool,
    progress=gr.Progress(),
):
    if isinstance(speaker, float):
        speaker = int(speaker)

    audio, sr = inference_fn(
        input_path=input_path,
        output_path=None,
        speaker=speaker,
        pitch_adjust=pitch_adjust,
        sampler_interval=sampler_interval,
        extract_vocals=extract_vocals,
        gradio_progress=progress,
    )

    return sr, audio


def launch_gradio(
    config,
    inference_fn,
    speaker,
    pitch_adjust: int,
    sampler_interval: int,
    extract_vocals: bool,
    share: bool = False,
):
    with gr.Blocks(title="Fish Diffusion") as app:
        gr.Markdown("# Fish Diffusion SVC Inference")

        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(
                    label="Input Audio",
                    type="filepath",
                )
                output_audio = gr.Audio(label="Output Audio")

            with gr.Column():
                if hasattr(config, "speaker_mapping"):
                    speaker_mapping = config.speaker_mapping
                    speaker = gr.Dropdown(
                        label="Speaker Name (Used for Multi-Speaker Models)",
                        choices=list(speaker_mapping.keys()),
                        value=speaker
                        if speaker in speaker_mapping
                        else list(speaker_mapping.keys())[0],
                    )
                else:
                    speaker = gr.Number(
                        label="Speaker ID (Used for Multi-Speaker Models)",
                        value=int(speaker),
                    )

                pitch_adjust = gr.Number(
                    label="Pitch Adjust (Semitones)", value=pitch_adjust
                )
                sampler_interval = gr.Slider(
                    label="Sampler Interval (⬆️ Faster Generation, ⬇️ Better Quality)",
                    value=sampler_interval or 10,
                    minimum=1,
                    maximum=100,
                )
                extract_vocals = gr.Checkbox(
                    label="Extract Vocals (For low quality audio)",
                    value=extract_vocals,
                )
                run_btn = gr.Button(label="Run")

            run_btn.click(
                partial(run_inference, inference_fn),
                [
                    input_audio,
                    speaker,
                    pitch_adjust,
                    sampler_interval,
                    extract_vocals,
                ],
                output_audio,
            )

    app.queue(concurrency_count=2).launch(share=share)
