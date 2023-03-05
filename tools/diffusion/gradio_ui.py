import json
import gradio as gr


def run_inference(
    model: str,
    input_path: str,
    speaker: Union[int, str],
    pitch_adjust: int,
    sampler_interval: int,
    extract_vocals: bool,
    device: str,
    progress=gr.Progress(),
    speaker_mapping: dict = None,
):
    if speaker_mapping is not None and isinstance(speaker, str):
        speaker = speaker_mapping[speaker]

    audio, sr = inference(
        model,
        input_path=input_path,
        output_path=None,
        speaker_id=speaker,
        pitch_adjust=pitch_adjust,
        sampler_interval=round(sampler_interval),
        extract_vocals=extract_vocals,
        merge_non_vocals=False,
        device=device,
        gradio_progress=progress,
    )

    return sr, audio

def launch_gradio(args):
    with gr.Blocks(title="Fish Diffusion") as app:
        gr.Markdown("# Fish Diffusion SVC Inference")

        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(
                    label="Input Audio",
                    type="filepath",
                    value=args.input,
                )
                output_audio = gr.Audio(label="Output Audio")

            with gr.Column():
                if args.speaker_mapping is not None:
                    speaker_mapping = json.load(open(args.speaker_mapping))

                    speaker = gr.Dropdown(
                        label="Speaker Name (Used for Multi-Speaker Models)",
                        choices=list(speaker_mapping.keys()),
                        value=list(speaker_mapping.keys())[0],
                    )
                else:
                    speaker_mapping = None
                    speaker = gr.Number(
                        label="Speaker ID (Used for Multi-Speaker Models)",
                        value=args.speaker_id,
                    )

                pitch_adjust = gr.Number(
                    label="Pitch Adjust (Semitones)", value=args.pitch_adjust
                )
                sampler_interval = gr.Slider(
                    label="Sampler Interval (⬆️ Faster Generation, ⬇️ Better Quality)",
                    value=args.sampler_interval or 10,
                    minimum=1,
                    maximum=100,
                )
                extract_vocals = gr.Checkbox(
                    label="Extract Vocals (For low quality audio)",
                    value=args.extract_vocals,
                )
                device = gr.Radio(
                    label="Device", choices=["cuda", "cpu"], value=args.device or "cuda"
                )

                run_btn = gr.Button(label="Run")

            run_btn.click(
                partial(
                    run_inference,
                    args.config,
                    args.checkpoint,
                    speaker_mapping=speaker_mapping,
                ),
                [
                    input_audio,
                    speaker,
                    pitch_adjust,
                    sampler_interval,
                    extract_vocals,
                    device,
                ],
                output_audio,
            )

    app.queue(concurrency_count=2).launch(share=args.gradio_share)
