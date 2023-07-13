import gradio as gr
import os
import subprocess
from regex import P
import yaml
from pathlib import Path

CN_CONFIG = "locales/zh_CN.yaml"
EN_CONFIG = "locales/en_US.yaml"
PREPROCESS_SCRIPT_PTH = "../preprocessing/gen_config.py"
TRAIN_SCRIPT_PTH = "../train.py"
INFERENCE_SCRIPT_PTH = "../inference.py"
FILE_PATH = os.path.abspath(__file__)
Log_Path = Path(FILE_PATH).parent / "logs"


class WebUI:
    def __init__(self, lan_config) -> None:
        self.train_cfg_pth = ""
        # can be "en" or "zh"
        lan_config = Path(FILE_PATH).parent / lan_config
        self.language_conf = yaml.load(lan_config.read_text(), Loader=yaml.FullLoader)
        self.ui_conf = yaml.load(
            (Path(FILE_PATH).parent / "webui.yaml").read_text(), Loader=yaml.FullLoader
        )
        self.user_config_pth = ""

        self.main_ui()

    def lauch(self, *args, **kwargs):
        self.ui.queue(concurrency_count=10).launch(*args, **kwargs)

    def show_log(self, log: str):
        log_file_path = Log_Path / log
        if not log_file_path.exists():
            return "No log file found"
        with open(log_file_path, "r") as f:
            lines = f.readlines()[::-1]
        return "\n".join(lines)

    def run_command(self, command, err_log=f"webui.err.log"):
        command = [item for item in command if item != ""]
        with open(Log_Path / err_log, "w") as f:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=f)

        result = self.show_log(err_log)
        # Check for errors
        if process.returncode != 0:
            return f"Error: {result} "

        # Return the output of the command, using stderr since loguru logs to stderr by default
        return f"Success: {result} "

    def generate_config(self, output, dir_name, model, dataset, scheduler, trainer):
        # Construct the command
        if output == "" or dir_name == "":
            return "Error: output and dir_name can not be empty"
        command = [
            "python",
            "tools/preprocessing/generate_config.py",
            "--output",
            output,
            "--dir-name",
            dir_name,
            "--model",
            model,
            "--dataset",
            dataset,
            "--scheduler",
            scheduler,
            "--trainer",
            trainer,
            "-ms",
        ]
        self.user_config_pth = dir_name + "/" + output + ".yaml"

        # Execute the command
        return self.run_command(command)

    # todo: check why the config path roll back to the default value
    def start_training(
        self,
        config_path,
        entity,
        tensorboard,
        resume,
        resume_id,
        checkpoint,
        wandb_run_name,
        pretrained,
        train_log,
    ):
        # Construct the command
        command = [
            env_prefix,
            "python",
            "tools/train.py",
            f"--config={config_path}",
            "--p" if pretrained else "",
            "--resume" if resume else "",
            f"--resume-id {resume_id}" if resume and resume_id != "" else "",
            f"--checkpoint {checkpoint}" if checkpoint != "" else "",
            f"-t" if tensorboard else "",
            f"-e {entity}" if entity != "" else "",
            f"--name {wandb_run_name}" if wandb_run_name != "" else "",
        ]

        # Execute the command
        return self.run_command(command, train_log)

    def start_preprocessing(self, config_path, clean, num_worker, preprocess_log):
        # python tools/preprocessing/extract_features.py diffsvc/svc_hubert_soft.yaml --num-workers=8 --clean
        command = [
            "python",
            "tools/preprocessing/extract_features.py",
            config_path,
            f"--num-workers={num_worker}",
            "--clean" if clean else "",
        ]

        return self.run_command(command, preprocess_log)

    def main_ui(self):
        with gr.Blocks() as ui:
            with gr.Tab(self.language_conf["main_ui"]["Intro_tab"]["title"]):
                with gr.Accordion(
                    self.language_conf["main_ui"]["Intro_tab"]["terms_of_use"],
                    open=True,
                ):
                    gr.Markdown(
                        self.language_conf["main_ui"]["Intro_tab"][
                            "terms_of_use_content"
                        ]
                    )
                gr.Markdown(self.language_conf["main_ui"]["Intro_tab"]["content"])

            with gr.Tab(self.language_conf["main_ui"]["Generate_Config_tab"]["title"]):
                # Assuming the language_conf dict has the same structure as the previous example
                gr.Markdown(
                    self.language_conf["main_ui"]["Generate_Config_tab"]["content"]
                )
                generate_config_tab = self.ui_conf["generate_config"]
                model_options = generate_config_tab["model_options"]
                optimizer_options = generate_config_tab["optimizer_scheduler_options"]
                trainer_options = generate_config_tab["trainer_options"]
                dataset_options = generate_config_tab["dataset_options"]

                # Create textboxes for entering output name and directory name
                output_textbox = gr.inputs.Textbox(
                    label="Output Name: name of config file"
                )
                dir_name_textbox = gr.inputs.Textbox(
                    label="Directory Name: the run directory for training and checkpoints"
                )

                # Create dropdowns for selecting model, optimizer and trainer
                model_dropdown = gr.inputs.Dropdown(
                    choices=model_options, label="Model", default="diff_svc_v2"
                )
                dataset_dropdown = gr.inputs.Dropdown(
                    choices=dataset_options, label="Dataset", default="naive_svc"
                )
                optimizer_dropdown = gr.inputs.Dropdown(
                    choices=optimizer_options,
                    label="Optimizer and Scheduler",
                    default="warmup_cosine",
                )
                trainer_dropdown = gr.inputs.Dropdown(
                    choices=trainer_options, label="Trainer", default="base"
                )

                # Create a button for generating config
                generate_button = gr.Button(
                    self.language_conf["main_ui"]["Generate_Config_tab"]["genrate_btn"]
                )

                config_output = gr.Textbox()

                generate_button.click(
                    fn=self.generate_config,
                    inputs=[
                        output_textbox,
                        dir_name_textbox,
                        model_dropdown,
                        dataset_dropdown,
                        optimizer_dropdown,
                        trainer_dropdown,
                    ],
                    outputs=config_output,
                )

            # with gr.Tab("Train"):
            with gr.Tab(self.language_conf["main_ui"]["Train_tab"]["title"]):
                # Map the Click options to Gradio components
                config_path = gr.inputs.Textbox(
                    label="Config Path",
                    default="./configs/diffsvc/svc_hubert_soft.yaml",
                )
                config_select_path = gr.inputs.File(label="Config Path")
                with gr.Accordion(
                    self.language_conf["main_ui"]["Train_tab"]["preprocessing"][
                        "title"
                    ],
                    open=True,
                ):
                    # Create a button for preprocessing
                    clean_checkbox = gr.inputs.Checkbox(label="Clean")
                    num_worker_textbox = gr.inputs.Textbox(
                        label="Number of workers",
                        default="8",
                        numeric=True,
                    )
                    extract_features_button = gr.Button(
                        self.language_conf["main_ui"]["Train_tab"]["preprocessing"][
                            "btn"
                        ]
                    )
                    # preprocessing_progress = gr.Progress()
                    preprocess_log = gr.inputs.Textbox(
                        label="Preprocessing log",
                        default="preprocessing.log",
                    )
                    preprocessing_output = gr.Textbox(label="Preprocessing output")
                    log_viewer = gr.Textbox(
                        label="Logs (updates every second)", lines=3
                    )
                    config = config_path if config_path != "" else config_select_path
                    extract_features_button.click(
                        fn=self.start_preprocessing,
                        inputs=[
                            config,
                            clean_checkbox,
                            num_worker_textbox,
                            preprocess_log,
                        ],
                        outputs=preprocessing_output,
                    )
                    dep = ui.load(
                        self.show_log,
                        inputs=preprocess_log,
                        outputs=log_viewer,
                        every=0.5,
                    )

                entity = gr.inputs.Textbox(
                    label="Entity for wandb", default="fish-audio"
                )
                tensorboard = gr.inputs.Checkbox(label="Log to tensorboard")
                resume = gr.inputs.Checkbox(
                    label="Resume training using the latest ckpt"
                )
                resume_id = gr.inputs.Textbox(label="Resume id for training")
                checkpoint = gr.inputs.Textbox(label="Resume training ckpt file")
                run_name = gr.inputs.Textbox(label="Run name for wandb")
                pretrained = gr.inputs.Checkbox(label="Use pretrained")

                # Create a button to start the training
                start_train_button = gr.Button(
                    self.language_conf["main_ui"]["Train_tab"]["train"]["btn"]
                )
                training_output = gr.Textbox(label="Training output")

                train_log = gr.inputs.Textbox(
                    label="Training log",
                    default="train_ui.log",
                )
                # Define the function to run the training script and set it as the button click action
                start_train_button.click(
                    fn=self.start_training,
                    inputs=[
                        config_path,
                        entity,
                        tensorboard,
                        resume,
                        resume_id,
                        checkpoint,
                        run_name,
                        pretrained,
                        train_log,
                    ],
                    outputs=training_output,
                )
                train_log_viewer = gr.Textbox(
                    label="Logs (updates every second)", lines=3
                )
                train_dep = ui.load(
                    self.show_log,
                    inputs=train_log,
                    outputs=train_log_viewer,
                    every=1,
                )
            with gr.Tab(self.language_conf["main_ui"]["Inference_tab"]["title"]):
                # Assuming the language_conf dict has the same structure as the previous example
                gr.Markdown(self.language_conf["main_ui"]["Inference_tab"]["content"])

        self.ui = ui


# todo: click

if __name__ == "__main__":
    webui = WebUI(EN_CONFIG)
    webui.lauch(
        server_name="100.70.71.133",
        server_port=7860,
    )
