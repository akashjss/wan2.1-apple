import gradio as gr
import re 
import subprocess
from tqdm import tqdm
from huggingface_hub import snapshot_download

#Download model
snapshot_download(
    repo_id = "Wan-AI/Wan2.1-T2V-1.3B",
    local_dir = "./Wan2.1-T2V-1.3B"
)

def infer(prompt, progress=gr.Progress(track_tqdm=True)):
    
    total_process_steps = 11
    irrelevant_steps = 4
    relevant_steps = total_process_steps - irrelevant_steps  # 7 steps

    # Create the overall progress bar for the process steps.
    overall_bar = tqdm(total=relevant_steps, desc="Overall Process", position=1, dynamic_ncols=True, leave=True)
    processed_steps = 0

    # Regex for detecting video generation progress lines.
    progress_pattern = re.compile(r"(\d+)%\|.*\| (\d+)/(\d+)")
    gen_progress_bar = None

    # We'll maintain a persistent sub-progress bar for the current step.
    current_sub_bar = None

    command = [
        "python", "-u", "-m", "generate",  # using -u for unbuffered output and omitting .py extension
        "--task", "t2v-1.3B",
        "--size", "832*480",
        "--ckpt_dir", "./Wan2.1-T2V-1.3B",
        "--sample_shift", "8",
        "--sample_guide_scale", "6",
        "--prompt", prompt,
        "--save_file", "generated_video.mp4"
    ]

    # Start the process with unbuffered output and combine stdout and stderr.
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # line-buffered
    )

    for line in iter(process.stdout.readline, ''):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        # Check if this is a video generation progress line.
        progress_match = progress_pattern.search(stripped_line)
        if progress_match:
            current = int(progress_match.group(2))
            total = int(progress_match.group(3))
            if gen_progress_bar is None:
                gen_progress_bar = tqdm(total=total, desc="Video Generation", position=0,
                                        dynamic_ncols=True, leave=True)
            gen_progress_bar.update(current - gen_progress_bar.n)
            gen_progress_bar.refresh()
            continue

        # Check for INFO lines.
        if "INFO:" in stripped_line:
            # Extract the INFO message.
            parts = stripped_line.split("INFO:", 1)
            msg = parts[1].strip() if len(parts) > 1 else ""
            tqdm.write(stripped_line)  # print the log line

            # Skip the first few irrelevant INFO lines.
            if processed_steps < irrelevant_steps:
                processed_steps += 1
            else:
                # If there's a current sub-progress bar, mark it complete and close it.
                if current_sub_bar is not None:
                    current_sub_bar.update(1)
                    current_sub_bar.close()
                    overall_bar.update(1)
                    overall_bar.refresh()
                # Now create a new sub-progress bar for this new step.
                # (It will remain visible until the next INFO message arrives.)
                current_sub_bar = tqdm(total=1, desc=msg, position=2,
                                       ncols=120, dynamic_ncols=False, leave=True)
        else:
            tqdm.write(stripped_line)

    # When process ends, if there's an open sub-bar, finish it.
    if current_sub_bar is not None:
        current_sub_bar.update(1)
        current_sub_bar.close()
        overall_bar.update(1)
        overall_bar.refresh()

    process.wait()
    if gen_progress_bar:
        gen_progress_bar.close()
    overall_bar.close()
    
    if process.returncode == 0:
        print("Command executed successfully.")
        return "generated_video.mp4"
    else:
        print("Error executing command.")
        raise Exception("Error executing command")

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("# Wan 2.1")
        prompt = gr.Textbox(label="Prompt")
        submit_btn = gr.Button("Submit")
        video_res = gr.Video(label="Generated Video")

    submit_btn.click(
        fn = infer,
        inputs = [prompt],
        outputs = [video_res]
    )

demo.queue().launch(show_error=True, show_api=False, ssr_mode=False)