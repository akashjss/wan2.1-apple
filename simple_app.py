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

    # This bar will track the generation progress (extracted from the stdout progress lines)
    gen_progress_bar = None
    # This bar will "simulate" a progress update for each log line (non-progress messages).
    # We start with a total of 0 and update its total dynamically.
    log_progress_bar = tqdm(total=0, desc="Logs", position=1, dynamic_ncols=True, leave=True)
    
    progress_pattern = re.compile(r"(\d+)%\|.*\| (\d+)/(\d+)")
    
    for line in iter(process.stdout.readline, ''):
        # Remove whitespace so we can check for empty lines.
        stripped_line = line.strip()
        if not stripped_line:
            continue
        
        # Check if the line matches the progress bar format from the external process.
        match = progress_pattern.search(stripped_line)
        if match:
            # Extract current step and total from the match.
            current = int(match.group(2))
            total = int(match.group(3))
            if gen_progress_bar is None:
                gen_progress_bar = tqdm(total=total, desc="Video Generation Progress", position=0, dynamic_ncols=True, leave=True)
            # Update generation progress (ensuring we only advance by the difference)
            gen_progress_bar.update(current - gen_progress_bar.n)
            gen_progress_bar.refresh()
        else:
            # For any log line that is not part of the progress output, update the fake log track.
            # Increase the total count by one and update one step.
            log_progress_bar.total += 1
            log_progress_bar.update(1)
            # Write the log line so it appears in order above the progress bars.
            tqdm.write(stripped_line)
    
    process.wait()
    if gen_progress_bar is not None:
        gen_progress_bar.close()
    log_progress_bar.close()

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