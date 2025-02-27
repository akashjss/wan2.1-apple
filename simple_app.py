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

    progress_bar = None
    # Regex pattern to capture lines like " 10%|█         | 5/50"
    progress_pattern = re.compile(r"(\d+)%\|.*\| (\d+)/(\d+)")
    
    for line in iter(process.stdout.readline, ''):
        match = progress_pattern.search(line)
        if match:
            # Update tqdm without printing the line to logs
            current = int(match.group(2))
            total = int(match.group(3))
            if progress_bar is None:
                progress_bar = tqdm(total=total, desc="Video Generation Progress", leave=True)
            progress_bar.update(current - progress_bar.n)
            # Optionally, refresh the tqdm display without a newline:
            progress_bar.refresh()
        else:
            # For non-progress lines, print them normally
            print(line, end="")

    process.wait()
    if progress_bar is not None:
        progress_bar.close()

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