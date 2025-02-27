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

    total_process_steps = 12
    irrelevant_steps = 3
    relevant_steps = total_process_steps - irrelevant_steps  # 9 steps

    # Create an overall process bar for the 9 relevant steps.
    overall_bar = tqdm(total=relevant_steps, desc="Overall Process", position=1, dynamic_ncols=True, leave=True)
    processed_steps = 0

    # Regex to extract the INFO message from each log line.
    info_pattern = re.compile(r"\[.*?\]\s+INFO:\s+(.*)")
    # Regex to capture progress lines from video generation (like " 10%|...| 5/50").
    progress_pattern = re.compile(r"(\d+)%\|.*\| (\d+)/(\d+)")
    
    gen_progress_bar = None

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

        # Check if this line is a progress update for video generation.
        progress_match = progress_pattern.search(stripped_line)
        if progress_match:
            current = int(progress_match.group(2))
            total = int(progress_match.group(3))
            if gen_progress_bar is None:
                gen_progress_bar = tqdm(total=total, desc="Video Generation", position=0, dynamic_ncols=True, leave=True)
            # Update the generation progress bar by the difference.
            gen_progress_bar.update(current - gen_progress_bar.n)
            gen_progress_bar.refresh()
            continue  # Skip further processing of this line.

        # Check for an INFO log line.
        info_match = info_pattern.search(stripped_line)
        if info_match:
            msg = info_match.group(1)
            # Skip the first three INFO messages.
            if processed_steps < irrelevant_steps:
                processed_steps += 1
            else:
                overall_bar.update(1)
                # Compute the current percentage.
                percentage = (overall_bar.n / overall_bar.total) * 100
                # Set the description to include both the percentage and the current info title.
                overall_bar.set_description(f"Overall Process - {percentage:.0f}% | {msg}")
            # Write the log line as well.
            tqdm.write(stripped_line)
        else:
            tqdm.write(stripped_line)

    process.wait()
    if gen_progress_bar is not None:
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