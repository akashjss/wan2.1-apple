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

    # Create an overall progress bar for the 9 relevant steps.
    overall_bar = tqdm(total=relevant_steps, desc="Overall Process", position=1, dynamic_ncols=True, leave=True)
    processed_steps = 0

    # Regex to extract the INFO message (everything after "INFO:")
    info_pattern = re.compile(r"\[.*?\]\s+INFO:\s+(.*)")
    # Regex to capture video generation progress lines (e.g., " 10%|...| 5/50")
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

        # Check if this is a video generation progress line.
        progress_match = progress_pattern.search(stripped_line)
        if progress_match:
            current = int(progress_match.group(2))
            total = int(progress_match.group(3))
            if gen_progress_bar is None:
                gen_progress_bar = tqdm(total=total, desc="Video Generation", position=0, dynamic_ncols=True, leave=True)
            # Update the video generation progress bar
            gen_progress_bar.update(current - gen_progress_bar.n)
            gen_progress_bar.refresh()
            continue  # Skip further processing for progress lines

        # Check if this is an INFO log line.
        info_match = info_pattern.search(stripped_line)
        if info_match:
            msg = info_match.group(1)
            # Always print the log line (using tqdm.write so it doesn't interfere with the bars)
            tqdm.write(stripped_line)
            # For relevant steps (i.e. after the first three), update the overall progress.
            if processed_steps < irrelevant_steps:
                processed_steps += 1
            else:
                overall_bar.update(1)
                percentage = (overall_bar.n / overall_bar.total) * 100
                # Update the description for the left part and set the postfix to the INFO message.
                overall_bar.set_description(f"Overall Process - {percentage:.1f}%")
                overall_bar.set_postfix_str(msg)
                overall_bar.refresh()
        else:
            # For any other line, print it.
            tqdm.write(stripped_line)

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