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

    overall_bar = tqdm(total=relevant_steps, desc="Overall Process", position=1, dynamic_ncols=True, leave=True)
    processed_steps = 0

    # Regex to capture video generation progress lines (e.g., "10%|...| 5/50")
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

        # First, check for video generation progress lines.
        progress_match = progress_pattern.search(stripped_line)
        if progress_match:
            current = int(progress_match.group(2))
            total = int(progress_match.group(3))
            if gen_progress_bar is None:
                gen_progress_bar = tqdm(total=total, desc="Video Generation", position=0, dynamic_ncols=True, leave=True)
            gen_progress_bar.update(current - gen_progress_bar.n)
            gen_progress_bar.refresh()
            continue

        # Now, check for INFO lines in a case-insensitive way.
        if "info:" in stripped_line.lower():
            # Split on "INFO:" (in a case-insensitive manner) to get the text after it.
            parts = stripped_line.split("INFO:", 1)
            msg = parts[1].strip() if len(parts) > 1 else ""
            # Print the log line.
            tqdm.write(stripped_line)
            # Skip updating the overall bar for the first three (irrelevant) steps.
            if processed_steps < irrelevant_steps:
                processed_steps += 1
            else:
                overall_bar.update(1)
                percentage = (overall_bar.n / overall_bar.total) * 100
                # Update the overall bar description with both the percentage and the extracted INFO message.
                overall_bar.set_description(f"Overall Process - {percentage:.1f}% | {msg}")
                overall_bar.refresh()
        else:
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