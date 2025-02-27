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

    last_msg = ""  # To store the last INFO message

    for line in iter(process.stdout.readline, ''):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        # Match video generation progress (e.g., "10%|...| 5/50")
        progress_match = progress_pattern.search(stripped_line)
        if progress_match:
            current = int(progress_match.group(2))
            total = int(progress_match.group(3))
            if gen_progress_bar is None:
                gen_progress_bar = tqdm(total=total, desc="Video Generation", position=0, dynamic_ncols=True, leave=True)
            gen_progress_bar.update(current - gen_progress_bar.n)
            gen_progress_bar.refresh()
            continue

        # Check for INFO lines and extract the message
        if "INFO:" in stripped_line:
            parts = stripped_line.split("INFO:", 1)
            msg = parts[1].strip() if len(parts) > 1 else ""

            # Debugging print to check extracted message
            tqdm.write(f"Extracted INFO message: '{msg}'")

            # Skip first three steps
            if processed_steps < irrelevant_steps:
                processed_steps += 1
            else:
                overall_bar.update(1)
                percentage = (overall_bar.n / overall_bar.total) * 100
                last_msg = msg  # Store last INFO message

                # Debugging print before updating description
                tqdm.write(f"Updating description: Overall Process - {percentage:.1f}% | {last_msg}")

                # Update progress bar description with INFO message
                overall_bar.set_description(f"Overall Process - {percentage:.1f}% | {last_msg}")
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