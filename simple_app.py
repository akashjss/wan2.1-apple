import gradio as gr
import re 
import subprocess
import time
import select
from tqdm import tqdm
from huggingface_hub import snapshot_download

#Download model
snapshot_download(
    repo_id = "Wan-AI/Wan2.1-T2V-1.3B",
    local_dir = "./Wan2.1-T2V-1.3B"
)

def infer(prompt, progress=gr.Progress(track_tqdm=True)):
    
    # Configuration:  
    total_process_steps = 11          # Total steps (including irrelevant ones)
    irrelevant_steps = 4              # First 4 INFO messages are skipped  
    relevant_steps = total_process_steps - irrelevant_steps  # 7 overall steps

    # Create overall progress bar (level 1)
    overall_bar = tqdm(total=relevant_steps, desc="Overall Process", position=1,
                       ncols=120, dynamic_ncols=False, leave=True)
    processed_steps = 0

    # Regex for video generation progress (level 3)
    progress_pattern = re.compile(r"(\d+)%\|.*\| (\d+)/(\d+)")
    video_progress_bar = None

    # Variables for sub-step progress bar (level 2)
    sub_bar = None
    sub_time_elapsed = 0  # seconds elapsed for the current sub-step
    video_phase = False   # flag indicating video generation phase

    command = [
        "python", "-u", "-m", "generate",  # -u: unbuffered output
        "--task", "t2v-1.3B",
        "--size", "832*480",
        "--ckpt_dir", "./Wan2.1-T2V-1.3B",
        "--sample_shift", "8",
        "--sample_guide_scale", "6",
        "--prompt", prompt,
        "--save_file", "generated_video.mp4"
    ]

    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True,
                               bufsize=1)

    # Poll the process's stdout in a loop.
    while True:
        # Wait up to 1 second for data.
        rlist, _, _ = select.select([process.stdout], [], [], 1)
        if rlist:
            # New line is available.
            line = process.stdout.readline()
            if not line:
                break
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # Check if line matches video generation progress.
            progress_match = progress_pattern.search(stripped_line)
            if progress_match:
                # Enter video phase: if a sub-step is active, finish it.
                if sub_bar is not None:
                    if sub_time_elapsed < 20:
                        sub_bar.update(20 - sub_time_elapsed)
                    sub_bar.close()
                    overall_bar.update(1)
                    overall_bar.refresh()
                    sub_bar = None
                    sub_time_elapsed = 0
                video_phase = True
                current = int(progress_match.group(2))
                total = int(progress_match.group(3))
                if video_progress_bar is None:
                    video_progress_bar = tqdm(total=total, desc="Video Generation", position=0,
                                              ncols=120, dynamic_ncols=True, leave=True)
                video_progress_bar.update(current - video_progress_bar.n)
                video_progress_bar.refresh()
                # When video progress is complete, finish the video phase.
                if video_progress_bar.n >= video_progress_bar.total:
                    video_phase = False
                    overall_bar.update(1)
                    overall_bar.refresh()
                    video_progress_bar.close()
                    video_progress_bar = None
                continue

            # Process INFO messages.
            if "INFO:" in stripped_line:
                parts = stripped_line.split("INFO:", 1)
                msg = parts[1].strip() if len(parts) > 1 else ""
                # Print the log line.
                print(stripped_line)

                if processed_steps < irrelevant_steps:
                    processed_steps += 1
                else:
                    # If we're in video phase, ignore new INFO messages.
                    if video_phase:
                        continue
                    # If a sub-step is already active, finish it.
                    if sub_bar is not None:
                        if sub_time_elapsed < 20:
                            sub_bar.update(20 - sub_time_elapsed)
                        sub_bar.close()
                        overall_bar.update(1)
                        overall_bar.refresh()
                        sub_bar = None
                        sub_time_elapsed = 0
                    # Start a new sub-step progress bar.
                    sub_bar = tqdm(total=20, desc=msg, position=2,
                                   ncols=120, dynamic_ncols=False, leave=True)
                    sub_time_elapsed = 0
                continue
            else:
                print(stripped_line)
        else:
            # No new data for 1 second.
            if sub_bar is not None:
                sub_bar.update(1)
                sub_time_elapsed += 1
                sub_bar.refresh()
                if sub_time_elapsed >= 20:
                    # Complete this sub-step.
                    sub_bar.close()
                    overall_bar.update(1)
                    overall_bar.refresh()
                    sub_bar = None
                    sub_time_elapsed = 0

        # Exit loop if the process is finished.
        if process.poll() is not None:
            break

    # Drain any remaining output.
    for line in process.stdout:
        print(line.strip())

    process.wait()
    if video_progress_bar is not None:
        video_progress_bar.close()
    if sub_bar is not None:
        sub_bar.close()
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