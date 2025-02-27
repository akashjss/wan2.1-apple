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
    sub_ticks = 0  # Each tick represents 40ms
    sub_tick_total = 500  # 500 ticks * 0.04 sec = 20 seconds
    video_phase = False

    command = [
        "python", "-u", "-m", "generate",  # using -u for unbuffered output
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

    while True:
        # Poll for new stdout data with a 40 ms timeout.
        rlist, _, _ = select.select([process.stdout], [], [], 0.04)
        if rlist:
            line = process.stdout.readline()
            if not line:
                break
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # Check for video generation progress (level 3).
            progress_match = progress_pattern.search(stripped_line)
            if progress_match:
                # If a sub-step bar is active, finish it before entering video phase.
                if sub_bar is not None:
                    if sub_ticks < sub_tick_total:
                        sub_bar.update(sub_tick_total - sub_ticks)
                    sub_bar.close()
                    overall_bar.update(1)
                    overall_bar.refresh()
                    sub_bar = None
                    sub_ticks = 0
                video_phase = True
                current = int(progress_match.group(2))
                total = int(progress_match.group(3))
                if video_progress_bar is None:
                    video_progress_bar = tqdm(total=total, desc="Video Generation", position=0,
                                              ncols=120, dynamic_ncols=True, leave=True)
                video_progress_bar.update(current - video_progress_bar.n)
                video_progress_bar.refresh()
                if video_progress_bar.n >= video_progress_bar.total:
                    video_phase = False
                    overall_bar.update(1)
                    overall_bar.refresh()
                    video_progress_bar.close()
                    video_progress_bar = None
                continue

            # Process INFO messages (level 2 sub-step).
            if "INFO:" in stripped_line:
                parts = stripped_line.split("INFO:", 1)
                msg = parts[1].strip() if len(parts) > 1 else ""
                print(stripped_line)  # Print log line

                if processed_steps < irrelevant_steps:
                    processed_steps += 1
                else:
                    # If in video phase, ignore new INFO messages.
                    if video_phase:
                        continue
                    # If a sub-step is already active, finish it.
                    if sub_bar is not None:
                        if sub_ticks < sub_tick_total:
                            sub_bar.update(sub_tick_total - sub_ticks)
                        sub_bar.close()
                        overall_bar.update(1)
                        overall_bar.refresh()
                        sub_bar = None
                        sub_ticks = 0
                    # Start a new sub-step progress bar with total=500 ticks.
                    sub_bar = tqdm(total=sub_tick_total, desc=msg, position=2,
                                   ncols=120, dynamic_ncols=False, leave=True)
                    sub_ticks = 0
                continue
            else:
                print(stripped_line)
        else:
            # No new data within 40ms; update the sub-step progress bar.
            if sub_bar is not None:
                sub_bar.update(1)
                sub_ticks += 1
                sub_bar.refresh()
                if sub_ticks >= sub_tick_total:
                    # 20 seconds have elapsed; complete this sub-step.
                    sub_bar.close()
                    overall_bar.update(1)
                    overall_bar.refresh()
                    sub_bar = None
                    sub_ticks = 0

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