import gradio as gr
import re 
import subprocess
import time
import threading
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
    # Relevant steps = 11 - 4 = 7 overall steps that will be shown
    relevant_steps = total_process_steps - irrelevant_steps

    # Create overall process progress bar (level 1)
    overall_bar = tqdm(total=relevant_steps, desc="Overall Process", position=1,
                       ncols=120, dynamic_ncols=False, leave=True)
    processed_steps = 0

    # Regex to capture video generation progress lines (for level 3)
    progress_pattern = re.compile(r"(\d+)%\|.*\| (\d+)/(\d+)")
    video_progress_bar = None

    # Variables for managing sub-step progress bar (level 2)
    current_sub_bar = None
    current_cancel_event = None
    sub_lock = threading.Lock()
    current_sub_thread = None

    # A flag to indicate if we are in video generation phase.
    video_phase = False

    def update_sub_bar(sub_bar, cancel_event):
        # Tick sub_bar once per second for up to 20 seconds
        for _ in range(20):
            if cancel_event.is_set():
                break
            time.sleep(1)
            with sub_lock:
                if sub_bar.n < sub_bar.total:
                    sub_bar.update(1)
                    sub_bar.refresh()

    def cancel_sub_bar():
        nonlocal current_sub_bar, current_cancel_event
        with sub_lock:
            if current_cancel_event:
                current_cancel_event.set()
            if current_sub_bar:
                # Finish any remaining ticks
                remaining = current_sub_bar.total - current_sub_bar.n
                if remaining > 0:
                    current_sub_bar.update(remaining)
                current_sub_bar.close()
                current_sub_bar = None
            overall_bar.update(1)
            overall_bar.refresh()
            current_cancel_event = None

    # Build the command.
    command = [
        "python", "-u", "-m", "generate",  # -u forces unbuffered output
        "--task", "t2v-1.3B",
        "--size", "832*480",
        "--ckpt_dir", "./Wan2.1-T2V-1.3B",
        "--sample_shift", "8",
        "--sample_guide_scale", "6",
        "--prompt", prompt,
        "--save_file", "generated_video.mp4"
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in iter(process.stdout.readline, ''):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        # Check for video generation progress (level 3)
        progress_match = progress_pattern.search(stripped_line)
        if progress_match:
            # On the first video progress line, if not already in video phase:
            if not video_phase:
                # Cancel any active sub-step bar before entering video phase.
                with sub_lock:
                    if current_sub_bar:
                        cancel_sub_bar()
                video_phase = True
                # Initialize video progress bar.
                # Here we assume the total will come from the log; if not, adjust as needed.
                current = int(progress_match.group(2))
                total = int(progress_match.group(3))
                if video_progress_bar is None:
                    video_progress_bar = tqdm(total=total, desc="Video Generation", position=0,
                                              ncols=120, dynamic_ncols=True, leave=True)
            # Update video generation progress.
            current = int(progress_match.group(2))
            total = int(progress_match.group(3))
            video_progress_bar.update(current - video_progress_bar.n)
            video_progress_bar.refresh()
            # If video progress is complete, finish the video phase.
            if video_progress_bar.n >= video_progress_bar.total:
                video_phase = False
                overall_bar.update(1)
                overall_bar.refresh()
                video_progress_bar.close()
                video_progress_bar = None
            continue

        # Process INFO messages (level 2 sub-step)
        if "INFO:" in stripped_line:
            # Extract the text after "INFO:"
            parts = stripped_line.split("INFO:", 1)
            msg = parts[1].strip() if len(parts) > 1 else ""
            tqdm.write(stripped_line)

            # Skip the first 4 irrelevant INFO messages.
            if processed_steps < irrelevant_steps:
                processed_steps += 1
            else:
                # If we are in video phase, ignore new INFO messages (or optionally queue them).
                if video_phase:
                    continue

                # If a sub-step bar is already active, cancel it.
                with sub_lock:
                    if current_sub_bar is not None:
                        cancel_sub_bar()
                    # Create a new sub-step bar for this INFO message.
                    current_cancel_event = threading.Event()
                    current_sub_bar = tqdm(total=20, desc=msg, position=2,
                                           ncols=120, dynamic_ncols=False, leave=True)
                    current_sub_thread = threading.Thread(
                        target=update_sub_bar,
                        args=(current_sub_bar, current_cancel_event),
                        daemon=True
                    )
                    current_sub_thread.start()
            continue

        else:
            tqdm.write(stripped_line)

    # Process finished; clean up any active sub-step.
    process.wait()
    with sub_lock:
        if current_cancel_event:
            current_cancel_event.set()
        if current_sub_bar:
            cancel_sub_bar()
    if video_progress_bar:
        video_progress_bar.close()
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