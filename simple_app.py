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
    
    total_process_steps = 11
    irrelevant_steps = 4
    # Only steps 5 through 11 (i.e. 7 steps) count.
    relevant_steps = total_process_steps - irrelevant_steps

    # Create the overall progress bar for the steps.
    overall_bar = tqdm(total=relevant_steps, desc="Overall Process", position=1,
                       ncols=120, dynamic_ncols=False, leave=True)
    processed_steps = 0

    # Regex for detecting video generation progress lines (e.g. "10%|...| 5/50")
    progress_pattern = re.compile(r"(\d+)%\|.*\| (\d+)/(\d+)")
    gen_progress_bar = None

    # Variables for managing the sub-progress bar for each step.
    current_sub_bar = None
    current_sub_thread = None
    current_cancel_event = None
    sub_lock = threading.Lock()

    def update_sub_bar(sub_bar, cancel_event):
        # Update sub-bar once per second for up to 20 seconds,
        # unless cancel_event is set.
        for i in range(20):
            if cancel_event.is_set():
                break
            time.sleep(1)
            sub_bar.update(1)
            sub_bar.refresh()
        # (Closing and overall-bar update are handled externally.)

    def cancel_sub_bar():
        nonlocal current_sub_bar, current_sub_thread, current_cancel_event
        with sub_lock:
            if current_cancel_event is not None:
                current_cancel_event.set()
            if current_sub_thread is not None:
                current_sub_thread.join(timeout=1)
                current_sub_thread = None
            if current_sub_bar is not None:
                # Complete any remaining ticks.
                remaining = current_sub_bar.total - current_sub_bar.n
                if remaining > 0:
                    current_sub_bar.update(remaining)
                current_sub_bar.close()
                current_sub_bar = None
            # Update overall progress by one step.
            overall_bar.update(1)
            overall_bar.refresh()
            current_cancel_event = None

    command = [
        "python", "-u", "-m", "generate",  # using unbuffered mode
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
        bufsize=1  # line-buffered
    )

    for line in iter(process.stdout.readline, ''):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        # Check for video generation progress lines.
        progress_match = progress_pattern.search(stripped_line)
        if progress_match:
            current = int(progress_match.group(2))
            total = int(progress_match.group(3))
            if gen_progress_bar is None:
                gen_progress_bar = tqdm(total=total, desc="Video Generation", position=0,
                                        ncols=120, dynamic_ncols=True, leave=True)
            gen_progress_bar.update(current - gen_progress_bar.n)
            gen_progress_bar.refresh()
            continue

        # Check for INFO lines.
        if "INFO:" in stripped_line:
            # Extract the INFO message (the text after "INFO:")
            parts = stripped_line.split("INFO:", 1)
            msg = parts[1].strip() if len(parts) > 1 else ""
            tqdm.write(stripped_line)  # Log the full line

            if processed_steps < irrelevant_steps:
                processed_steps += 1
            else:
                with sub_lock:
                    # If a sub-bar is active, cancel it immediately.
                    if current_sub_bar is not None:
                        cancel_sub_bar()
                    # Create a new sub-progress bar for this step (lasting up to 20 seconds).
                    current_sub_bar = tqdm(total=20, desc=msg, position=2,
                                           ncols=120, dynamic_ncols=False, leave=True)
                    current_cancel_event = threading.Event()
                    current_sub_thread = threading.Thread(target=update_sub_bar, args=(current_sub_bar, current_cancel_event))
                    current_sub_thread.daemon = True
                    current_sub_thread.start()
            continue

        else:
            tqdm.write(stripped_line)

    # Process has ended; cancel any active sub-progress bar.
    process.wait()
    with sub_lock:
        if current_cancel_event is not None:
            current_cancel_event.set()
        if current_sub_bar is not None:
            cancel_sub_bar()

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