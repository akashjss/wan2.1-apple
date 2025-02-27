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
    relevant_steps = total_process_steps - irrelevant_steps  # 7 steps

    # Overall progress bar for the process steps.
    overall_bar = tqdm(total=relevant_steps, desc="Overall Process", position=1,
                       ncols=120, dynamic_ncols=False, leave=True)
    processed_steps = 0

    # Regex for detecting video generation progress lines (e.g., "10%|...| 5/50")
    progress_pattern = re.compile(r"(\d+)%\|.*\| (\d+)/(\d+)")
    gen_progress_bar = None

    # Variables for managing the sub-progress bar for each step.
    current_sub_bar = None
    current_sub_thread = None
    current_cancel_event = None
    sub_lock = threading.Lock()

    def update_sub_bar(sub_bar, cancel_event):
        # This function updates the sub_bar once per second up to 20 seconds,
        # unless the cancel_event is set.
        for i in range(20):
            if cancel_event.is_set():
                break
            time.sleep(1)
            sub_bar.update(1)
            sub_bar.refresh()
        # When done (or canceled), do nothing here;
        # closing will be done in close_sub_bar() from the main thread.
    
    def close_sub_bar():
        nonlocal current_sub_bar, current_sub_thread, current_cancel_event
        with sub_lock:
            if current_sub_bar is not None:
                try:
                    # Complete any remaining ticks (if any)
                    remaining = current_sub_bar.total - current_sub_bar.n
                    if remaining > 0:
                        current_sub_bar.update(remaining)
                except Exception:
                    pass
                current_sub_bar.close()
                overall_bar.update(1)
                overall_bar.refresh()
                current_sub_bar = None
            if current_sub_thread is not None:
                current_sub_thread.join()
                current_sub_thread = None
            if current_cancel_event is not None:
                current_cancel_event = None

    command = [
        "python", "-u", "-m", "generate",  # using -u for unbuffered output and module name without .py
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

        # Check if this is a video generation progress line.
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
            parts = stripped_line.split("INFO:", 1)
            msg = parts[1].strip() if len(parts) > 1 else ""
            tqdm.write(stripped_line)

            if processed_steps < irrelevant_steps:
                processed_steps += 1
            else:
                with sub_lock:
                    # If a sub-bar is already active, cancel its update and close it.
                    if current_sub_bar is not None:
                        if current_cancel_event is not None:
                            current_cancel_event.set()
                        close_sub_bar()
                    # Now create a new sub-bar for the current step.
                    current_sub_bar = tqdm(total=20, desc=msg, position=2,
                                           ncols=120, dynamic_ncols=False, leave=True)
                    current_cancel_event = threading.Event()
                    current_sub_thread = threading.Thread(target=update_sub_bar,
                                                          args=(current_sub_bar, current_cancel_event))
                    current_sub_thread.start()
            continue

        else:
            tqdm.write(stripped_line)

    process.wait()
    # After process ends, if a sub-bar is still active, cancel and close it.
    with sub_lock:
        if current_cancel_event is not None:
            current_cancel_event.set()
        if current_sub_bar is not None:
            close_sub_bar()

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