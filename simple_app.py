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
    relevant_steps = total_process_steps - irrelevant_steps

    overall_bar = tqdm(total=relevant_steps, desc="Overall Process", position=1,
                       ncols=120, dynamic_ncols=False, leave=True)
    processed_steps = 0

    progress_pattern = re.compile(r"(\d+)%\|.*\| (\d+)/(\d+)")
    gen_progress_bar = None

    current_sub_bar = None
    current_cancel_event = None
    sub_lock = threading.Lock()

    def update_sub_bar(sub_bar, cancel_event):
        """Updates the sub-bar every second up to 20 seconds unless canceled."""
        for _ in range(20):
            if cancel_event.is_set():
                break
            time.sleep(1)
            with sub_lock:
                if sub_bar.n < sub_bar.total:
                    sub_bar.update(1)
                    sub_bar.refresh()

    def cancel_sub_bar():
        """Cancels the current sub-bar and advances the overall process."""
        nonlocal current_sub_bar, current_cancel_event
        with sub_lock:
            if current_cancel_event:
                current_cancel_event.set()
            if current_sub_bar:
                remaining = current_sub_bar.total - current_sub_bar.n
                if remaining > 0:
                    current_sub_bar.update(remaining)
                current_sub_bar.close()
                current_sub_bar = None
            overall_bar.update(1)
            overall_bar.refresh()
            current_cancel_event = None

    command = [
        "python", "-u", "-m", "generate",
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

        # Check for video generation progress
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

        # Check for INFO messages
        if "INFO:" in stripped_line:
            parts = stripped_line.split("INFO:", 1)
            msg = parts[1].strip() if len(parts) > 1 else ""
            tqdm.write(stripped_line)

            if processed_steps < irrelevant_steps:
                processed_steps += 1
            else:
                with sub_lock:
                    # Cancel the previous sub-bar if it exists
                    cancel_sub_bar()
                    # Start a new sub-bar
                    current_sub_bar = tqdm(total=20, desc=msg, position=2,
                                           ncols=120, dynamic_ncols=False, leave=True)
                    current_cancel_event = threading.Event()
                    threading.Thread(target=update_sub_bar, args=(current_sub_bar, current_cancel_event),
                                     daemon=True).start()
            continue

        else:
            tqdm.write(stripped_line)

    process.wait()
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