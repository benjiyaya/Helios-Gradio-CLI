## Helios-Distilled Gradio WebUI

This folder contains a small Gradio WebUI for running the **Helios-Distilled** video model (`BestWishYsh/Helios-Distilled`) locally with Diffusers, supporting **Text → Video**, and **Image → Video**.

The implementation follows the official usage examples from the model card on Hugging Face (`https://huggingface.co/BestWishYsh/Helios-Distilled`).

### 1. Install CUDA and CUDA-enabled PyTorch (required for GPU)

Helios runs on **GPU**; without CUDA, PyTorch uses the CPU and the model runs from RAM, which is very slow.

1. **Install NVIDIA CUDA Toolkit** (if not already installed):
   - Check your GPU driver supports it: [NVIDIA Driver Support](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).
   - Download and install CUDA from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (e.g. CUDA 12.x for recent GPUs).
   - Reboot if the installer asks you to.

2. **Install PyTorch with CUDA support** inside your `.venv`:
   - Go to [PyTorch Get Started](https://pytorch.org/get-started/locally/) and pick your OS, pip, and CUDA version (e.g. CUDA 12.4).
   - Run the suggested `pip install` command, for example:
     ```powershell
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
     ```
   - Then install the rest of the project: `pip install -r requirements.txt` (this may reinstall CPU-only torch if your requirements pin `torch`; in that case run the PyTorch CUDA `pip install` again after).

3. **Check that the GPU is used** when you start the app: the console should print something like `Using device: cuda`. If it prints `Using device: cpu`, CUDA is not available and the model will load into RAM and run slowly.

### 2. Prepare Python environment (.venv)

From the project root (`\Helios`):

```powershell
python -m venv .venv

for Lunix: .venv\Scripts\activate

for Windows: .\.venv\Scripts\activate.bat

Set a temp folder for your image input

Powershell >>> $env:GRADIO_TEMP_DIR = "F:\Helios-ui\outputs\gradio_tmp"
Or 
CMD >>> set GRADIO_TEMP_DIR=F:\Helios-ui\outputs\gradio_tmp


pip install --upgrade pip

pip install -r requirements.txt
```

> Make sure you have a recent NVIDIA GPU with enough VRAM (at least ~6 GB recommended if using CPU offload) and a recent CUDA-enabled PyTorch.

### 3. Download the model into `models/`

Create a `models` folder and download **Helios-Distilled** there (you mentioned this is already done):

```powershell
mkdir models
cd models
huggingface-cli download BestWishYsh/Helios-Distilled --local-dir Helios-Distilled
cd ..
```

If your local path is different, set:

```powershell
$env:HELIOS_MODEL_PATH = "models\Helios-Distilled"
```

before starting the WebUI.

### 4. Run the Gradio WebUI

From `f:\Helios` with `.venv` activated:

```powershell
python helios_gradio.py
```

Then open the printed local URL (by default `http://127.0.0.1:7860`) in your browser.

### 5. Features

- **Text → Video**: type a prompt, adjust **width**, **height**, frames/FPS/pyramid steps/guidance/seed, and generate a new video.
- **Image → Video**: upload a still image (resized to your chosen width×height) and generate motion guided by an optional prompt.
- **Video → Video**: upload an input video (e.g. MP4), optionally add a guiding prompt; output uses your chosen width×height.

**Video resolution (width × height):**
- **Gradio:** Each tab has **Width** and **Height** number inputs. Default **640×384**. Allowed range: width 256–1280, height 256–720. Values are rounded down to a multiple of **16** (required by the model). Larger resolution uses more VRAM and time.
- **CLI:** Use `--width` and `--height` (defaults 640 and 384). Example: `python run_t2v_cli.py "prompt" --width 832 --height 480`. Same rounding to multiples of 16 applies.
- For **16GB VRAM** or limited GPU memory, keep **640×384**; increase only if you have headroom.

**Frames:** The UI slider goes up to **2500 frames** (rounded to a multiple of 33). The model uses chunks of 33 frames. For very long videos, use the CLI with a higher `--num-frames` (e.g. 1452 for ~60s at 24 FPS). More frames need more VRAM/time.

**Gradio options (top of page):**
- **Use CPU offload (saves VRAM, slower):** when ON, the model stays in RAM and is moved to GPU per step; when OFF, the full model stays on GPU. You can change this anytime; the pipeline switches without reloading.
- **Use 4-bit quantization (saves VRAM; applied when model is first loaded):** when checked, the transformer is loaded in 4-bit on the first run (requires `bitsandbytes`; best supported on Linux). Takes effect only when the model is first loaded in that session; restart the app to change it.

**Dtype / acceleration:** The app uses **bfloat16 or float16** on GPU. For 4-bit in the **CLI**, set the env var before running: `$env:HELIOS_4BIT = "1"` (PowerShell) or `HELIOS_4BIT=1` (bash), and install `pip install bitsandbytes`. 4-bit is best supported on Linux; on Windows it may not be available.

Generated videos are saved under the `outputs/` folder by default. You can override the output directory by setting:

```powershell
$env:HELIOS_OUTPUT_DIR = "my_outputs_folder"
```

### 6. Run and debug without restarting the app (CLI)

To test or debug without starting the Gradio server each time, use the CLI scripts. Each runs one job and exits.

**Text-to-Video**
```powershell
.venv\Scripts\activate
cd F:\Helios
python run_t2v_cli.py "A cat walking on a beach"
python run_t2v_cli.py "Ocean waves" --num-frames 132 --seed 123
python run_t2v_cli.py "Sunset" --width 832 --height 480
```

**Image-to-Video**
```powershell
python run_i2v_cli.py path/to/image.jpg "A wave crashing on the shore"
python run_i2v_cli.py frame.png "Clouds moving" --num-frames 264 --width 640 --height 384
```

**Video-to-Video**
```powershell
python run_v2v_cli.py path/to/input.mp4 "A time-lapse from a train window"
python run_v2v_cli.py clip.mp4 "Dynamic scenery" --num-frames 264 --height 416
```

Options (all three CLIs): `--negative`, `--num-frames`, `--fps`, `--width`, `--height`, `--pyramid-steps S1 S2 S3`, `--guidance-scale`, `--seed`, `--cpu-offload` / `--no-cpu-offload`, `--no-amplify-first-chunk`. Default resolution **640×384**; width and height are rounded to a multiple of 16. Outputs: `outputs/t2v_*.mp4`, `outputs/i2v_*.mp4`, `outputs/v2v_*.mp4`. For **4-bit quantization** with the CLI, set `HELIOS_4BIT=1` (PowerShell: `$env:HELIOS_4BIT = "1"`) before running; requires `bitsandbytes`.

### 7. Recommended settings for 16GB VRAM

Helios-Distilled can run comfortably on a **16GB GPU** if you keep the resolution at **640×384** (default width×height), use CPU offload when needed, and choose conservative frame counts and steps.

- **Width and height:** Keep **640×384** on 16GB. Larger resolutions (e.g. 832×480) use more VRAM; only increase if you have headroom or after enabling CPU offload / 4-bit.
- Use the **"Use CPU offload (saves VRAM, slower)"** checkbox: when ON, the model is kept in RAM and moved to GPU per step (saves VRAM, slower); when OFF, the full model stays on GPU (faster, needs more VRAM).
- **4-bit quantization:** Check **"Use 4-bit quantization"** (Gradio) or set `HELIOS_4BIT=1` (CLI) before the first load to reduce VRAM further; requires `bitsandbytes` (Linux best supported).
- Use a recent CUDA GPU and let the app use half-precision (`bfloat16`/`float16`) as it already does.

#### 7.1 Short clips (~4–6 seconds, higher quality)

Use these as a starting point for 16GB cards:

- **Resolution (width × height):** keep **640×384** (default). Use the Width/Height inputs in the UI or `--width` / `--height` in the CLI; values are rounded to a multiple of 16.
- **Number of frames**: `132` or `165` (4–5 chunks of 33 frames).
- **FPS**: `24`.
- **Pyramid steps**: `[2, 2, 2]` (default) or `[3, 3, 3]` if you can tolerate more time/VRAM.
- **Guidance scale**: between `1.0` and `2.0`.
- **Use CPU offload**: **ON** (saves VRAM; turn OFF if you have enough VRAM for faster runs).
- **Enable Stage 2**: **ON** (better quality; OK for short clips on 16GB).
- **Amplify First Chunk**: **ON**.

These settings are designed to stay within 16GB with some headroom on a typical modern NVIDIA GPU.

#### 7.2 Medium clips (~10–12 seconds, safer VRAM)

For longer clips on 16GB, keep things a bit more conservative:

- **Number of frames**: `264` (8 chunks of 33 frames).
- **FPS**: `24`.
- **Pyramid steps**: `[2, 2, 2]` (do not increase).
- **Guidance scale**: `1.0`.
- **Use CPU offload**: **ON** (critical for 16GB when generating longer clips).
- **Enable Stage 2**: **OFF** (turn off to reduce VRAM/time at this length).
- **Amplify First Chunk**: **ON** (turn OFF only if you are right at the limit).

If you still hit CUDA out-of-memory errors on 16GB:

- First, double-check that **Use CPU offload** is **ON** (or reduce frames / disable Stage 2).
- Then reduce **Number of frames** (e.g. 264 → 198 → 132).
- If Stage 2 is enabled, try disabling it.
- Keep pyramid steps at `[2, 2, 2]` and guidance at `1.0`.


