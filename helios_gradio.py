import gc
import os
import sys
from typing import List, Optional

# Windows: avoid noisy ConnectionResetError when browser tab closes
if sys.platform == "win32":
    import asyncio
    from asyncio import proactor_events

    class _PatchedProactorPipeTransport(proactor_events._ProactorBasePipeTransport):
        def _call_connection_lost(self, exc):
            try:
                super()._call_connection_lost(exc)
            except ConnectionResetError:
                pass

    proactor_events._ProactorBasePipeTransport = _PatchedProactorPipeTransport  # type: ignore[assignment]

import gradio as gr
import torch
from diffusers import AutoModel, HeliosPyramidPipeline
from diffusers.utils import export_to_video, load_image


# --------- Basic configuration (matches HF example) ----------

# Always use the local model directory under ./models/
MODEL_ID = "./models/Helios-Distilled"

# Output folder for generated videos
OUTPUT_DIR = os.environ.get("HELIOS_OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _get_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        return "cuda", torch.float16
    return "cpu", torch.float32


DEVICE, TORCH_DTYPE = _get_device_and_dtype()

print(f"Using device: {DEVICE}, dtype: {TORCH_DTYPE}")


_PIPELINE: Optional[HeliosPyramidPipeline] = None
_PIPELINE_CPU_OFFLOAD: Optional[bool] = None
_PIPELINE_USE_4BIT: Optional[bool] = None


def load_pipeline(enable_cpu_offload: bool, use_4bit: bool) -> HeliosPyramidPipeline:
    """
    Load Helios-Distilled, with optional CPU offload and 4-bit quantization.
    This is a light wrapper around the official Diffusers example.
    """
    global _PIPELINE, _PIPELINE_CPU_OFFLOAD, _PIPELINE_USE_4BIT

    # Reuse existing pipeline if settings match
    if _PIPELINE is not None and _PIPELINE_CPU_OFFLOAD == enable_cpu_offload and _PIPELINE_USE_4BIT == use_4bit:
        return _PIPELINE

    print(f"Loading Helios-Distilled from: {MODEL_ID}")

    vae = AutoModel.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    quantization_config = None
    if use_4bit and DEVICE == "cuda":
        try:
            from diffusers import PipelineQuantizationConfig

            quantization_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": TORCH_DTYPE,
                },
                components_to_quantize=["transformer"],
            )
            print("Loading transformer in 4-bit.")
        except Exception as e:  # pragma: no cover - optional path
            print(f"4-bit loading failed ({e}). Using full precision.")
            quantization_config = None

    pipe = HeliosPyramidPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        torch_dtype=TORCH_DTYPE,
        quantization_config=quantization_config,
    )

    if enable_cpu_offload and DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(DEVICE)

    _PIPELINE = pipe
    _PIPELINE_CPU_OFFLOAD = enable_cpu_offload
    _PIPELINE_USE_4BIT = use_4bit
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return pipe


def _cleanup() -> None:
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


DEFAULT_NEGATIVE_PROMPT = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality,
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured,
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
""".strip()


def _round_num_frames(num_frames: int) -> int:
    if num_frames <= 0:
        return 33
    if num_frames % 33 == 0:
        return num_frames
    return ((num_frames // 33) + 1) * 33


def _round_to_16(x: int) -> int:
    """
    Helios requires width and height to be divisible by 16.
    """
    x = int(x)
    if x <= 0:
        return 16
    return max(16, (x // 16) * 16)


def _save_video_from_frames(frames: List, fps: int, prefix: str) -> str:
    filename = f"{prefix}_{torch.randint(0, 10**8, (1,)).item()}.mp4"
    out_path = os.path.join(OUTPUT_DIR, filename)
    export_to_video(frames, out_path, fps=fps)
    return out_path


# --------- Inference functions (T2V / I2V) ----------


def run_t2v(
    enable_cpu_offload: bool,
    use_4bit: bool,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    pyramid_steps_1: int,
    pyramid_steps_2: int,
    pyramid_steps_3: int,
    guidance_scale: float,
    is_enable_stage2: bool,
    is_amplify_first_chunk: bool,
    seed: int,
) -> str:
    """
    Core Text-to-Video runner used by both the CLI and Gradio UI.
    The is_enable_stage2 flag is accepted for API compatibility but ignored,
    since HeliosPyramidPipeline does not use it in diffusers.
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt must not be empty.")

    pipe = load_pipeline(enable_cpu_offload=enable_cpu_offload, use_4bit=use_4bit)

    num_frames = _round_num_frames(int(num_frames))
    width = _round_to_16(width)
    height = _round_to_16(height)
    steps = [int(pyramid_steps_1), int(pyramid_steps_2), int(pyramid_steps_3)]
    generator = torch.Generator(DEVICE).manual_seed(int(seed))

    result = pipe(
        prompt=prompt.strip(),
        negative_prompt=(negative_prompt or DEFAULT_NEGATIVE_PROMPT),
        width=width,
        height=height,
        num_frames=num_frames,
        pyramid_num_inference_steps_list=steps,
        guidance_scale=float(guidance_scale),
        is_amplify_first_chunk=bool(is_amplify_first_chunk),
        generator=generator,
    ).frames[0]

    video_path = _save_video_from_frames(result, fps=int(fps), prefix="t2v")
    _cleanup()
    return video_path


def run_i2v(
    enable_cpu_offload: bool,
    use_4bit: bool,
    prompt: str,
    image,  # CLI passes a PIL.Image; Gradio passes a filepath string
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    pyramid_steps_1: int,
    pyramid_steps_2: int,
    pyramid_steps_3: int,
    guidance_scale: float,
    is_enable_stage2: bool,
    is_amplify_first_chunk: bool,
    seed: int,
) -> str:
    """
    Core Image-to-Video runner used by both the CLI and Gradio UI.
    Accepts either a PIL.Image or a filepath string for the image argument.
    """
    if image is None:
        raise gr.Error("Please provide an input image.")

    pipe = load_pipeline(enable_cpu_offload=enable_cpu_offload, use_4bit=use_4bit)

    num_frames = _round_num_frames(int(num_frames))
    width = _round_to_16(width)
    height = _round_to_16(height)
    steps = [int(pyramid_steps_1), int(pyramid_steps_2), int(pyramid_steps_3)]
    generator = torch.Generator(DEVICE).manual_seed(int(seed))

    # Normalize image input
    if isinstance(image, str):
        img = load_image(image).resize((width, height))
    else:
        img = image.convert("RGB").resize((width, height))

    result = pipe(
        prompt=prompt.strip() if prompt else "",
        negative_prompt=(negative_prompt or DEFAULT_NEGATIVE_PROMPT),
        image=img,
        width=width,
        height=height,
        num_frames=num_frames,
        pyramid_num_inference_steps_list=steps,
        guidance_scale=float(guidance_scale),
        is_amplify_first_chunk=bool(is_amplify_first_chunk),
        generator=generator,
    ).frames[0]

    video_path = _save_video_from_frames(result, fps=int(fps), prefix="i2v")
    _cleanup()
    return video_path


def ui_run_t2v(
    enable_cpu_offload: bool,
    use_4bit: bool,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    guidance_scale: float,
    pyramid_step_1: int,
    pyramid_step_2: int,
    pyramid_step_3: int,
    seed: int,
) -> str:
    return run_t2v(
        enable_cpu_offload=enable_cpu_offload,
        use_4bit=use_4bit,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        pyramid_steps_1=pyramid_step_1,
        pyramid_steps_2=pyramid_step_2,
        pyramid_steps_3=pyramid_step_3,
        guidance_scale=guidance_scale,
        is_enable_stage2=True,
        is_amplify_first_chunk=True,
        seed=seed,
    )


def ui_run_i2v(
    enable_cpu_offload: bool,
    use_4bit: bool,
    image_source: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    guidance_scale: float,
    pyramid_step_1: int,
    pyramid_step_2: int,
    pyramid_step_3: int,
    seed: int,
) -> str:
    if not image_source:
        raise gr.Error("Please provide an input image.")

    return run_i2v(
        enable_cpu_offload=enable_cpu_offload,
        use_4bit=use_4bit,
        prompt=prompt,
        image=image_source,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        pyramid_steps_1=pyramid_step_1,
        pyramid_steps_2=pyramid_step_2,
        pyramid_steps_3=pyramid_step_3,
        guidance_scale=guidance_scale,
        is_enable_stage2=True,
        is_amplify_first_chunk=True,
        seed=seed,
    )


# --------- Gradio UI ----------


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Helios-Distilled (Diffusers)") as demo:
        gr.Markdown(
            "## Helios-Distilled Video Generation\n"
            "Text→Video and Image→Video using "
            "[BestWishYsh/Helios-Distilled](https://huggingface.co/BestWishYsh/Helios-Distilled).\n\n"
            "Settings are kept close to the official Diffusers example."
        )

        with gr.Tab("Text → Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2v_prompt = gr.Textbox(
                        label="Prompt",
                        lines=5,
                        placeholder="Describe the video you want to generate...",
                    )
                    t2v_negative = gr.Textbox(
                        label="Negative Prompt",
                        value=DEFAULT_NEGATIVE_PROMPT,
                        lines=3,
                    )
                    with gr.Row():
                        t2v_width = gr.Number(
                            label="Width",
                            value=640,
                            minimum=256,
                            maximum=1280,
                            step=16,
                            precision=0,
                        )
                        t2v_height = gr.Number(
                            label="Height",
                            value=384,
                            minimum=256,
                            maximum=720,
                            step=16,
                            precision=0,
                        )
                    t2v_num_frames = gr.Slider(
                        label="Number of Frames (multiple of 33 recommended)",
                        minimum=33,
                        maximum=726,
                        step=33,
                        value=240,
                    )
                    t2v_fps = gr.Slider(
                        label="FPS",
                        minimum=8,
                        maximum=30,
                        step=1,
                        value=24,
                    )
                    with gr.Row():
                        t2v_step1 = gr.Slider(
                            label="Pyramid Steps Level 1",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=2,
                        )
                        t2v_step2 = gr.Slider(
                            label="Pyramid Steps Level 2",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=2,
                        )
                        t2v_step3 = gr.Slider(
                            label="Pyramid Steps Level 3",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=2,
                        )
                    t2v_guidance = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.5,
                        maximum=10.0,
                        step=0.1,
                        value=1.0,
                    )
                    t2v_cpu_offload = gr.Checkbox(
                        label="Use CPU offload (saves VRAM, slower)",
                        value=True,
                    )
                    t2v_use_4bit = gr.Checkbox(
                        label="Use 4-bit quantization (experimental, needs bitsandbytes)",
                        value=False,
                    )
                    t2v_seed = gr.Number(label="Seed", value=42, precision=0)
                    t2v_button = gr.Button("Generate T2V Video")
                with gr.Column(scale=1):
                    t2v_output = gr.Video(label="Generated Video")

            t2v_button.click(
                fn=ui_run_t2v,
                inputs=[
                    t2v_cpu_offload,
                    t2v_use_4bit,
                    t2v_prompt,
                    t2v_negative,
                    t2v_width,
                    t2v_height,
                    t2v_num_frames,
                    t2v_fps,
                    t2v_guidance,
                    t2v_step1,
                    t2v_step2,
                    t2v_step3,
                    t2v_seed,
                ],
                outputs=t2v_output,
            )

        with gr.Tab("Image → Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2v_image = gr.File(
                        label="Input Image (local file, e.g. .png/.jpg)",
                        file_count="single",
                        type="filepath",
                    )
                    i2v_prompt = gr.Textbox(
                        label="Prompt (optional, can refine motion/style)",
                        lines=4,
                    )
                    i2v_negative = gr.Textbox(
                        label="Negative Prompt",
                        value=DEFAULT_NEGATIVE_PROMPT,
                        lines=3,
                    )
                    with gr.Row():
                        i2v_width = gr.Number(
                            label="Width",
                            value=640,
                            minimum=256,
                            maximum=1280,
                            step=16,
                            precision=0,
                        )
                        i2v_height = gr.Number(
                            label="Height",
                            value=384,
                            minimum=256,
                            maximum=720,
                            step=16,
                            precision=0,
                        )
                    i2v_num_frames = gr.Slider(
                        label="Number of Frames (multiple of 33 recommended)",
                        minimum=33,
                        maximum=726,
                        step=33,
                        value=240,
                    )
                    i2v_fps = gr.Slider(
                        label="FPS",
                        minimum=8,
                        maximum=30,
                        step=1,
                        value=24,
                    )
                    with gr.Row():
                        i2v_step1 = gr.Slider(
                            label="Pyramid Steps Level 1",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=2,
                        )
                        i2v_step2 = gr.Slider(
                            label="Pyramid Steps Level 2",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=2,
                        )
                        i2v_step3 = gr.Slider(
                            label="Pyramid Steps Level 3",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=2,
                        )
                    i2v_guidance = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.5,
                        maximum=10.0,
                        step=0.1,
                        value=1.0,
                    )
                    i2v_cpu_offload = gr.Checkbox(
                        label="Use CPU offload (saves VRAM, slower)",
                        value=True,
                    )
                    i2v_use_4bit = gr.Checkbox(
                        label="Use 4-bit quantization (experimental, needs bitsandbytes)",
                        value=False,
                    )
                    i2v_seed = gr.Number(label="Seed", value=42, precision=0)
                    i2v_button = gr.Button("Generate I2V Video")
                with gr.Column(scale=1):
                    i2v_output = gr.Video(label="Generated Video")

            i2v_button.click(
                fn=ui_run_i2v,
                inputs=[
                    i2v_cpu_offload,
                    i2v_use_4bit,
                    i2v_image,
                    i2v_prompt,
                    i2v_negative,
                    i2v_width,
                    i2v_height,
                    i2v_num_frames,
                    i2v_fps,
                    i2v_guidance,
                    i2v_step1,
                    i2v_step2,
                    i2v_step3,
                    i2v_seed,
                ],
                outputs=i2v_output,
            )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.queue().launch(server_name="127.0.0.1", server_port=7860)

