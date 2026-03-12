"""
Run Text-to-Video once from the command line. Use this to test and debug T2V
without starting the Gradio app. Run repeatedly with different prompts or
options; no need to restart a long-running process.

Example:
    .venv\\Scripts\\activate
    python run_t2v_cli.py "A cat walking on a beach"
    python run_t2v_cli.py "Ocean waves" --num-frames 132 --seed 123
"""
from __future__ import annotations

import argparse
import os
import sys

# Run from project root so imports and model path resolve correctly
if os.path.dirname(os.path.abspath(__file__)) != os.path.abspath(os.getcwd()):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

from helios_gradio import (
    DEFAULT_NEGATIVE_PROMPT,
    load_pipeline,
    run_t2v,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Run Helios-Distilled Text-to-Video once.")
    p.add_argument("prompt", nargs="?", default="", help="Text prompt for the video")
    p.add_argument("--negative", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt")
    p.add_argument("--num-frames", type=int, default=132, help="Target number of frames (rounded to multiple of 33)")
    p.add_argument("--fps", type=int, default=24, help="Output FPS")
    p.add_argument("--pyramid-steps", type=int, nargs=3, default=[2, 2, 2], metavar=("S1", "S2", "S3"))
    p.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--width", type=int, default=640, help="Output video width (divisible by 16)")
    p.add_argument("--height", type=int, default=384, help="Output video height (divisible by 16)")
    p.add_argument("--cpu-offload", action="store_true", default=True, help="Use CPU offload to save VRAM (default: True)")
    p.add_argument("--no-cpu-offload", action="store_false", dest="cpu_offload")
    p.add_argument("--amplify-first-chunk", action="store_true", default=True, help="Amplify first chunk (default: True)")
    p.add_argument("--no-amplify-first-chunk", action="store_false", dest="amplify_first_chunk")
    args = p.parse_args()

    if not (args.prompt or args.prompt.strip()):
        # Allow reading prompt from stdin for long prompts
        args.prompt = sys.stdin.read().strip()
    if not args.prompt.strip():
        p.error("Provide a prompt as argument or via stdin.")

    use_4bit = os.environ.get("HELIOS_4BIT", "").strip().lower() in ("1", "true", "yes")
    print("Loading pipeline...")
    load_pipeline(enable_cpu_offload=args.cpu_offload, use_4bit=use_4bit)
    print("Running Text-to-Video...")
    out_path = run_t2v(
        enable_cpu_offload=args.cpu_offload,
        use_4bit=use_4bit,
        prompt=args.prompt.strip(),
        negative_prompt=args.negative or DEFAULT_NEGATIVE_PROMPT,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        pyramid_steps_1=args.pyramid_steps[0],
        pyramid_steps_2=args.pyramid_steps[1],
        pyramid_steps_3=args.pyramid_steps[2],
        guidance_scale=args.guidance_scale,
        is_enable_stage2=True,
        is_amplify_first_chunk=args.amplify_first_chunk,
        seed=args.seed,
    )
    print("Done.")
    print("Output:", os.path.abspath(out_path))


if __name__ == "__main__":
    main()
