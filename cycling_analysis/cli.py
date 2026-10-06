import argparse
from pathlib import Path
from .pipeline import analyze_pose

def main(argv=None):
    parser = argparse.ArgumentParser(prog="cycling-analyzer")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze-pose", "analyze"):
        p = commands.add_parser(name)
        p.add_argument("input", type=Path)
        p.add_argument("--output", type=Path)
        p.add_argument("--fps", type=float, help="Pose mode: sampling FPS override")
        if name == "analyze": p.add_argument("--skip-pose", type=Path, help="Use an existing X3D.npy for this video")
    args = parser.parse_args(argv)
    if args.command == "analyze-pose":
        result = analyze_pose(args.input, fps=args.fps or 60.0, output_dir=args.output)
        out = args.output or args.input.parent / f"{args.input.stem}_analysis"
    elif args.skip_pose:
        from pipeline.video_pipeline import video_fps
        fps = args.fps or video_fps(args.input)
        out = args.output or Path("runtime/cli") / args.input.stem
        result = analyze_pose(args.skip_pose, fps=fps, output_dir=out)
    else:
        from pipeline.video_pipeline import analyze_video
        out = args.output or Path("runtime/cli") / args.input.stem
        result = analyze_video(args.input, output_dir=out)
    print(f"{out / 'results.json'}: {result['cycles']['count']} cycles")

if __name__ == "__main__": main()
