"""Build three full synchronized playback clips from verified DEV3 outputs."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import subprocess


ARMS = ("Dense", "Q1r2", "Global-P64", "L0", "L+", "Q32", "N32")
SOURCES = ("vbi_0392", "vbi_1013", "vbi_0800")
TOOL_HASHES = {
    "ffmpeg": "a8b8bee4b9048222d76524dfa3248c63ce1f2fad899a335a5e0cd12c71b4eacc",
    "ffprobe": "040709d6ae5123136c2c397e7164e4d2cc909b81e1357263795feb3c09a07d7d",
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LAYOUT = "208_0|624_0|1040_0|0_258|416_258|832_258|1248_258"


def sha(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def make_preview(source, videos, root, tools):
    if tuple(item["arm"] for item in videos) != ARMS:
        raise ValueError(f"{source}: exact seven-arm preview requires fixed ordering")
    codec = [str(tools["ffmpeg"]), "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
    filters = []
    for i, video in enumerate(videos):
        if sha(video["path"]) != video["sha256"]:
            raise ValueError(f"{source}/{video['arm']}: input media hash differs")
        codec.extend(("-threads", "2", "-i", video["path"]))
        label = ARMS[i]
        filters.append(f"[{i}:v]setpts=PTS-STARTPTS,scale=416:232,setsar=1,"
                       f"pad=416:258:0:26:color=0x151C2A,"
                       f"drawtext=fontfile={FONT}:text='{label}':x=10:y=5:fontsize=16:fontcolor=white[v{i}]")
    groups = "".join(f"[v{i}]" for i in range(len(videos)))
    filters.append(f"{groups}xstack=inputs=7:layout={LAYOUT}:fill=0x151C2A:shortest=1[all]")
    path = root / f"{source}_synchronized.mp4"
    codec.extend(("-filter_complex_threads", "2", "-filter_complex", ";".join(filters), "-map", "[all]",
                  "-an", "-frames:v", "81", "-r", "16", "-c:v", "libx264", "-threads", "2",
                  "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)))
    subprocess.run(codec, check=True)
    received = subprocess.run([str(tools["ffprobe"]), "-v", "error", "-select_streams", "v:0", "-count_frames",
                               "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
                               "-of", "json", str(path)], check=True, capture_output=True, text=True)
    stream, = json.loads(received.stdout)["streams"]
    if (stream["width"], stream["height"], stream["avg_frame_rate"], int(stream["nb_read_frames"])) != (1664, 516, "16/1", 81):
        raise ValueError(f"{source}: synchronized output has an unexpected frame grid")
    return {"source": source, "file": str(path), "sha256": sha(path), "input_videos": videos, **stream}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--playback", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--ffmpeg-dir", required=True, type=Path)
    args = cli.parse_args()
    tools = {name: args.ffmpeg_dir / name for name in TOOL_HASHES}
    if any(sha(tools[name]) != digest for name, digest in TOOL_HASHES.items()):
        raise ValueError("The bound FFmpeg toolchain changed")
    manifest = json.loads(args.playback.read_bytes())
    provided = tuple(row["source"] for row in manifest["sources"])
    if not provided or provided != tuple(source for source in SOURCES if source in provided):
        raise ValueError("Preview source order differs")
    args.output.mkdir(exist_ok=False, parents=True)
    result = [make_preview(row["source"], row["videos"], args.output, tools) for row in manifest["sources"]]
    with (args.output / "PREVIEW_MANIFEST.json").open("x", encoding="utf-8") as handle:
        json.dump({"source": str(args.playback), "previews": result, "status": "PENDING_HUMAN_REVIEW"},
                  handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()

