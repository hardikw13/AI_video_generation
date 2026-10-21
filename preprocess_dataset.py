import cv2
import csv
from pathlib import Path

root = Path("physics_dataset")
input_dir = root / "videos"
output_dir = root / "training_videos"

output_dir.mkdir(exist_ok=True)

rows = list(csv.DictReader(open(
    root / "metadata_49frames.csv",
    encoding="utf-8"
)))

success = 0
failed = 0

for i, row in enumerate(rows, 1):
    src = root / row["video"]
    out = output_dir / src.name

    cap = cv2.VideoCapture(str(src))

    if not cap.isOpened():
        print(f"[FAILED] {src.name}")
        failed += 1
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while len(frames) < 49:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) < 49:
        print(f"[SKIP] {src.name} - only {len(frames)} frames")
        failed += 1
        continue

    frames = frames[:49]

    h, w = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out),
        fourcc,
        24.0,
        (640, 360)
    )

    for frame in frames:
        frame = cv2.resize(frame, (640, 360))
        writer.write(frame)

    writer.release()

    success += 1
    print(f"[{i}/{len(rows)}] {src.name}")

print()
print("================================")
print("Preprocessing complete")
print("Successful:", success)
print("Failed:", failed)
print("================================")
