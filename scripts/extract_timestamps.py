import cv2
from pathlib import Path
import pandas as pd

def extract_thumbnails(video_path, output_dir, interval=0.1, zip_output=True):
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    return

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  duration = total_frames / fps
  frame_interval = int(fps * interval)

  saved_thumbnails = []

  for i in range(0, total_frames, frame_interval):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
      continue
    timestamp = i/fps
    file_name = output_path / f"frame_{i:03d}_{int(timestamp*1000)}ms.jpg"
    cv2.imwrite(str(file_name), frame)
    saved_thumbnails.append((timestamp, str(file_name)))

  cap.release()
  print(f"Finished extracting thumbnails from {video_path}")
  return saved_thumbnails

def match_timestamps(df, header_timestamps, start_time):
  video_start = pd.to_datetime(start_time)
  video_iso = [video_start + pd.Timedelta(milliseconds=ms) for ms in header_timestamps]
  matched = []
  for vid_time, vid_iso in zip(header_timestamps, video_iso):
    closest_idx = (df['isoTimestamp'] - vid_iso).abs().idxmin()
    matched.append({
      'video_ms': vid_time,
      'video_isoTimestamp': vid_iso,
      'matched_sensor_isoTimestamp': df.loc[closest_idx, 'isoTimestamp'],
      'sensor_relative_sec': (df.loc[closest_idx, 'isoTimestamp'] - df['isoTimestamp'].iloc[0]).total_seconds()
    })
    return pd.DataFrame(matched)

def main():
  video_path = "../data/videos/2-21-2025.MOV" # replace with video path
  output_dir = "thumbnails_02_21" # replace with output directory
  interval = 0.1
  date = "2025-02-21" # replace with date
  start_time = "2025-02-21 19:50:09" # replace with video start time

  df = pd.read_csv("../data/raw/back.csv") # replace with sensor data file path
  df['isoTimestamp'] = pd.to_datetime(df['isoTimestamp'])
  df = df.sort_values(by='isoTimestamp')
  df = df[df['isoTimestamp'].dt.date == pd.to_datetime(date).date()]
  df['relative_seconds'] = (df['isoTimestamp'] - df['isoTimestamp'].iloc[0]).dt.total_seconds()

  thumbnails = extract_thumbnails(video_path, output_dir, interval)
  print(f"Extracted {len(thumbnails)} thumbnails")
  
  header_ms = input("Enter header timestamps in ms (comma-separated): ")
  header_ms = [int(x) for x in header_ms.split(",")]
  matched_timestamps = match_timestamps(df, header_ms, start_time)
  print(f"Matched Timestamps: {matched_timestamps}")
  print(f"isoTmestamp: {matched_timestamps['matched_isoTimestamp']}")

main()