"""Live detection + simultaneous video recording on the Pi.
Run (needs video_detect.py in the same folder):
    python3 live_detect.py            # runs until Ctrl+C
    python3 live_detect.py 60         # stops after 60 s
Outputs in runs/<time>/ : video.h264 (full 1080p recording), detections.csv, annotated frames
"""
import sys, os, time, csv
import cv2
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from video_detect import detect          # reuses the same ONNX inference code

CONF = 0.40
MIN_GAP = 1.0        # seconds between saved images of detections
DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else None

run = time.strftime("runs/%Y%m%d_%H%M%S")
os.makedirs(run, exist_ok=True)

cam = Picamera2()
cfg = cam.create_video_configuration(
    main={"size": (1920, 1080), "format": "RGB888"},      # recorded
    lores={"size": (960, 540), "format": "YUV420"},       # analysed
    controls={"FrameRate": 30})
cam.configure(cfg)
cam.start_recording(H264Encoder(bitrate=8_000_000), FileOutput(os.path.join(run, "video.h264")))
t_start = time.time()
print("Recording + detecting. Ctrl+C to stop. Output:", run)

f = open(os.path.join(run, "detections.csv"), "w", newline="")
wr = csv.writer(f); wr.writerow(["time", "t_since_start_s", "class", "conf", "x1", "y1", "x2", "y2", "image"])
last_save, n, t_fps = 0, 0, time.time()
try:
    while DURATION is None or time.time() - t_start < DURATION:
        yuv = cam.capture_array("lores")          # newest frame, older ones are dropped
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV420p2BGR)
        dets = detect(frame, CONF)
        n += 1
        now = time.time()
        faults = [d for d in dets if "non" not in d[0].lower()]
        if faults and now - last_save >= MIN_GAP:
            last_save = now
            vis = frame.copy()
            img = f"det_{int(now - t_start):05d}s.jpg"
            for c, s, x1, y1, x2, y2 in dets:
                col = (0, 200, 0) if "non" in c.lower() else (0, 0, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
                cv2.putText(vis, f"{c} {s:.2f}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(run, img), vis)
            for c, s, x1, y1, x2, y2 in faults:
                wr.writerow([time.strftime("%H:%M:%S"), round(now - t_start, 1), c, round(s, 3), x1, y1, x2, y2, img])
                print(f"[{now - t_start:6.1f}s] FAULT {c} {s:.2f}")
            f.flush()
        if n % 20 == 0:
            print(f"{20 / (time.time() - t_fps):.1f} fps")
            t_fps = time.time()
except KeyboardInterrupt:
    pass
finally:
    cam.stop_recording(); cam.close(); f.close()
    print("Stopped. Saved in", run)
