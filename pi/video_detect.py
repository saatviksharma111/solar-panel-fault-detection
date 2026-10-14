"""Record video with the Pi camera, then extract frames and run the ONNX model on them.
Usage:
  python3 video_detect.py record 30        # record 30 s -> videos/flight_<time>.mp4, then process it
  python3 video_detect.py process videos/x.mp4   # process an existing video
Optional: add  --every 0.5  (seconds between sampled frames) and  --conf 0.40
"""
import sys, os, time, csv, ast, subprocess, argparse
import cv2, numpy as np, onnxruntime as ort

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "best_yolo26.onnx")
SIZE = 640

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
names = ast.literal_eval(sess.get_modelmeta().custom_metadata_map.get("names", "{}"))
inp = sess.get_inputs()[0].name

def letterbox(img):
    h, w = img.shape[:2]
    r = SIZE / max(h, w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    px, py = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, r, px, py

def detect(img, conf=0.40, iou=0.45):
    """returns list of (class_name, score, x1, y1, x2, y2) in original image pixels"""
    lb, r, px, py = letterbox(img)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    out = sess.run(None, {inp: x})[0]
    boxes, scores, cls = [], [], []
    if out.ndim == 3 and out.shape[-1] == 6:            # end-to-end (1,N,6): x1,y1,x2,y2,score,cls
        for x1, y1, x2, y2, s, c in out[0]:
            if s >= conf:
                boxes.append([x1, y1, x2 - x1, y2 - y1]); scores.append(float(s)); cls.append(int(c))
        keep = range(len(boxes))
    else:                                               # raw head (1,4+nc,N): cx,cy,w,h,scores...
        p = out[0].T
        sc = p[:, 4:]
        c = sc.argmax(1); s = sc.max(1)
        m = s >= conf
        p, c, s = p[m], c[m], s[m]
        for (cx, cy, w, h), ci, si in zip(p[:, :4], c, s):
            boxes.append([cx - w / 2, cy - h / 2, w, h]); scores.append(float(si)); cls.append(int(ci))
        keep = []
        for ci in set(cls):
            idx = [i for i in range(len(cls)) if cls[i] == ci]
            k = cv2.dnn.NMSBoxes([boxes[i] for i in idx], [scores[i] for i in idx], conf, iou)
            keep += [idx[int(j)] for j in np.array(k).flatten()]
    res = []
    for i in keep:
        x, y, w, h = boxes[i]
        x1, y1 = (x - px) / r, (y - py) / r
        x2, y2 = (x + w - px) / r, (y + h - py) / r
        res.append((names.get(cls[i], str(cls[i])), scores[i], int(x1), int(y1), int(x2), int(y2)))
    return res

def record(seconds):
    os.makedirs("videos", exist_ok=True)
    path = time.strftime("videos/flight_%Y%m%d_%H%M%S.mp4")
    cmd = ["rpicam-vid", "-t", str(int(seconds * 1000)), "--width", "1920", "--height", "1080",
           "--framerate", "30", "--codec", "libav", "--libav-format", "mp4", "-n", "-o", path]
    print("Recording", seconds, "s ...")
    subprocess.run(cmd, check=True)
    print("Saved", path)
    return path

def process(path, every=0.5, conf=0.40):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(fps * every)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir = os.path.splitext(os.path.basename(path))[0] + "_results"
    os.makedirs(out_dir, exist_ok=True)
    f = open(os.path.join(out_dir, "detections.csv"), "w", newline="")
    wr = csv.writer(f); wr.writerow(["video_time_s", "frame", "class", "conf", "x1", "y1", "x2", "y2", "image"])
    i = n_proc = n_det = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            dets = detect(frame, conf)
            n_proc += 1
            if dets:
                vis = frame.copy()
                img_name = f"frame_{i:06d}.jpg"
                for c, s, x1, y1, x2, y2 in dets:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255) if "non" not in c.lower() else (0, 200, 0), 3)
                    cv2.putText(vis, f"{c} {s:.2f}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    wr.writerow([round(i / fps, 2), i, c, round(s, 3), x1, y1, x2, y2, img_name])
                    n_det += 1
                cv2.imwrite(os.path.join(out_dir, img_name), vis)
        i += 1
    f.close(); cap.release()
    print(f"Done: {n_proc} frames analysed out of {total}, {n_det} detections, {time.time()-t0:.0f}s")
    print("Results in", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "process"])
    ap.add_argument("arg")
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.40)
    a = ap.parse_args()
    p = record(float(a.arg)) if a.mode == "record" else a.arg
    process(p, a.every, a.conf)
