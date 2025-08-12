# mini_coram_demo.py (patched)
import argparse, os, json, math, uuid
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import open_clip
import faiss

# ---------------- IoU tracker (demo) ----------------
class IoUTracker:
    def __init__(self, iou_thr=0.3, max_age=30):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.tracks = {}   # tid -> dict
        self.next_tid = 1

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        xi1, yi1 = max(ax1, bx1), max(ay1, by1)
        xi2, yi2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, xi2-xi1), max(0, yi2-yi1)
        inter = iw*ih
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter/ua if ua>0 else 0.0

    def update(self, frame_idx, dets):
        assigned = set()
        tids = list(self.tracks.keys())
        for tid in tids:
            best_iou, best_j = 0.0, -1
            for j, d in enumerate(dets):
                if j in assigned: continue
                iou = self.iou(self.tracks[tid]['bbox'], d[:4])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= self.iou_thr:
                d = dets[best_j]; assigned.add(best_j)
                self._append(tid, frame_idx, d)

        for j, d in enumerate(dets):
            if j in assigned: continue
            tid = self.next_tid; self.next_tid += 1
            self.tracks[tid] = dict(
                bbox=d[:4], last_frame=frame_idx, frames=[frame_idx],
                boxes=[d[:4]], areas=[(d[2]-d[0])*(d[3]-d[1])],
                crops=[], times=[]
            )

        # finalize by age (not strictly needed now that we flush on EOF)
        to_finalize = []
        for tid, t in list(self.tracks.items()):
            if t['last_frame'] < frame_idx - self.max_age:
                to_finalize.append(tid)
        return to_finalize

    def _append(self, tid, frame_idx, d):
        t = self.tracks[tid]
        t['bbox'] = d[:4]; t['last_frame'] = frame_idx
        t['frames'].append(frame_idx); t['boxes'].append(d[:4])
        t['areas'].append((d[2]-d[0])*(d[3]-d[1]))

    def harvest_track(self, tid):
        return self.tracks.pop(tid, None)

    def all_active_ids(self):
        return list(self.tracks.keys())

# ---------------- Simple attributes ----------------
BASIC_COLORS = {
    'red':   (0, 10),  'orange': (10, 22), 'yellow': (22, 35),
    'green': (35, 85), 'cyan': (85, 100),  'blue':   (100, 135),
    'purple':(135, 160),'pink': (160, 175)
}
def dominant_color_hsv(img_bgr):
    if img_bgr is None or img_bgr.size == 0: return None
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[...,0].astype(np.float32) * 2.0  # 0..360
    s = hsv[...,1].astype(np.float32)/255.0
    v = hsv[...,2].astype(np.float32)/255.0
    mask = (s > 0.25) & (v > 0.2)
    if not np.any(mask): return None
    h_sel = h[mask]
    hrad = np.deg2rad(h_sel)
    ang = math.degrees(np.arctan2(np.mean(np.sin(hrad)), np.mean(np.cos(hrad))))
    if ang < 0: ang += 360
    for name,(lo,hi) in BASIC_COLORS.items():
        if lo <= ang < hi: return name
    if ang >= 175 or ang < 0: return 'red'
    return 'unknown'

def tinted_windows_heuristic(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0: return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    h = gray.shape[0]
    top = gray[:max(1,h//3), :]
    m_top = float(np.mean(top)); m_all = float(np.mean(gray))
    return (m_top < 0.35) and (m_top + 0.12 < m_all)

# ---------------- OpenCLIP ----------------
def load_openclip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='openai'
    )
    tok = open_clip.get_tokenizer('ViT-B-16')
    model = model.to(device).eval()
    return model, preprocess, tok

@torch.no_grad()
def clip_image_embed(model, preprocess, device, bgr_imgs):
    from PIL import Image
    imgs = []
    for im in bgr_imgs:
        if im is None or im.size == 0: continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs.append(preprocess(Image.fromarray(rgb)))
    if not imgs: return None
    batch = torch.stack(imgs).to(device)
    feats = model.encode_image(batch)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.mean(dim=0).cpu().numpy()

@torch.no_grad()
def clip_text_embed(model, tok, device, text):
    toks = tok([text]).to(device)
    feats = model.encode_text(toks)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.squeeze(0).cpu().numpy()

# ---------------- FAISS helpers ----------------
def build_faiss(dim): return faiss.IndexFlatIP(dim)
def save_index(index, path): faiss.write_index(index, path)
def load_index(path): return faiss.read_index(path)

# ---------------- Indexing ----------------
VEHICLE_COCO_IDS = {2,3,5,7}   # car, motorcycle, bus, truck
MIN_BOX_AREA = 24 * 24         # very small crops are noisy

def save_thumb(out_dir, crop):
    thumbs = Path(out_dir)/"thumbs"; thumbs.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex[:8]}.jpg"
    path = str(thumbs/name)
    cv2.imwrite(path, crop)
    return path

def finalize_track_record(track, video_stem, fps, clip_model, clip_pre, device, out_dir):
    if track is None or len(track['crops']) == 0: return None, None
    best_crop = max(track['crops'], key=lambda c: c.size)
    color = dominant_color_hsv(best_crop) or "unknown"
    tinted = tinted_windows_heuristic(best_crop)
    vec = clip_image_embed(clip_model, clip_pre, device, track['crops'])
    if vec is None: return None, None
    payload = {
        "id": f"track_{uuid.uuid4().hex[:8]}",
        "camera_id": video_stem,
        "t_start": float(track['times'][0]),
        "t_end": float(track['times'][-1]),
        "color": color,
        "make": None, "model": None,
        "type": "vehicle",
        "tinted_windows": bool(tinted),
        "logos": [],
        "ocr_tokens": [],
        "thumb_path": None,  # fill after write
        "clip_ref": f"{video_stem}#t={track['times'][0]:.2f},{track['times'][-1]:.2f}"
    }
    payload["thumb_path"] = save_thumb(out_dir, best_crop)
    return vec.astype('float32'), payload

def index_video(args):
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo = YOLO('yolov8n.pt')
    clip_model, clip_pre, clip_tok = load_openclip(device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    tracker = IoUTracker(iou_thr=0.3, max_age=int(1.0*fps))  # allow ~1s gaps

    payloads, vectors = [], []
    frame_idx = 0
    sample_every = max(1, int(fps//2))  # ~2 fps sampling

    while True:
        ok, frame = cap.read()
        if not ok: break

        res = yolo.predict(source=frame, imgsz=640, conf=0.20, verbose=False)[0]
        dets = []
        for b in res.boxes:
            cls = int(b.cls.item()); conf = float(b.conf.item())
            if cls in VEHICLE_COCO_IDS:
                x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                x1,y1,x2,y2 = max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)
                if (x2-x1)*(y2-y1) >= MIN_BOX_AREA:
                    dets.append((x1,y1,x2,y2,conf,cls))

        to_finalize = tracker.update(frame_idx, dets)

        # sample crops for active tracks
        for tid, t in tracker.tracks.items():
            if frame_idx % sample_every == 0:
                x1,y1,x2,y2 = map(int, t['bbox'])
                crop = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                if crop.size:
                    t['crops'].append(crop)
                    t['times'].append(frame_idx / fps)

        # finalize tracks that aged out
        for tid in to_finalize:
            rec = tracker.harvest_track(tid)
            vec, payload = finalize_track_record(rec, Path(args.video).stem, fps, clip_model, clip_pre, device, args.out)
            if vec is not None:
                vectors.append(vec); payloads.append(payload)

        frame_idx += 1

    # -------- FLUSH: finalize all remaining active tracks at EOF --------
    for tid in tracker.all_active_ids():
        rec = tracker.harvest_track(tid)
        vec, payload = finalize_track_record(rec, Path(args.video).stem, fps, clip_model, clip_pre, device, args.out)
        if vec is not None:
            vectors.append(vec); payloads.append(payload)

    cap.release()

    # Safety fallback: if no tracks, embed a few whole frames so pipeline can be tested
    if not vectors:
        print("No tracks finalized; falling back to whole-frame embeddings for demo...")
        cap = cv2.VideoCapture(args.video)
        frames, step = [], max(1, int((fps or 25)//2))
        i = 0
        while True:
            ok, f = cap.read()
            if not ok: break
            if i % step == 0: frames.append(f)
            if len(frames) >= 8: break
            i += 1
        cap.release()
        vec = clip_image_embed(clip_model, clip_pre, device, frames)
        if vec is not None:
            vectors.append(vec.astype('float32'))
            payloads.append({
                "id": f"frame_{uuid.uuid4().hex[:8]}",
                "camera_id": Path(args.video).stem,
                "t_start": 0.0, "t_end": 0.0,
                "color": None, "make": None, "model": None,
                "type": "frame",
                "tinted_windows": None,
                "logos": [], "ocr_tokens": [],
                "thumb_path": save_thumb(args.out, frames[0]),
                "clip_ref": f"{args.video}#t=0,0"
            })

    if not vectors:
        print("Still no vectors created. Try a video with visible vehicles or lower detector thresholds.")
        return

    mat = np.vstack(vectors).astype('float32')
    faiss.normalize_L2(mat)
    index = build_faiss(mat.shape[1]); index.add(mat)
    save_index(index, os.path.join(args.out, "index.faiss"))
    with open(os.path.join(args.out, "payloads.json"), "w") as f:
        json.dump(payloads, f, indent=2)
    print(f"Indexed {len(payloads)} items → {args.out}")

# ---------------- Search ----------------
def search_query(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, _, clip_tok = load_openclip(device)
    idx_path = os.path.join(args.out, "index.faiss")
    payloads_path = os.path.join(args.out, "payloads.json")
    if not (os.path.exists(idx_path) and os.path.exists(payloads_path)):
        raise SystemExit("Index not found. Run in indexing mode first.")

    index = load_index(idx_path)
    payloads = json.load(open(payloads_path))
    qvec = clip_text_embed(clip_model, clip_tok, device, args.query).astype('float32').reshape(1, -1)
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, k=min(args.topk, len(payloads)))

    results_dir = Path(args.out)/"results"; results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nTop {len(I[0])} results for: \"{args.query}\"")
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        p = payloads[int(idx)]
        thumb = cv2.imread(p["thumb_path"])
        outp = str(results_dir / f"{rank:02d}_{Path(p['thumb_path']).name}")
        if thumb is not None: cv2.imwrite(outp, thumb)
        print(f"{rank:2d}. score={score:.3f}  color={p['color']}  tinted={p['tinted_windows']}  "
              f"time=[{p['t_start']:.2f},{p['t_end']:.2f}]  clip={p['clip_ref']}  thumb={outp}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, help="path to MP4 to index")
    ap.add_argument("--out", type=str, required=True, help="output dir")
    ap.add_argument("--query", type=str, help="text query (search mode)")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    if args.query:
        search_query(args)
    else:
        if not args.video:
            raise SystemExit("Provide --video to index, or --query to search.")
        index_video(args)

if __name__ == "__main__":
    main()
