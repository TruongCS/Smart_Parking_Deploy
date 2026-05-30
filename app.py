import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import sqlite3
import tempfile
import time
import uuid
import subprocess
from pathlib import Path
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter

SYDNEY_TZ = ZoneInfo("Australia/Sydney")

st.set_page_config(
    page_title="Smart Parking ALPR",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    .plate-text {
        font-family: monospace;
        font-size: 28px;
        font-weight: bold;
        color: #00ff88;
        text-align: center;
        padding: 10px;
        background: #1e2130;
        border-radius: 8px;
        border: 2px solid #00ff88;
        margin: 10px 0;
    }
    .log-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 10px;
        border-bottom: 1px solid #2a3140;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

YOLO_WEIGHTS = "models/yolov26s/yolov26s.pt"
DCE_WEIGHTS  = "models/zero_dce/zero_dce_best.pth"
PARSEQ_CKPT  = "models/parseq_finetune/parseq_finetuned_best.pt"
PARSEQ_REPO  = "/tmp/parseq"
DB_PATH      = "smart_parking.db"
SCALE_FACTOR = 12

def now_sydney():
    return datetime.now(SYDNEY_TZ).strftime("%Y-%m-%d %H:%M:%S")

# ── Database ──────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS parking_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_text TEXT NOT NULL,
        entry_time TEXT NOT NULL,
        exit_time TEXT,
        duration_min REAL DEFAULT 0,
        fee REAL DEFAULT 0,
        rate_per_min REAL DEFAULT 0.5
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS plate_detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_text TEXT NOT NULL,
        confidence REAL DEFAULT 0,
        condition TEXT DEFAULT 'normal',
        ocr_conf REAL DEFAULT 0,
        source TEXT DEFAULT 'image',
        created_at TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()

def register_gate_event(plate, rate=0.50):
    plate = plate.upper().strip()
    if len(plate) < 3:
        return None, None
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT id, entry_time FROM parking_sessions
                 WHERE plate_text=? AND exit_time IS NULL
                 ORDER BY entry_time DESC LIMIT 1""", (plate,))
    open_s = c.fetchone()
    now = now_sydney()
    if open_s is None:
        c.execute("INSERT INTO parking_sessions (plate_text,entry_time,rate_per_min) VALUES (?,?,?)",
                  (plate, now, rate))
        conn.commit()
        conn.close()
        return "entry", {"plate": plate, "entry_time": now}
    else:
        sid = open_s[0]
        entry_time = datetime.strptime(open_s[1], "%Y-%m-%d %H:%M:%S")
        duration = round((datetime.now(SYDNEY_TZ).replace(tzinfo=None)-entry_time).total_seconds()/60, 2)
        fee = round(duration * rate, 2)
        c.execute("UPDATE parking_sessions SET exit_time=?,duration_min=?,fee=? WHERE id=?",
                  (now, duration, fee, sid))
        conn.commit()
        conn.close()
        return "exit", {"plate": plate, "entry_time": open_s[1],
                        "exit_time": now, "duration": duration, "fee": fee}

def log_detection(plate, conf, condition="normal", ocr_conf=0, source="image"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO plate_detections (plate_text,confidence,condition,ocr_conf,source,created_at) VALUES (?,?,?,?,?,?)",
                 (plate, conf, condition, ocr_conf, source, now_sydney()))
    conn.commit()
    conn.close()

def get_dashboard_data():
    conn = sqlite3.connect(DB_PATH)
    open_n   = conn.execute("SELECT COUNT(*) FROM parking_sessions WHERE exit_time IS NULL").fetchone()[0]
    closed_n = conn.execute("SELECT COUNT(*) FROM parking_sessions WHERE exit_time IS NOT NULL").fetchone()[0]
    revenue  = conn.execute("SELECT COALESCE(SUM(fee),0) FROM parking_sessions WHERE exit_time IS NOT NULL").fetchone()[0]
    recent   = conn.execute("SELECT plate_text,confidence,condition,created_at FROM plate_detections ORDER BY created_at DESC LIMIT 10").fetchall()
    sessions = conn.execute("SELECT plate_text,entry_time,exit_time,duration_min,fee FROM parking_sessions ORDER BY entry_time DESC LIMIT 10").fetchall()
    conn.close()
    return open_n, closed_n, revenue, recent, sessions

def clear_all_data():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM plate_detections")
    conn.execute("DELETE FROM parking_sessions")
    conn.commit()
    conn.close()

# ── Models ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class CSDN_Tem(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
            self.point_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        def forward(self, x):
            return self.point_conv(self.depth_conv(x))

    class ZeroDCE(nn.Module):
        def __init__(self, scale_factor=12):
            super().__init__()
            self.scale_factor = scale_factor
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            self.relu = nn.ReLU(inplace=True)
            nf = 32
            self.e_conv1 = CSDN_Tem(3, nf)
            self.e_conv2 = CSDN_Tem(nf, nf)
            self.e_conv3 = CSDN_Tem(nf, nf)
            self.e_conv4 = CSDN_Tem(nf, nf)
            self.e_conv5 = CSDN_Tem(nf*2, nf)
            self.e_conv6 = CSDN_Tem(nf*2, nf)
            self.e_conv7 = CSDN_Tem(nf*2, 3)
        def enhance(self, x, x_r):
            for _ in range(8):
                x = x + x_r * (torch.pow(x, 2) - x)
            return x
        def forward(self, x):
            if self.scale_factor == 1:
                x_down = x
            else:
                x_down = F.interpolate(x, scale_factor=1/self.scale_factor, mode="bilinear")
            x1 = self.relu(self.e_conv1(x_down))
            x2 = self.relu(self.e_conv2(x1))
            x3 = self.relu(self.e_conv3(x2))
            x4 = self.relu(self.e_conv4(x3))
            x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
            x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
            x_r = torch.tanh(self.e_conv7(torch.cat([x1,x6],1)))
            if self.scale_factor != 1:
                x_r = self.upsample(x_r)
            return self.enhance(x, x_r), x_r

    dce = ZeroDCE(SCALE_FACTOR).to(device)
    dce.load_state_dict(torch.load(DCE_WEIGHTS, map_location=device))
    dce.eval()

    from ultralytics import YOLO
    detector = YOLO(YOLO_WEIGHTS)
    # Auto-clone PARSEQ if not present
    if not os.path.exists(PARSEQ_REPO):
        import subprocess as _sp2
        _sp2.run(["git", "clone",
                  "https://github.com/baudm/parseq",
                  PARSEQ_REPO], check=True)
    import subprocess as _sp
    _sp.run([sys.executable, "-m", "pip", "install",
             PARSEQ_REPO, "--quiet"], capture_output=True)
    sys.path.insert(0, PARSEQ_REPO)
    for k in list(sys.modules.keys()):
        if "strhub" in k:
            del sys.modules[k]
    from strhub.data.module import SceneTextDataModule
    from strhub.models.utils import create_model
    import strhub.data.utils as su

    mp = create_model("parseq", pretrained=False).to(device)
    tf = SceneTextDataModule.get_transform(mp.hparams.img_size)
    ck = torch.load(PARSEQ_CKPT, map_location=device, weights_only=False)
    cs = ck["charset"]
    for s in ("[E]","[B]","[P]"):
        cs = cs.replace(s, "")

    oc = list(getattr(mp.tokenizer, "_itos", []))
    os_ = {c: i for i, c in enumerate(oc)}
    nt = su.Tokenizer(cs)
    nc = list(getattr(nt, "_itos", cs))
    nn_ = len(nc)

    oe, ep = None, None
    for n, m in mp.named_modules():
        if isinstance(m, nn.Embedding) and "pos" not in n.lower():
            oe, ep = m, n
            break

    oh = mp.model.head
    dm = oe.embedding_dim
    es = oe.num_embeddings
    hs = oh.out_features
    ho = es - hs
    nhs = max(1, nn_ - ho)

    ne = nn.Embedding(nn_, dm)
    nh = nn.Linear(dm, nhs)
    nn.init.normal_(ne.weight, std=0.02)
    nn.init.zeros_(nh.bias)
    nn.init.normal_(nh.weight, std=0.02)

    with torch.no_grad():
        for ni, ch in enumerate(nc):
            if ch in os_:
                oi = os_[ch]
                if oi < es:
                    ne.weight[ni] = oe.weight[oi]
                ohi = oi - ho
                nhi = ni - ho
                if 0 <= ohi < hs and 0 <= nhi < nhs:
                    nh.weight[nhi] = oh.weight[ohi]
                    nh.bias[nhi] = oh.bias[ohi]

    pts = ep.split(".")
    par = mp
    for pt in pts[:-1]:
        par = getattr(par, pt)
    setattr(par, pts[-1], ne)
    mp.model.head = nh
    mp.tokenizer = nt
    for a in ("pad_id","bos_id","eos_id"):
        if hasattr(nt, a):
            setattr(mp, a, getattr(nt, a))

    mp.load_state_dict(ck["model_state"])
    mp = mp.to(device)
    if hasattr(mp.model, "pos_queries"):
        mp.model.pos_queries = mp.model.pos_queries.to(device)
    mp.eval()

    return device, dce, detector, mp, tf

# ── Helper functions ──────────────────────────────────────────
def enhance_if_dark(img, dce, device, thr=45):
    if np.mean(img) >= thr:
        return img, False
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    t = torch.from_numpy(rgb.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
    ph, pw = (-h) % SCALE_FACTOR, (-w) % SCALE_FACTOR
    if ph or pw:
        t = F.pad(t, (0,pw,0,ph), mode="reflect")
    with torch.no_grad():
        enh, _ = dce(t)
    enh = enh[:,:,:h,:w]
    out = (np.clip(enh.squeeze(0).permute(1,2,0).cpu().numpy(),0,1)*255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR), True

def detect_and_read(img, detector, mp, tf, device):
    t0 = time.time()
    res = detector.predict(img, conf=0.25, verbose=False)
    boxes = res[0].boxes
    dets = []
    ann = img.copy()
    if boxes is not None and len(boxes) > 0:
        for bi in range(len(boxes)):
            x1,y1,x2,y2 = map(int, boxes.xyxy[bi].tolist())
            conf = float(boxes.conf[bi])
            h, w = img.shape[:2]
            crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            plate, ocr_conf = "", 0.0
            if crop.size > 0:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                inp = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    logits = mp(inp)
                    probs = logits.softmax(-1)
                    pred, _ = mp.tokenizer.decode(probs)
                plate = str(pred[0]).upper().strip()
                ocr_conf = float(probs.max())
            cv2.rectangle(ann, (x1,y1), (x2,y2), (0,200,0), 3)
            cv2.putText(ann, f"{plate} | Conf. {conf:.2f}",
                       (x1, max(y1-8,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)
            dets.append({"plate": plate, "confidence": round(conf,3),
                         "ocr_conf": round(ocr_conf,3), "box": [x1,y1,x2,y2]})
    fps = 1/(time.time()-t0+1e-6)
    return ann, dets, fps

def save_video_h264(frames, fps, output_path):
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        raw_path = output_path.replace(".mp4", "_raw.mp4")
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            raw_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(fps, 5),
            (w, h)
        )
        for frame in frames:
            writer.write(frame)
        writer.release()
        subprocess.run([
            ffmpeg_exe, "-y", "-i", raw_path,
            "-vcodec", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            output_path
        ], capture_output=True, timeout=120)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            os.remove(raw_path)
            return True
    except Exception as e:
        st.warning(f"Video conversion: {e}")
    return False

# ── Main App ──────────────────────────────────────────────────
def main():
    init_db()
    st.markdown("<h2 style='text-align:center'>🚗 Smart Parking ALPR</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#8a93a3'>YOLOv26s + Zero-DCE++ + PARSEQ-FT + ByteTrack</p>", unsafe_allow_html=True)
    st.divider()

    with st.spinner("Loading models..."):
        device, dce, detector, mp, tf = load_models()
    st.success("✅ Models loaded!")

    fee_rate = st.sidebar.slider("💰 Parking rate ($/min)", 0.1, 2.0, 0.5)
    st.sidebar.divider()
    st.sidebar.markdown("**Model Info**")
    st.sidebar.caption("🎯 YOLOv26s — plate detector")
    st.sidebar.caption("🌙 Zero-DCE++ — night enhancement")
    st.sidebar.caption("📝 PARSEQ-FT — OCR (val_acc=95.9%)")
    st.sidebar.caption("🔄 ByteTrack — vehicle tracking")
    st.sidebar.divider()
    st.sidebar.caption(f"🕐 Sydney time: {datetime.now(SYDNEY_TZ).strftime('%H:%M:%S')}")

    col1, col2, col3 = st.columns([1, 2, 1.5])

    # ── Column 1: Input Control ───────────────────────────────
    with col1:
        st.markdown("### 1. Input Control")
        mode = st.radio("", ["📷 Upload Image", "🎥 Upload Video", "📸 Live Webcam"],
                        label_visibility="collapsed")
        st.divider()
        uploaded_img   = None
        uploaded_video = None
        webcam_img     = None
        if mode == "📷 Upload Image":
            uploaded_img = st.file_uploader("Choose image",
                           type=["jpg","jpeg","png"], label_visibility="collapsed")
        elif mode == "🎥 Upload Video":
            uploaded_video = st.file_uploader("Choose video",
                             type=["mp4","avi","mov"], label_visibility="collapsed")
            st.caption("Processes every 5th frame with ByteTrack")
        elif mode == "📸 Live Webcam":
            webcam_img = st.camera_input("", label_visibility="collapsed")

    # ── Column 2: Detection & Recognition ────────────────────
    with col2:
        st.markdown("### 2. Detection & Recognition")
        result_placeholder = st.empty()
        current_dets = []
        fps_val = 0.0
        condition = "normal"
        plate_text = ""
        conf_val = 0.0
        ocr_conf = 0.0

        if uploaded_img:
            arr = np.frombuffer(uploaded_img.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            img, enhanced = enhance_if_dark(img, dce, device)
            condition = "night" if enhanced else "normal"
            if enhanced:
                st.info("🌙 Dark image detected — Zero-DCE++ enhancement applied")
            ann, dets, fps_val = detect_and_read(img, detector, mp, tf, device)
            result_placeholder.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
            current_dets = dets
            if dets:
                plate_text = dets[0]["plate"]
                conf_val   = dets[0]["confidence"]
                ocr_conf   = dets[0]["ocr_conf"]

        elif webcam_img:
            arr = np.frombuffer(webcam_img.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            img, enhanced = enhance_if_dark(img, dce, device)
            condition = "night" if enhanced else "normal"
            ann, dets, fps_val = detect_and_read(img, detector, mp, tf, device)
            result_placeholder.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
            current_dets = dets
            if dets:
                plate_text = dets[0]["plate"]
                conf_val   = dets[0]["confidence"]
                ocr_conf   = dets[0]["ocr_conf"]

        elif uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            votes = {}
            fc = 0
            prog = st.progress(0)
            stat = st.empty()
            annotated_frames = []

            while cap.isOpened() and fc < 500:
                ret, frame = cap.read()
                if not ret:
                    break
                fc += 1
                if fc % 5 != 0:
                    continue
                prog.progress(min(fc/500, 1.0))
                stat.text(f"Processing frame {fc}...")
                frame, _ = enhance_if_dark(frame, dce, device)
                res = detector.track(frame, persist=True, tracker="bytetrack.yaml",
                                    conf=0.25, verbose=False)
                bxs = res[0].boxes
                annotated_frames.append(res[0].plot())
                if bxs is not None and len(bxs) > 0:
                    for bi in range(len(bxs)):
                        tid = int(bxs.id[bi].item()) if bxs.id is not None else -1
                        if tid < 0:
                            continue
                        x1,y1,x2,y2 = map(int, bxs.xyxy[bi].tolist())
                        h, w = frame.shape[:2]
                        crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                        if crop.size == 0:
                            continue
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        inp = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)
                        with torch.inference_mode():
                            logits = mp(inp)
                            pred, _ = mp.tokenizer.decode(logits.softmax(-1))
                        p = str(pred[0]).upper().strip()
                        if p:
                            votes.setdefault(tid, []).append(p)

            cap.release()
            prog.empty()
            stat.empty()

            if annotated_frames:
                out_path = f"/tmp/annotated_{uuid.uuid4().hex[:8]}.mp4"
                saved = save_video_h264(annotated_frames, src_fps/5, out_path)
                if saved and os.path.exists(out_path):
                    with open(out_path, "rb") as vf:
                        result_placeholder.video(vf.read(), format="video/mp4")
                else:
                    result_placeholder.image(
                        cv2.cvtColor(annotated_frames[-1], cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                        caption="Last processed frame"
                    )

            plate_support = Counter()
            for tid, preds in votes.items():
                if len(preds) >= 3:
                    voted, _ = Counter(preds).most_common(1)[0]
                    plate_support[voted] += len(preds)

            for plate, support in plate_support.items():
                ev, info = register_gate_event(plate, fee_rate)
                if ev == "entry":
                    st.success(f"✅ CHECK-IN: **{plate}** ({support} frames)")
                elif ev == "exit":
                    st.warning(f"🚪 CHECK-OUT: **{plate}** | {info['duration']} min | **${info['fee']}**")

        else:
            result_placeholder.markdown("""
            <div style='background:#1e2130;border-radius:12px;padding:60px;
                        text-align:center;min-height:300px'>
                <div style='font-size:80px'>🚗</div>
                <p style='color:#8a93a3;margin-top:10px'>
                    Upload an image or video or use webcam
                </p>
            </div>""", unsafe_allow_html=True)

        if plate_text:
            st.markdown(f"<div class='plate-text'>{plate_text}</div>", unsafe_allow_html=True)

        if current_dets or fps_val > 0:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Recognised Plate", plate_text or "—")
            m2.metric("Detection Conf.",  f"{conf_val:.0%}" if conf_val else "—")
            m3.metric("Condition",        condition)
            m4.metric("FPS",              f"{fps_val:.1f}" if fps_val else "—")
            if ocr_conf:
                proc = 1/fps_val if fps_val else 0
                st.caption(f"OCR confidence: {ocr_conf:.2f}  |  Processing time: {proc:.2f}s")

        if current_dets and (uploaded_img or webcam_img):
            for d in current_dets:
                if d["plate"]:
                    log_detection(d["plate"], d["confidence"], condition,
                                 d["ocr_conf"], "image" if uploaded_img else "webcam")
                    ev, info = register_gate_event(d["plate"], fee_rate)
                    if ev == "entry":
                        st.success(f"✅ CHECK-IN: **{d['plate']}**")
                    elif ev == "exit":
                        st.warning(f"🚪 CHECK-OUT: **{d['plate']}** | {info['duration']} min | **${info['fee']}**")

    # ── Column 3: Results Log ─────────────────────────────────
    with col3:
        st.markdown("### 3. Results Log")
        open_n, closed_n, revenue, recent, sessions = get_dashboard_data()

        k1, k2, k3 = st.columns(3)
        k1.metric("🚗 Parked",  open_n)
        k2.metric("✅ Done",    closed_n)
        k3.metric("💰 Revenue", f"${revenue:.2f}")
        st.divider()

        st.markdown("**Recent Detections**")
        if recent:
            for row in recent:
                plate, conf, cond, ts = row
                time_str = ts.split(" ")[1] if " " in ts else ts
                st.markdown(f"""<div class='log-row'>
                    <span style='color:#00ff88;font-family:monospace;font-weight:bold'>{plate}</span>
                    <span style='color:#6db3f2'>{conf:.2f}</span>
                    <span style='color:#8a93a3'>{time_str}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.caption("No detections yet.")

        st.divider()
        st.markdown("**Parking Sessions**")
        if sessions:
            for row in sessions:
                plate, entry, exit_t, dur, fee = row
                entry_str = entry.split(" ")[1] if " " in entry else entry
                if exit_t:
                    st.markdown(f"""<div class='log-row'>
                        <span style='color:#fbbf24;font-family:monospace;font-weight:bold'>{plate}</span>
                        <span style='color:#8a93a3'>{dur:.1f}min</span>
                        <span style='color:#fbbf24'>${fee:.2f}</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class='log-row'>
                        <span style='color:#00ff88;font-family:monospace;font-weight:bold'>{plate}</span>
                        <span style='color:#8a93a3'>In: {entry_str}</span>
                        <span style='color:#00ff88'>Parked</span>
                    </div>""", unsafe_allow_html=True)
        else:
            st.caption("No sessions yet.")

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📥 Export", key="btn_export"):
                import pandas as pd
                conn = sqlite3.connect(DB_PATH)
                df = pd.read_sql("SELECT * FROM plate_detections", conn)
                conn.close()
                st.download_button("⬇️ Download", df.to_csv(index=False),
                                  "detections.csv", "text/csv",
                                  key="btn_download")
        with c2:
            if st.button("🔄 Refresh", key="btn_refresh"):
                st.rerun()
        with c3:
            if st.button("🗑️ Clear", key="btn_clear", type="secondary"):
                clear_all_data()
                st.success("✅ Cleared!")
                st.rerun()

if __name__ == "__main__":
    main()