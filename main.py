import cv2
import numpy as np
import os
import time
import base64
import threading
import re
from collections import deque, defaultdict
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ultralytics import YOLO

# ============================================================
#                    CONFIGURATION
# ============================================================
GOOGLE_API_KEY = "AQ.Ab8RN6JBDDLVQ3LDoXgd8oMoA4Mn693DJ6C71cfqhVhEiNM2MQ"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

PRIMARY_MODEL  = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3.1-flash-lite"

FRAME_FOLDER = "captures"
SUSPECT_DIR  = "suspects"
ALERT_DIR    = "alerts"
LOG_FILE     = "observations.txt"
SUSPECT_LOG  = "suspect_log.txt"
ALERT_LOG    = "alerts.txt"
for d in (FRAME_FOLDER, SUSPECT_DIR, ALERT_DIR):
    os.makedirs(d, exist_ok=True)

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH  = 1020
FRAME_HEIGHT = 560

# Gemini throttling
SEND_INTERVAL     = 45
IDLE_INTERVAL     = 120
DAILY_QUOTA_LIMIT = 18
URGENT_INTERVAL   = 5     # seconds between Gemini calls when weapon/fight detected

# Pose / fight / abuse tuning
POSE_CONF         = 0.4
PERSON_CONF       = 0.4
FIGHT_DIST        = 180
FIGHT_SPEED       = 40
STRUGGLE_MOTION   = 25

# Weapon detection tuning
WEAPON_CONF       = 0.35
WEAPON_CLASSES    = [
    "knife", "gun", "pistol", "handgun", "rifle",
    "machete", "sword", "dagger", "blade",
    "baseball bat", "rod", "stick", "hammer", "axe", "crowbar",
    "scissors", "screwdriver", "bottle", "iron rod"
]
WEAPON_MODEL_PATH = "yolov8s-worldv2.pt"   # zero-shot model
# If you want a dedicated weapon-trained model instead, set:
# WEAPON_MODEL_PATH = "weapon_yolov8.pt"   # and skip YOLO-World set_classes below

# Models
yolo_pose   = YOLO("yolov8n-pose.pt")
yolo_weapon = YOLO(WEAPON_MODEL_PATH)
USING_WORLD = "world" in WEAPON_MODEL_PATH.lower()
if USING_WORLD:
    yolo_weapon.set_classes(WEAPON_CLASSES)

PERSON_CLASS_ID = 0

gemini_primary  = ChatGoogleGenerativeAI(model=PRIMARY_MODEL)
gemini_fallback = ChatGoogleGenerativeAI(model=FALLBACK_MODEL)

# ============================================================
#                    GLOBAL STATE
# ============================================================
last_sent_time        = 0
quota_used_times      = deque()
quota_exhausted_until = 0
track_lock            = threading.Lock()

pose_history   = defaultdict(lambda: deque(maxlen=30))
event_cooldown = {}
suspicious_track_id    = None
suspicious_description = ""
active_alerts = deque(maxlen=8)
CASH_COUNTER_ROI = None

# COCO keypoint indices
KP_NOSE, KP_L_SHLDR, KP_R_SHLDR = 0, 5, 6
KP_L_WRIST, KP_R_WRIST          = 9, 10


# ============================================================
#                    UTILITIES
# ============================================================
def extract_text(response):
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and "text" in b:
                parts.append(b["text"])
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts).strip()
    return str(content).strip()


def quota_allows():
    now = time.time()
    cutoff = now - 86400
    while quota_used_times and quota_used_times[0] < cutoff:
        quota_used_times.popleft()
    return len(quota_used_times) < DAILY_QUOTA_LIMIT


def register_quota_use():
    quota_used_times.append(time.time())


def can_fire(key, cooldown=8.0):
    now = time.time()
    if now - event_cooldown.get(key, 0) < cooldown:
        return False
    event_cooldown[key] = now
    return True


def add_alert(text, color=(0, 0, 255)):
    active_alerts.append((time.time(), text, color))
    print(f"🚨 ALERT: {text}")
    with open(ALERT_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d_%H-%M-%S')}] {text}\n")


# ============================================================
#                WEAPON DETECTION
# ============================================================
def detect_weapons(frame):
    """
    Runs a weapon model on the frame.
    Returns list of (label, conf, bbox) for any detected weapon.
    """
    detections = []
    try:
        if USING_WORLD:
            results = yolo_weapon.predict(frame, conf=WEAPON_CONF, verbose=False)[0]
            names = yolo_weapon.names
        else:
            results = yolo_weapon.predict(frame, conf=WEAPON_CONF, verbose=False)[0]
            names = yolo_weapon.names

        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf   = float(box.conf.cpu().numpy()[0])
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            # filter false-positive-prone objects when using World
            if USING_WORLD and label.lower() in ("bottle", "scissors", "screwdriver"):
                # require higher confidence for ambiguous household items
                if conf < 0.5:
                    continue
            detections.append((label, conf, (x1, y1, x2, y2)))
    except Exception as e:
        print(f"⚠️ Weapon detection error: {e}")
    return detections


# ============================================================
#                POSE-BASED ACTION DETECTION
# ============================================================
def keypoints_of(kps, idx):
    if kps is None or idx >= len(kps):
        return None
    x, y, c = kps[idx]
    return (float(x), float(y), float(c))


def wrist_motion(history):
    if len(history) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(1, len(history)):
        _, wl_prev, wr_prev, _ = history[i - 1]
        _, wl_cur,  wr_cur,  _ = history[i]
        for a, b in ((wl_prev, wl_cur), (wr_prev, wr_cur)):
            if a is None or b is None:
                continue
            total += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            count += 1
    return total / count if count else 0.0


def bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_center_dist(b1, b2):
    c1, c2 = bbox_center(b1), bbox_center(b2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def point_in_bbox(pt, bbox, pad=10):
    x, y = pt
    x1, y1, x2, y2 = bbox
    return (x1 - pad <= x <= x2 + pad) and (y1 - pad <= y <= y2 + pad)


def detect_actions(frame, pose_result):
    events = []
    if not pose_result or pose_result.keypoints is None:
        return events

    kps_all = pose_result.keypoints.data.cpu().numpy()
    boxes   = pose_result.boxes.xyxy.cpu().numpy()
    ids     = (pose_result.boxes.id.cpu().numpy().astype(int)
               if pose_result.boxes.id is not None
               else np.arange(len(boxes)))

    now = time.time()

    # store pose history
    for kps, box, tid in zip(kps_all, boxes, ids):
        wl = keypoints_of(kps, KP_L_WRIST)
        wr = keypoints_of(kps, KP_R_WRIST)
        wl = wl if wl and wl[2] > POSE_CONF else None
        wr = wr if wr and wr[2] > POSE_CONF else None
        pose_history[tid].append((now, wl, wr, tuple(box)))

    tids = list(ids)

    # -------- FIGHT --------
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            t1, t2 = tids[i], tids[j]
            h1, h2 = pose_history[t1], pose_history[t2]
            if len(h1) < 2 or len(h2) < 2:
                continue
            d = bbox_center_dist(h1[-1][3], h2[-1][3])
            if d > FIGHT_DIST:
                continue
            m1, m2 = wrist_motion(h1), wrist_motion(h2)
            if m1 > FIGHT_SPEED or m2 > FIGHT_SPEED:
                if can_fire(f"fight_{min(t1,t2)}_{max(t1,t2)}", 6):
                    events.append(f"FIGHT between ID {t1} and ID {t2}")
                    add_alert(f"FIGHT: ID {t1} & ID {t2}")

    # -------- SNATCH / GRAB --------
    for kps, box, tid in zip(kps_all, boxes, ids):
        h = pose_history[tid]
        if len(h) < 3:
            continue
        wl = keypoints_of(kps, KP_L_WRIST)
        wr = keypoints_of(kps, KP_R_WRIST)
        sl = keypoints_of(kps, KP_L_SHLDR)
        sr = keypoints_of(kps, KP_R_SHLDR)
        if not (wl and wr and sl and sr):
            continue
        wrists_high = (wl[1] < sl[1]) and (wr[1] < sr[1])
        m = wrist_motion(h)
        if wrists_high and m > STRUGGLE_MOTION:
            if can_fire(f"snatch_{tid}", 8):
                events.append(f"SNATCH/GRAB by ID {tid}")
                add_alert(f"SNATCH/GRAB: ID {tid}")

    # -------- ABUSE / HARASSMENT --------
    for i in range(len(tids)):
        for j in range(len(tids)):
            if i == j:
                continue
            t1, t2 = tids[i], tids[j]
            h1, h2 = pose_history[t1], pose_history[t2]
            if len(h1) < 3 or len(h2) < 3:
                continue
            d_prev = bbox_center_dist(h1[-2][3], h2[-2][3])
            d_now  = bbox_center_dist(h1[-1][3], h2[-1][3])
            closing_fast = (d_prev - d_now) > 25
            if closing_fast and wrist_motion(h1) > STRUGGLE_MOTION:
                if can_fire(f"abuse_{t1}_{t2}", 10):
                    events.append(f"HARASSMENT: ID {t1} approaching ID {t2}")
                    add_alert(f"HARASSMENT: ID {t1} → ID {t2}")

    return events


def weapon_events(weapons, track_summary, frame):
    """
    Turn raw weapon detections into alert strings and draw their boxes.
    Also marks any person whose bbox overlaps a weapon as the suspect.
    """
    global suspicious_track_id, suspicious_description
    events = []
    for label, conf, wbox in weapons:
        # draw box
        x1, y1, x2, y2 = wbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"{label.upper()} {conf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # find nearest person holding it
        wc = ((x1 + x2) / 2, (y1 + y2) / 2)
        best_tid, best_d = None, 1e9
        for tid, pbox in track_summary:
            if point_in_bbox(wc, pbox, pad=20):
                best_tid = tid
                break
            d = bbox_center_dist(pbox, wbox)
            if d < best_d:
                best_d, best_tid = d, tid

        if best_tid is not None and best_d < 200:
            with track_lock:
                suspicious_track_id = best_tid
                suspicious_description = f"Armed with {label}"
            events.append(f"WEAPON: {label} (conf {conf:.2f}) — ID {best_tid}")
            if can_fire(f"weapon_{label}_{best_tid}", 6):
                add_alert(f"WEAPON DETECTED: {label} → ID {best_tid}")
        else:
            events.append(f"WEAPON: {label} (conf {conf:.2f}) — unassigned")
            if can_fire(f"weapon_{label}_unassigned", 8):
                add_alert(f"WEAPON DETECTED: {label}")
    return events


# ============================================================
#                GEMINI VERIFICATION
# ============================================================
def analyze_with_gemini(image_path, timestamp, track_summary, yolo_events):
    global suspicious_track_id, suspicious_description, quota_exhausted_until
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        tracks_txt = "\n".join(
            f"- Track ID {tid}: bbox=({x1},{y1},{x2},{y2})"
            for tid, (x1, y1, x2, y2) in track_summary
        ) or "- No persons"

        yolo_txt = "\n".join(f"- {e}" for e in yolo_events) or "- none"

        prompt = f"""
You are a security analyst reviewing a CCTV frame.

Persons visible (with tracking IDs):
{tracks_txt}

Automatic YOLO signals (may be wrong, confirm or reject):
{yolo_txt}

Look for and describe:
- WEAPONS: knife, gun, pistol, rifle, machete, sword, rod, bat, hammer, axe, crowbar, any blade
- FIGHTING / assault / physical violence
- WOMAN ABUSE / harassment / molestation / stalking
- CHAIN SNATCHING / bag snatching / robbery / pickpocketing
- THEFT / concealment / shoplifting
- THREATENING gesture / pointing weapon / aggressive posture
- VANDALISM / trespassing / forced entry

Respond ONLY as this markdown table:

| Activity | Observed? (Yes/No) | Suspect Track ID | Suspect Description |
|----------|--------------------|------------------|---------------------|
| <activity> | <Yes/No> | <ID or none> | <gender, clothing color, weapon visible?, mask, bag, features> |

Rules:
- If nothing: one row with "None", "No", "none", "".
- One row per distinct activity.
- Explicitly confirm or reject any YOLO signals above.
- Description must be visual and specific, and mention weapon if any.
"""

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ])

        try:
            response = gemini_primary.invoke([message])
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print("↩ Primary quota hit, using fallback...")
                response = gemini_fallback.invoke([message])
            else:
                raise

        observation = extract_text(response)
        print(f"🔎 Gemini ({timestamp}):\n{observation}\n")

        is_sus, findings = parse_table(observation)
        if is_sus:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} - {observation}\n\n")
            for activity, tid, desc in findings:
                with track_lock:
                    if tid is not None:
                        suspicious_track_id = tid
                    if desc:
                        suspicious_description = desc
                with open(SUSPECT_LOG, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] TRACK_ID={tid} ACTIVITY={activity} DESC={desc}\n")
                add_alert(f"GEMINI: {activity} (ID {tid})")
                if tid is not None:
                    save_suspect_crop(image_path, track_summary, tid, timestamp)

    except Exception as e:
        msg = str(e)
        m = re.search(r"retry in ([\d\.]+)s", msg) or re.search(r"'retryDelay': '(\d+)s'", msg)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            delay = float(m.group(1)) if m else 60.0
            quota_exhausted_until = time.time() + delay + 5
            print(f"⏸ Rate-limit backoff {delay:.0f}s.")
        else:
            print(f"❌ Gemini error: {e}")


def parse_table(observation):
    findings, is_sus = [], False
    for line in observation.splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line or "observed?" in line.lower():
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        if cells[1].lower().startswith("yes"):
            is_sus = True
            tid = None
            for tok in cells[2].replace("#", " ").split():
                if tok.isdigit():
                    tid = int(tok)
                    break
            findings.append((cells[0], tid, cells[3]))
    return is_sus, findings


def save_suspect_crop(image_path, track_summary, tid, timestamp):
    try:
        img = cv2.imread(image_path)
        for t, (x1, y1, x2, y2) in track_summary:
            if t == tid:
                pad = 20
                x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
                x2p, y2p = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
                crop = img[y1p:y2p, x1p:x2p]
                if crop.size:
                    cv2.imwrite(
                        os.path.join(SUSPECT_DIR, f"suspect_id{tid}_{timestamp}.jpg"),
                        crop)
                break
    except Exception as e:
        print(f"⚠️ crop save failed: {e}")


# ============================================================
#                    DISPATCH
# ============================================================
def maybe_send_frame(frame, track_summary, yolo_events):
    global last_sent_time
    now = time.time()
    if now < quota_exhausted_until or not quota_allows():
        return

    urgent = any(("WEAPON" in e) or ("FIGHT" in e) or ("SNATCH" in e) for e in yolo_events)
    interval = URGENT_INTERVAL if urgent else SEND_INTERVAL

    if now - last_sent_time < interval:
        return

    last_sent_time = now
    register_quota_use()

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    fname = os.path.join(FRAME_FOLDER, f"frame_{ts}.jpg")
    cv2.imwrite(fname, frame)

    threading.Thread(
        target=analyze_with_gemini,
        args=(fname, ts, track_summary, yolo_events),
        daemon=True,
    ).start()


# ============================================================
#                    MAIN LOOP
# ============================================================
def open_camera():
    if isinstance(CAMERA_INDEX, str) and CAMERA_INDEX.startswith(("rtsp", "http")):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not open camera {CAMERA_INDEX}")
    return cap


def draw_alert_feed(frame):
    y = 80
    now = time.time()
    for ts, text, color in list(active_alerts)[::-1]:
        if now - ts > 6:
            continue
        cv2.putText(frame, f"⚠ {text}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 22


def start_monitoring():
    global suspicious_track_id, suspicious_description, CASH_COUNTER_ROI
    try:
        cap = open_camera()
    except RuntimeError as e:
        print(e)
        return

    cv2.namedWindow("Security Monitor — Weapons / Fight / Abuse", cv2.WINDOW_NORMAL)
    print("▶ q=quit  r=reset suspect  c=clear ROI")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # --- person pose tracking ---
        pose_res = yolo_pose.track(
            frame, persist=True, verbose=False,
            classes=[PERSON_CLASS_ID], conf=PERSON_CONF,
            tracker="bytetrack.yaml",
        )[0]

        track_summary = []
        if pose_res.boxes is not None and pose_res.boxes.id is not None:
            boxes = pose_res.boxes.xyxy.cpu().numpy()
            ids   = pose_res.boxes.id.cpu().numpy().astype(int)
            for b, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, b)
                track_summary.append((tid, (x1, y1, x2, y2)))

        # --- pose-based action detection ---
        action_events = detect_actions(frame, pose_res)

        # --- weapon detection ---
        weapons = detect_weapons(frame)
        weapon_alerts = weapon_events(weapons, track_summary, frame)

        yolo_events = action_events + weapon_alerts

        # --- draw person boxes ---
        for tid, (x1, y1, x2, y2) in track_summary:
            with track_lock:
                is_suspect = (suspicious_track_id is not None
                              and tid == suspicious_track_id)
            color = (0, 0, 255) if is_suspect else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_suspect else 2)
            label = f"ID {tid}" + ("  ⚠ SUSPECT" if is_suspect else "")
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if is_suspect and suspicious_description:
                cv2.putText(frame, suspicious_description[:60],
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 255), 1)

        # --- draw pose dots ---
        if pose_res.keypoints is not None:
            for kps in pose_res.keypoints.data.cpu().numpy():
                for x, y, c in kps:
                    if c > POSE_CONF:
                        cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 255), -1)

        # --- HUD ---
        cv2.putText(frame, f"Persons: {len(track_summary)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Quota: {len(quota_used_times)}/{DAILY_QUOTA_LIMIT}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        draw_alert_feed(frame)

        # --- Gemini dispatch ---
        if track_summary or yolo_events:
            maybe_send_frame(frame.copy(), track_summary, yolo_events)

        cv2.imshow("Security Monitor — Weapons / Fight / Abuse", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            with track_lock:
                suspicious_track_id = None
                suspicious_description = ""
            print("♻ Suspect reset.")
        elif key == ord("c"):
            CASH_COUNTER_ROI = None
            print("ROI cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Monitoring stopped.")


if __name__ == "__main__":
    start_monitoring()