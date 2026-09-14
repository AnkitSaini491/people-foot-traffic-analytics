
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# =========================
# SETTINGS
# =========================

VIDEO_PATH = "input.mp4"
OUTPUT_PATH = "output.mp4"

# YOLO model
model = YOLO("yolo11n.pt")

# Person class in COCO
PERSON_CLASS = 0

# Counting line
LINE_Y = 380

# Vision point (approximately like your video)
VISION_POINT = (570, 45)

# =========================
# VIDEO
# =========================

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

# =========================
# TRACKING DATA
# =========================

track_history = defaultdict(list)

counted_ids = set()

inside_ids = set()

in_count = 0
out_count = 0

# =========================
# FUNCTIONS
# =========================

def draw_corner_box(frame, x1, y1, x2, y2):
    """
    Draw cyan corner-style bounding box.
    """

    color = (255, 255, 0)
    thickness = 2
    corner = 18

    # Top left
    cv2.line(frame, (x1, y1), (x1 + corner, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), color, thickness)

    # Top right
    cv2.line(frame, (x2, y1), (x2 - corner, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), color, thickness)

    # Bottom left
    cv2.line(frame, (x1, y2), (x1 + corner, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), color, thickness)

    # Bottom right
    cv2.line(frame, (x2, y2), (x2 - corner, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), color, thickness)


def draw_label(frame, text, x, y):

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1

    (tw, th), _ = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    cv2.rectangle(
        frame,
        (x, y - th - 8),
        (x + tw + 8, y),
        (255, 255, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 4, y - 5),
        font,
        scale,
        (0, 0, 0),
        thickness
    )


def draw_dashboard(frame, frame_number):

    # Dashboard background
    cv2.rectangle(
        frame,
        (10, 10),
        (195, 82),
        (25, 25, 25),
        -1
    )

    cv2.rectangle(
        frame,
        (10, 10),
        (195, 82),
        (80, 80, 80),
        1
    )

    # Title
    cv2.putText(
        frame,
        "VISIONEYE  -  FOOT TRAFFIC",
        (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"people now: {len(inside_ids)}",
        (20, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"total seen: {len(track_history)}",
        (110, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"IN: {in_count}   OUT: {out_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"frame: {frame_number}",
        (20, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (180, 180, 180),
        1
    )


def draw_scene_map(frame):

    map_w = 280
    map_h = 175

    x1 = width - map_w - 10
    y1 = height - map_h - 10

    x2 = width - 10
    y2 = height - 10

    # Map
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (35, 35, 35),
        -1
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (100, 100, 100),
        2
    )

    cv2.putText(
        frame,
        "SCENE MAP (top view)",
        (x1 + 8, y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1
    )

    # Grid
    for i in range(1, 5):

        gx = x1 + i * map_w // 5

        cv2.line(
            frame,
            (gx, y1 + 25),
            (gx, y2),
            (60, 60, 60),
            1
        )

    for i in range(1, 4):

        gy = y1 + 25 + i * (map_h - 25) // 4

        cv2.line(
            frame,
            (x1, gy),
            (x2, gy),
            (60, 60, 60),
            1
        )

    # Draw people positions
    for history in track_history.values():

        if len(history) == 0:
            continue

        px, py = history[-1]

        # Convert original position to map position
        mx = int(
            x1 + (px / width) * map_w
        )

        my = int(
            y1 + 30 + (py / height) * (map_h - 30)
        )

        mx = max(x1 + 5, min(x2 - 5, mx))
        my = max(y1 + 30, min(y2 - 5, my))

        cv2.circle(
            frame,
            (mx, my),
            5,
            (255, 255, 0),
            -1
        )


# =========================
# MAIN LOOP
# =========================

frame_number = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # --------------------------------
    # YOLO TRACKING
    # --------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=[PERSON_CLASS],
        tracker="bytetrack.yaml",
        verbose=False
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()

        ids = results[0].boxes.id.cpu().numpy().astype(int)

        classes = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(
            boxes,
            ids,
            classes
        ):

            if int(cls) != PERSON_CLASS:
                continue

            x1, y1, x2, y2 = map(
                int,
                box
            )

            # Center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # --------------------------------
            # SAVE TRACK HISTORY
            # --------------------------------

            track_history[track_id].append(
                (cx, cy)
            )

            # Keep last 30 positions
            if len(track_history[track_id]) > 30:

                track_history[track_id].pop(0)

            # --------------------------------
            # DRAW TRACKING PATH
            # --------------------------------

            points = track_history[track_id]

            for i in range(1, len(points)):

                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    (255, 255, 0),
                    2
                )

            # --------------------------------
            # COUNTING
            # --------------------------------

            if len(track_history[track_id]) >= 2:

                previous_y = track_history[track_id][-2][1]
                current_y = cy

                # Moving downward = IN
                if (
                    previous_y < LINE_Y
                    and current_y >= LINE_Y
                    and track_id not in counted_ids
                ):

                    in_count += 1

                    inside_ids.add(track_id)

                    counted_ids.add(track_id)

                # Moving upward = OUT
                elif (
                    previous_y > LINE_Y
                    and current_y <= LINE_Y
                    and track_id not in counted_ids
                ):

                    out_count += 1

                    if track_id in inside_ids:
                        inside_ids.remove(track_id)

                    counted_ids.add(track_id)

            # --------------------------------
            # BOUNDING BOX
            # --------------------------------

            draw_corner_box(
                frame,
                x1,
                y1,
                x2,
                y2
            )

            # --------------------------------
            # PERSON LABEL
            # --------------------------------

            draw_label(
                frame,
                "person",
                x1,
                y1
            )

            # --------------------------------
            # CENTER POINT
            # --------------------------------

            cv2.circle(
                frame,
                (cx, cy),
                5,
                (255, 255, 0),
                -1
            )

            cv2.circle(
                frame,
                (cx, cy),
                2,
                (0, 0, 0),
                -1
            )

    # =================================
    # COUNTING LINE
    # =================================

    cv2.line(
        frame,
        (30, LINE_Y),
        (width - 20, LINE_Y),
        (230, 230, 230),
        2
    )

    # IN label
    cv2.rectangle(
        frame,
        (30, LINE_Y - 18),
        (70, LINE_Y),
        (0, 180, 0),
        -1
    )

    cv2.putText(
        frame,
        f"IN {in_count}",
        (34, LINE_Y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1
    )

    # OUT label
    cv2.rectangle(
        frame,
        (width - 45, LINE_Y - 18),
        (width - 5, LINE_Y),
        (0, 120, 255),
        -1
    )

    cv2.putText(
        frame,
        f"{out_count} OUT",
        (width - 43, LINE_Y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (255, 255, 255),
        1
    )

    # =================================
    # VISION POINT
    # =================================

    vx, vy = VISION_POINT

    cv2.rectangle(
        frame,
        (vx - 55, vy - 25),
        (vx + 55, vy),
        (255, 220, 0),
        -1
    )

    cv2.putText(
        frame,
        "VISION POINT",
        (vx - 48, vy - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (0, 0, 0),
        1
    )

    # Draw perspective rays
    for px in range(100, width, 100):

        cv2.line(
            frame,
            (vx, vy),
            (px, LINE_Y),
            (0, 220, 255),
            1
        )

    # =================================
    # DASHBOARD
    # =================================

    draw_dashboard(
        frame,
        frame_number
    )

    # =================================
    # SCENE MAP
    # =================================

    draw_scene_map(frame)

    # =================================
    # DISPLAY
    # =================================

    cv2.imshow(
        "VisionEye - Foot Traffic Analytics",
        frame
    )

    out.write(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# =========================
# RELEASE
# =========================

cap.release()
out.release()

cv2.destroyAllWindows()

print("================================")
print("PROCESSING COMPLETED")
print("================================")
print("People detected :", len(track_history))
print("IN count        :", in_count)
print("OUT count       :", out_count)
print("Output saved as :", OUTPUT_PATH)
