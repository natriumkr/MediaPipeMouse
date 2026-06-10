import math
import os
import time
import ctypes
from collections import deque
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import pyautogui


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
FLIP_FRAME = True
INPUT_MODE = os.getenv("INPUT_MODE", "game").lower()  # game | desktop

try:
    import pydirectinput
except Exception:
    pydirectinput = None

if INPUT_MODE == "game" and pydirectinput is None:
    INPUT_MODE = "desktop"


def mouse_move_absolute(x: int, y: int):
    pyautogui.moveTo(x, y)


def mouse_move_relative(dx: int, dy: int):
    if pydirectinput is not None:
        pydirectinput.moveRel(dx, dy, relative=True)
    else:
        pyautogui.moveRel(dx, dy)


def mouse_left_down():
    if INPUT_MODE == "game" and pydirectinput is not None:
        pydirectinput.mouseDown(button="left")
    else:
        pyautogui.mouseDown(button="left")


def mouse_left_up():
    if INPUT_MODE == "game" and pydirectinput is not None:
        pydirectinput.mouseUp(button="left")
    else:
        pyautogui.mouseUp(button="left")


def mouse_left_click():
    if INPUT_MODE == "game" and pydirectinput is not None:
        pydirectinput.click(button="left")
    else:
        pyautogui.click(button="left")


def mouse_right_click():
    if INPUT_MODE == "game" and pydirectinput is not None:
        pydirectinput.click(button="right")
    else:
        pyautogui.click(button="right")


def mouse_scroll(amount: int):
    if INPUT_MODE == "game" and pydirectinput is not None and hasattr(pydirectinput, "scroll"):
        pydirectinput.scroll(amount)
        return
    pyautogui.scroll(amount)


def zoom_scroll(amount: int):
    if INPUT_MODE == "game":
        mouse_scroll(amount)
        return
    pyautogui.keyDown("ctrl")
    pyautogui.scroll(amount)
    pyautogui.keyUp("ctrl")


def to_pixel(landmark, width: int, height: int) -> Tuple[int, int]:
    x = max(0, min(width - 1, int(landmark.x * width)))
    y = max(0, min(height - 1, int(landmark.y * height)))
    return x, y


def distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def smoothing_factor(dt: float, cutoff: float) -> float:
    if dt <= 0:
        return 1.0
    r = 2.0 * math.pi * cutoff * dt
    return r / (r + 1.0)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.1, beta: float = 0.06, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.prev_t: Optional[float] = None
        self.prev_x: Optional[float] = None
        self.prev_dx = 0.0

    def filter(self, x: float, now_t: float) -> float:
        if self.prev_t is None or self.prev_x is None:
            self.prev_t = now_t
            self.prev_x = x
            self.prev_dx = 0.0
            return x

        dt = max(1e-3, now_t - self.prev_t)
        dx = (x - self.prev_x) / dt

        a_d = smoothing_factor(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.prev_dx

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.prev_x

        self.prev_t = now_t
        self.prev_x = x_hat
        self.prev_dx = dx_hat
        return x_hat


def is_pinch(thumb: Tuple[int, int], finger: Tuple[int, int], hand_scale: float, ratio: float) -> bool:
    return distance(thumb, finger) < (hand_scale * ratio)


def is_open_palm(lm) -> bool:
    # Y is downwards in image coordinates.
    pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    extended = 0
    for tip, pip in pairs:
        if lm[tip].y < lm[pip].y:
            extended += 1
    return extended >= 3


def is_thumb_pinky_only(lm) -> bool:
    idx_folded = lm[8].y > lm[6].y
    mid_folded = lm[12].y > lm[10].y
    ring_folded = lm[16].y > lm[14].y
    pinky_extended = lm[20].y < lm[18].y
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    index_mcp = lm[5]
    thumb_open = math.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y) > math.hypot(thumb_ip.x - index_mcp.x, thumb_ip.y - index_mcp.y) * 1.15
    return idx_folded and mid_folded and ring_folded and pinky_extended and thumb_open


def lock_windows_screen():
    try:
        ctypes.windll.user32.LockWorkStation()
    except Exception:
        pass


def close_active_window():
    try:
        if INPUT_MODE == "game" and pydirectinput is not None:
            pydirectinput.keyDown("alt")
            pydirectinput.press("f4")
            pydirectinput.keyUp("alt")
        else:
            pyautogui.hotkey("alt", "f4")
    except Exception:
        pass


def wrist_point_px(lm, width: int, height: int) -> Tuple[int, int]:
    return to_pixel(lm[0], width, height)


def assign_primary_hands(hands, tracked_wrists, frame_w: int, frame_h: int):
    if not hands or not tracked_wrists:
        return [], tracked_wrists

    candidates = []
    for i, hand in enumerate(hands):
        w = wrist_point_px(hand["landmarks"], frame_w, frame_h)
        candidates.append((i, w))

    max_dist = math.hypot(frame_w, frame_h) * 0.45
    assigned = []
    used_hand_indices = set()
    new_wrists = list(tracked_wrists)

    for slot in range(min(2, len(tracked_wrists))):
        best_idx = None
        best_dist = None
        for hand_idx, w in candidates:
            if hand_idx in used_hand_indices:
                continue
            d = distance(w, tracked_wrists[slot])
            if best_dist is None or d < best_dist:
                best_dist = d
                best_idx = hand_idx
        if best_idx is not None and best_dist is not None and best_dist < max_dist:
            used_hand_indices.add(best_idx)
            assigned.append(hands[best_idx])
            new_wrists[slot] = wrist_point_px(hands[best_idx]["landmarks"], frame_w, frame_h)

    return assigned, new_wrists


def build_solution_backend():
    try:
        mp_solutions = mp.solutions
    except AttributeError:
        return None

    hands = mp_solutions.hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return ("solutions", hands, mp_solutions)


def build_tasks_backend():
    model_path = os.getenv("HAND_LANDMARKER_MODEL", "hand_landmarker.task")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Missing hand_landmarker.task. Put the model in project root "
            "or set HAND_LANDMARKER_MODEL to the model path."
        )

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = HandLandmarker.create_from_options(options)
    return ("tasks", landmarker, None)


def build_backend():
    backend = build_solution_backend()
    if backend is not None:
        return backend
    return build_tasks_backend()


def get_landmarks(backend_type: str, backend_obj, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if backend_type == "solutions":
        result = backend_obj.process(rgb)
        if not result.multi_hand_landmarks:
            return []
        handedness = result.multi_handedness or []
        out = []
        for i, hand in enumerate(result.multi_hand_landmarks):
            label = None
            if i < len(handedness) and handedness[i].classification:
                label = handedness[i].classification[0].label
            out.append({"landmarks": hand.landmark, "label": label})
        return out
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = backend_obj.detect(mp_image)
    hand_landmarks = result.hand_landmarks or []
    handedness = result.handedness or []
    out = []
    for i, lm in enumerate(hand_landmarks):
        label = None
        if i < len(handedness) and handedness[i]:
            label = handedness[i][0].category_name
        out.append({"landmarks": lm, "label": label})
    return out


def draw_hands(frame, backend_type: str, mp_solutions, landmarks_list):
    if backend_type == "solutions":
        for hand in landmarks_list:
            lm = hand["landmarks"]
            class _Hand:
                pass

            hand_obj = _Hand()
            hand_obj.landmark = lm
            mp_solutions.drawing_utils.draw_landmarks(
                frame,
                hand_obj,
                mp_solutions.hands.HAND_CONNECTIONS,
            )
    else:
        for hand in landmarks_list:
            lm = hand["landmarks"]
            for p in lm:
                px, py = to_pixel(p, frame.shape[1], frame.shape[0])
                cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return
    if pydirectinput is None:
        print("pydirectinput not found. Running in desktop input mode.")

    try:
        backend_type, backend_obj, mp_solutions = build_backend()
    except Exception as e:
        print(f"Init failed: {e}")
        cap.release()
        return

    screen_w, screen_h = pyautogui.size()
    cursor_gain = 1.5
    sent_x: Optional[float] = None
    sent_y: Optional[float] = None
    game_sensitivity = 0.55
    # Adaptive smoothing: stable at low speed, responsive at high speed.
    x_filter = OneEuroFilter(min_cutoff=1.15, beta=0.055, d_cutoff=1.0)
    y_filter = OneEuroFilter(min_cutoff=1.15, beta=0.055, d_cutoff=1.0)

    left_pinched_prev = False
    right_pinched_prev = False
    drag_active = False
    click_cooldown = 0.2
    last_left_click = 0.0
    last_right_click = 0.0

    prev_zoom_dist: Optional[float] = None
    last_zoom_time = 0.0
    zoom_cooldown = 0.06
    prev_scroll_y: Optional[float] = None
    last_scroll_time = 0.0
    scroll_cooldown = 0.03
    calibrated = False
    calibrate_count = 0
    calibrate_need = 12
    tracked_wrists = []
    snap_times = deque()
    last_snap_time = 0.0
    prev_thumb_middle_norm: Optional[float] = None
    prev_snap_sample_time: Optional[float] = None
    lock_pose_start_time: Optional[float] = None
    last_gesture_lock_time = 0.0
    gesture_lock_cooldown = 8.0
    read_fail_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                read_fail_count += 1
                if read_fail_count % 20 == 0:
                    cap.release()
                    time.sleep(0.15)
                    cap = cv2.VideoCapture(0)
                time.sleep(0.03)
                continue
            read_fail_count = 0

            if FLIP_FRAME:
                frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape
            mode = "No hand"
            safety_text = "Lock gesture: Thumb+Pinky only hold 5s"
            now = time.time()

            raw_hands = get_landmarks(backend_type, backend_obj, frame)
            hand_count = len(raw_hands)
            hands = []

            if not calibrated:
                open_hands = [h for h in raw_hands if is_open_palm(h["landmarks"])]
                if len(open_hands) >= 2:
                    open_hands.sort(key=lambda h: wrist_point_px(h["landmarks"], frame_w, frame_h)[0])
                    candidate = open_hands[:2]
                    calibrate_count += 1
                    mode = f"Calibrating {calibrate_count}/{calibrate_need}"
                    if calibrate_count >= calibrate_need:
                        tracked_wrists = [
                            wrist_point_px(candidate[0]["landmarks"], frame_w, frame_h),
                            wrist_point_px(candidate[1]["landmarks"], frame_w, frame_h),
                        ]
                        calibrated = True
                        mode = "Calibrated"
                else:
                    calibrate_count = 0
                    mode = "Show both open palms"
                hands = []
            else:
                hands, tracked_wrists = assign_primary_hands(raw_hands, tracked_wrists, frame_w, frame_h)
                hand_count = len(hands)

            if calibrated and hands:
                expected_right_label = "Left" if FLIP_FRAME else "Right"
                right_hands = [h for h in hands if h.get("label") == expected_right_label]
                if right_hands:
                    primary_hand = right_hands[0]
                else:
                    primary_hand = None

                if primary_hand is None:
                    prev_zoom_dist = None
                    prev_scroll_y = None
                    lock_pose_start_time = None
                    left_pinched_prev = False
                    right_pinched_prev = False
                    x_filter.prev_t = None
                    y_filter.prev_t = None
                    if drag_active:
                        mouse_left_up()
                        drag_active = False
                    mode = "Right hand only"
                    draw_hands(frame, backend_type, mp_solutions, raw_hands)
                    cv2.putText(
                        frame,
                        f"Backend: {backend_type} | Hands(raw/primary): {len(raw_hands)}/{len(hands)} | Mode: {mode}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (50, 220, 50),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        "Right only | Thumb+Pinky: scroll | 2 hands Thumb+Index: zoom | 3 snaps/3s close | ESC",
                        (10, frame_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (220, 220, 220),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("MediaPipe Gesture Mouse", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

                lm = primary_hand["landmarks"]
                if is_thumb_pinky_only(lm):
                    if lock_pose_start_time is None:
                        lock_pose_start_time = now
                    hold = now - lock_pose_start_time
                    safety_text = f"Lock in {max(0.0, 5.0 - hold):.1f}s"
                    if hold >= 5.0 and (now - last_gesture_lock_time) >= gesture_lock_cooldown:
                        lock_windows_screen()
                        last_gesture_lock_time = now
                        lock_pose_start_time = now
                        safety_text = "Locked by gesture"
                else:
                    lock_pose_start_time = None

                thumb_tip = to_pixel(lm[4], frame_w, frame_h)
                idx_tip = to_pixel(lm[8], frame_w, frame_h)
                mid_tip = to_pixel(lm[12], frame_w, frame_h)
                ring_tip = to_pixel(lm[16], frame_w, frame_h)
                pinky_tip = to_pixel(lm[20], frame_w, frame_h)
                wrist = to_pixel(lm[0], frame_w, frame_h)
                middle_mcp = to_pixel(lm[9], frame_w, frame_h)
                hand_scale = max(30.0, distance(wrist, middle_mcp))

                # Snap heuristic: fast thumb-middle closing motion.
                thumb_middle_dist = distance(thumb_tip, mid_tip)
                thumb_middle_norm = thumb_middle_dist / hand_scale
                snap_pose = lm[12].y < lm[10].y
                if prev_thumb_middle_norm is not None and prev_snap_sample_time is not None:
                    dt = max(1e-3, now - prev_snap_sample_time)
                    closing_speed = (prev_thumb_middle_norm - thumb_middle_norm) / dt
                    if (
                        snap_pose
                        and thumb_middle_norm < 0.42
                        and closing_speed > 1.05
                        and (now - last_snap_time) > 0.10
                    ):
                        snap_times.append(now)
                        last_snap_time = now
                        mode = f"Snap {len(snap_times)}"
                prev_thumb_middle_norm = thumb_middle_norm
                prev_snap_sample_time = now

                while snap_times and (now - snap_times[0]) > 3.0:
                    snap_times.popleft()
                if len(snap_times) >= 3:
                    if drag_active:
                        mouse_left_up()
                        drag_active = False
                    mode = "Close Window"
                    close_active_window()
                    snap_times.clear()

                thumb_pinky_pinched = is_pinch(thumb_tip, pinky_tip, hand_scale, ratio=0.38)
                zoom_active = False
                lm2 = None
                if len(hands) >= 2:
                    other_hands = [h for h in hands if h is not primary_hand]
                    if other_hands:
                        lm2 = other_hands[0]["landmarks"]
                        thumb2 = to_pixel(lm2[4], frame_w, frame_h)
                        idx2 = to_pixel(lm2[8], frame_w, frame_h)
                        wrist2 = to_pixel(lm2[0], frame_w, frame_h)
                        mcp2 = to_pixel(lm2[9], frame_w, frame_h)
                        scale2 = max(30.0, distance(wrist2, mcp2))
                        zoom_active = is_pinch(thumb_tip, idx_tip, hand_scale, ratio=0.36) and is_pinch(thumb2, idx2, scale2, ratio=0.36)

                if zoom_active and lm2 is not None:
                    if drag_active:
                        mouse_left_up()
                        drag_active = False
                    left_pinched_prev = False
                    right_pinched_prev = False
                    prev_scroll_y = None

                    idx1 = to_pixel(lm[8], frame_w, frame_h)
                    idx2 = to_pixel(lm2[8], frame_w, frame_h)
                    zoom_dist = distance(idx1, idx2)
                    if prev_zoom_dist is None:
                        prev_zoom_dist = zoom_dist
                        mode = "Zoom Ready"
                    else:
                        delta = zoom_dist - prev_zoom_dist
                        threshold = max(8.0, frame_w * 0.012)
                        if (now - last_zoom_time) > zoom_cooldown:
                            if delta > threshold:
                                zoom_scroll(240)
                                mode = "Zoom In"
                                last_zoom_time = now
                            elif delta < -threshold:
                                zoom_scroll(-240)
                                mode = "Zoom Out"
                                last_zoom_time = now
                            else:
                                mode = "Zoom Ready"
                        prev_zoom_dist = zoom_dist

                elif thumb_pinky_pinched:
                    prev_zoom_dist = None
                    if drag_active:
                        mouse_left_up()
                        drag_active = False
                    left_pinched_prev = False
                    right_pinched_prev = False

                    if prev_scroll_y is None:
                        prev_scroll_y = idx_tip[1]
                        mode = "Scroll Ready"
                    else:
                        if (now - last_scroll_time) > scroll_cooldown:
                            dy = prev_scroll_y - idx_tip[1]
                            if abs(dy) > 6:
                                amount = int(dy * 6)
                                mouse_scroll(amount)
                                mode = "Scroll Up" if amount > 0 else "Scroll Down"
                                last_scroll_time = now
                            else:
                                mode = "Scroll Ready"
                        prev_scroll_y = idx_tip[1]

                else:
                    prev_zoom_dist = None
                    prev_scroll_y = None

                    target_x = (idx_tip[0] / frame_w * screen_w - (screen_w / 2)) * cursor_gain + (screen_w / 2)
                    target_y = (idx_tip[1] / frame_h * screen_h - (screen_h / 2)) * cursor_gain + (screen_h / 2)
                    target_x = max(0, min(screen_w - 1, target_x))
                    target_y = max(0, min(screen_h - 1, target_y))
                    filtered_x = x_filter.filter(target_x, now)
                    filtered_y = y_filter.filter(target_y, now)

                    if INPUT_MODE == "game":
                        if sent_x is None or sent_y is None:
                            sent_x, sent_y = filtered_x, filtered_y
                        dx = int((filtered_x - sent_x) * game_sensitivity)
                        dy = int((filtered_y - sent_y) * game_sensitivity)
                        if abs(dx) >= 1 or abs(dy) >= 1:
                            mouse_move_relative(dx, dy)
                            sent_x += dx / game_sensitivity
                            sent_y += dy / game_sensitivity
                    else:
                        mouse_move_absolute(int(filtered_x), int(filtered_y))
                        sent_x, sent_y = filtered_x, filtered_y
                    mode = "Move"

                    left_pinched = is_pinch(thumb_tip, idx_tip, hand_scale, ratio=0.30)
                    right_pinched = is_pinch(thumb_tip, mid_tip, hand_scale, ratio=0.30)
                    drag_dist = distance(thumb_tip, ring_tip)
                    drag_start_th = hand_scale * 0.50
                    drag_end_th = hand_scale * 0.62

                    if not drag_active and drag_dist < drag_start_th:
                        mouse_left_down()
                        drag_active = True

                    if drag_active:
                        mode = "Drag"
                        if drag_dist > drag_end_th:
                            mouse_left_up()
                            drag_active = False
                    else:
                        if left_pinched and not left_pinched_prev and (now - last_left_click) > click_cooldown:
                            mouse_left_click()
                            last_left_click = now
                            mode = "Left Click"
                        elif right_pinched and not right_pinched_prev and (now - last_right_click) > click_cooldown:
                            mouse_right_click()
                            last_right_click = now
                            mode = "Right Click"

                    left_pinched_prev = left_pinched
                    right_pinched_prev = right_pinched
            elif calibrated:
                prev_zoom_dist = None
                prev_scroll_y = None
                lock_pose_start_time = None
                prev_thumb_middle_norm = None
                prev_snap_sample_time = None
                left_pinched_prev = False
                right_pinched_prev = False
                x_filter.prev_t = None
                y_filter.prev_t = None
                if drag_active:
                    mouse_left_up()
                    drag_active = False

            draw_hands(frame, backend_type, mp_solutions, raw_hands)

            cv2.putText(
                frame,
                f"Backend: {backend_type} | Hands(raw/primary): {len(raw_hands)}/{len(hands) if calibrated else 0} | Mode: {mode}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 220, 50),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Right only | Thumb+Pinky: scroll | 2 hands Thumb+Index: zoom | 3 snaps/3s close",
                (10, frame_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                safety_text,
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 120),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("MediaPipe Gesture Mouse", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        if drag_active:
            mouse_left_up()
        backend_obj.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
