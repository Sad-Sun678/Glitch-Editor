import cv2
import numpy as np
import keyboard
import effects
import time
from effects import change_knob

key_state = {}

#keyboard repeat helper
def key_tapped(key):
    pressed = keyboard.is_pressed(key)

    if key not in key_state:
        key_state[key] = pressed
        return False

    if pressed and not key_state[key]:
        key_state[key] = pressed
        return True

    key_state[key] = pressed
    return False

# ----------------------------
# Camera setup
# ----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)

# ----------------------------
# State
# ----------------------------
prev_gray = None
show_posterize = False

show_motion = False
show_slices = False
zoom_punch_effect = False
show_feedback = False
show_rgb_wave = False
show_motion_smear = False
show_motion_mask_effect = False
show_cycle_mask_effect = False

feedback = None
auto_cycle = False
cycle_interval = 2.0  # seconds per preset
last_cycle_time = time.time()

# ----------
# KNOBS
# ----------
rgb_wave_strength = 12
posterize_strength = 6
motion_smear_strength = 60
feedback_strength = .9
effect_strength_list = [rgb_wave_strength, posterize_strength, motion_smear_strength, feedback_strength]
# ----------------------------
# Main loop
# ----------------------------
preset_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = frame.copy()

    if prev_gray is None:
        prev_gray = gray.copy()
    now = time.time()

    if auto_cycle and (now - last_cycle_time) >= cycle_interval:
        preset_count = (preset_count + 1) % 22  # 0–21
        last_cycle_time = now
    # ------------------------
    # Effects stack (NOT elif)
    # ------------------------

    motion_mask = effects.compute_motion_mask(gray, prev_gray, threshold=25, blur_ksize=5)

    if show_motion:
        # show a visualization (optional)
        output = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    if show_motion_mask_effect:
        output = effects.show_motion_mask_effect(gray,prev_gray,frame)
    if show_cycle_mask_effect:
        output = effects.cycle_masks(gray,prev_gray,frame,preset_count)

    if show_motion_smear:
        output = effects.motion_smear(output, motion_mask, effect_strength_list[2])

    if show_slices:
        output = effects.glitch_slices(output, 12, 40)

    if zoom_punch_effect:
        output = effects.zoom_punch(output, 0.05)
    if show_posterize:
        output = effects.posterize(output, effect_strength_list[1])

    if show_feedback:
        feedback = effects.feedback_loop(output, feedback, 0.9)
        output = feedback
    if show_rgb_wave:
        output = effects.rgb_wave(output, effect_strength_list[0])

    # ------------------------
    # Update previous frame
    # ------------------------
    prev_gray = gray.copy()

    # ------------------------
    # Input toggles
    # ------------------------
    if key_tapped("1"):
        show_motion_mask_effect = not show_motion_mask_effect

    if key_tapped("2"):
        show_slices = not show_slices

    if key_tapped("3"):
        zoom_punch_effect = not zoom_punch_effect

    if key_tapped("4"):
        show_feedback = not show_feedback

    if key_tapped("5"):
        show_rgb_wave = not show_rgb_wave
    if key_tapped("6"):
        show_posterize = not show_posterize
    if key_tapped("7"):
        show_motion_smear = not show_motion_smear
    if key_tapped("8"):
        show_cycle_mask_effect = not show_cycle_mask_effect
    if key_tapped("9"):
        auto_cycle = not auto_cycle
        last_cycle_time = time.time()  # reset timer on toggle
    if key_tapped("]"):
        if preset_count < 21:
            preset_count += 1
        else:
            preset_count = 0
    if key_tapped("["):
        if preset_count > 0:
            preset_count -= 1
        else:
            preset_count = 21

    if keyboard.is_pressed("+"):
        feedback_strength += 0.1

        effect_strength_list = change_knob("up", effect_strength_list)
    if keyboard.is_pressed("-"):
        feedback_strength -= 0.1
        effect_strength_list = change_knob("down", effect_strength_list)
    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("glitch mirror", output)

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()
