import cv2
import numpy as np
import keyboard
import effects
import time
from effects import change_knob

key_state = {}

# keyboard repeat helper
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

# ----------------------------
# State
# ----------------------------
prev_gray = None

show_motion = False
show_slices = False
zoom_punch_effect = False
show_feedback = False
show_rgb_wave = False
show_motion_smear = False
show_motion_mask_effect = False
show_cycle_mask_effect = False
show_datamosh_vector_effect = False
show_pixel_sort_effect = False
alternate_frames = False
show_posterize = False

feedback = None

auto_cycle = False
cycle_interval = 2.0
last_cycle_time = time.time()

transitioning = False
transition_start = 0.0
transition_duration = 1.0
from_preset = 0
to_preset = 0

# ----------------------------
# Knobs
# ----------------------------
rgb_wave_strength = 12
posterize_strength = 6
motion_smear_strength = 60
feedback_strength = 0.9
effect_strength_list = [
    rgb_wave_strength,
    posterize_strength,
    motion_smear_strength,
    feedback_strength
]

# ----------------------------
# Main loop
# ----------------------------
preset_count = 0
frame_count = 0

while True:
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = frame.copy()

    if prev_gray is None:
        prev_gray = gray.copy()

    # ----------------------------
    # Motion
    # ----------------------------
    motion_mask = effects.compute_motion_mask(
        gray, prev_gray, threshold=25, blur_ksize=5
    )

    # ----------------------------
    # Auto preset cycling
    # ----------------------------
    now = time.time()
    blend_t = 1.0

    if transitioning:
        blend_t = (now - transition_start) / transition_duration
        blend_t = np.clip(blend_t, 0.0, 1.0)
        blend_t = blend_t * blend_t * (3 - 2 * blend_t)

        if blend_t >= 1.0:
            transitioning = False
            preset_count = to_preset

    if auto_cycle and not transitioning and (now - last_cycle_time) >= cycle_interval:
        from_preset = preset_count
        to_preset = (preset_count + 1) % 22
        transitioning = True
        transition_start = now
        last_cycle_time = now

    # ----------------------------
    # Effects stack
    # ----------------------------
    if show_motion:
        output = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

    if show_pixel_sort_effect:
        if alternate_frames:
            if frame_count % 2 == 0:
                output = effects.pixel_sort_horizontal(
                    output, motion_mask, thresh=50, min_len=16
                )
            else:
                output = effects.pixel_sort_vertical(
                    output, motion_mask, thresh=50, min_len=16
                )
        else:
            output = effects.pixel_sort_horizontal(
                output, motion_mask, thresh=50, min_len=16
            )

    if show_datamosh_vector_effect:
        output = effects.datamosh_vector(
            output, prev_gray, motion_mask, 20
        )

    if show_motion_mask_effect:
        output = effects.show_motion_mask_effect(
            gray, prev_gray, output
        )

    if show_cycle_mask_effect:
        if transitioning:
            output = effects.cycle_masks(
                gray, prev_gray, output,
                from_preset, to_preset, blend_t
            )
        else:
            output = effects.cycle_masks(
                gray, prev_gray, output,
                preset_count, preset_count, 1.0
            )

    if show_motion_smear:
        output = effects.motion_smear(
            output, motion_mask, effect_strength_list[2]
        )

    if show_slices:
        output = effects.glitch_slices(output, 12, 40)

    if zoom_punch_effect:
        output = effects.zoom_punch(output, 0.05)

    if show_posterize:
        output = effects.posterize(output, effect_strength_list[1])

    if show_feedback:
        feedback = effects.feedback_loop(output, feedback, feedback_strength)
        output = feedback

    if show_rgb_wave:
        output = effects.rgb_wave(output, effect_strength_list[0])

    # ----------------------------
    # Update previous frame
    # ----------------------------
    prev_gray = gray.copy()

    # ----------------------------
    # Input toggles
    # ----------------------------
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
        show_datamosh_vector_effect = not show_datamosh_vector_effect
    if key_tapped("p"):
        show_pixel_sort_effect = not show_pixel_sort_effect
    if key_tapped("/"):
        alternate_frames = not alternate_frames
        print(f"alternate_frames = {alternate_frames}")
    if key_tapped("c"):
        auto_cycle = not auto_cycle
        last_cycle_time = time.time()

    if key_tapped("]"):
        preset_count = (preset_count + 1) % 22
    if key_tapped("["):
        preset_count = (preset_count - 1) % 22

    if keyboard.is_pressed("+"):
        effect_strength_list = change_knob("up", effect_strength_list)
    if keyboard.is_pressed("-"):
        effect_strength_list = change_knob("down", effect_strength_list)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cv2.imshow("glitch mirror", output)

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()