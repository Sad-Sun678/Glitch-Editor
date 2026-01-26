import cv2 as cv
from collections import deque
import numpy as np
frame_buffer = deque(maxlen=12)


def rescale_frame(frame,scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
def compute_motion_mask(gray, prev_gray, threshold=25, blur_ksize=5):
    diff = cv.absdiff(gray, prev_gray)
    _, motion_mask = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
    motion_mask = cv.GaussianBlur(motion_mask, (blur_ksize, blur_ksize), 0)
    return motion_mask

def show_motion_mask_effect(gray, prev_gray, frame):
    diff = cv.absdiff(gray, prev_gray)
    motion_strength = cv.normalize(diff, None, 0, 255, cv.NORM_MINMAX)

    _, motion_mask = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
    motion_mask = cv.GaussianBlur(motion_mask, (5, 5), 0)

    color_map = cv.applyColorMap(motion_strength, cv.COLORMAP_JET)

    output = frame.copy()
    output[motion_strength > 25] = color_map[motion_strength > 25]

    return output
def brightness(row):
    # row: (N, 3) BGR
    return row[:, 0] * 0.114 + row[:, 1] * 0.587 + row[:, 2] * 0.299
def pixel_sort_horizontal(frame, motion_mask, thresh=40, min_len=12):
    h, w, _ = frame.shape
    output = frame.copy()

    for y in range(h):
        mask_row = motion_mask[y] > 0
        x = 0

        while x < w:
            if not mask_row[x]:
                x += 1
                continue

            # start of motion segment
            start = x
            while x < w and mask_row[x]:
                x += 1
            end = x

            if end - start < min_len:
                continue

            segment = output[y, start:end]

            # brightness threshold gating
            b = brightness(segment)
            valid = b > thresh
            if valid.sum() < min_len:
                continue

            sortable = segment[valid]
            order = np.argsort(brightness(sortable))
            sortable_sorted = sortable[order]

            segment[valid] = sortable_sorted
            output[y, start:end] = segment

    return output
def pixel_sort_vertical(frame, motion_mask, thresh=40, min_len=12):
    h, w, _ = frame.shape
    output = frame.copy()

    for x in range(w):
        mask_col = motion_mask[:, x] > 0
        y = 0

        while y < h:
            if not mask_col[y]:
                y += 1
                continue

            # start of motion segment
            start = y
            while y < h and mask_col[y]:
                y += 1
            end = y

            if end - start < min_len:
                continue

            segment = output[start:end, x]

            # brightness threshold gating
            b = brightness(segment)
            valid = b > thresh
            if valid.sum() < min_len:
                continue

            sortable = segment[valid]
            order = np.argsort(brightness(sortable))
            sortable_sorted = sortable[order]

            segment[valid] = sortable_sorted
            output[start:end, x] = segment

    return output
def cycle_masks(gray, prev_gray, frame, preset_a, preset_b, t):
    presets = {
        0: cv.COLORMAP_AUTUMN,
        1: cv.COLORMAP_BONE,
        2: cv.COLORMAP_JET,
        3: cv.COLORMAP_WINTER,
        4: cv.COLORMAP_RAINBOW,
        5: cv.COLORMAP_OCEAN,
        6: cv.COLORMAP_SUMMER,
        7: cv.COLORMAP_SPRING,
        8: cv.COLORMAP_COOL,
        9: cv.COLORMAP_HSV,
        10: cv.COLORMAP_PINK,
        11: cv.COLORMAP_HOT,
        12: cv.COLORMAP_PARULA,
        13: cv.COLORMAP_MAGMA,
        14: cv.COLORMAP_INFERNO,
        15: cv.COLORMAP_PLASMA,
        16: cv.COLORMAP_VIRIDIS,
        17: cv.COLORMAP_CIVIDIS,
        18: cv.COLORMAP_TWILIGHT,
        19: cv.COLORMAP_TWILIGHT_SHIFTED,
        20: cv.COLORMAP_TURBO,
        21: cv.COLORMAP_DEEPGREEN,
    }
    diff = cv.absdiff(gray, prev_gray)
    motion_strength = cv.normalize(diff, None, 0, 255, cv.NORM_MINMAX)
    motion_strength = motion_strength.astype(np.uint8)

    _, motion_mask = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)

    color_a = cv.applyColorMap(motion_strength, presets[preset_a])
    color_b = cv.applyColorMap(motion_strength, presets[preset_b])

    # crossfade
    color_map = cv.addWeighted(color_a, 1 - t, color_b, t, 0)

    output = frame.copy()
    output[motion_strength > 25] = color_map[motion_strength > 25]

    return output

def delay_echo(frame, buffer, alpha=0.6):
    if len(buffer) > 0:
        echo = buffer[0]
        frame = cv.addWeighted(frame, alpha, echo, 1 - alpha, 0)
    buffer.append(frame.copy())
    return frame
def glitch_slices(frame, slice_height=12, max_shift=40):
    h, w, _ = frame.shape
    output = frame.copy()

    for y in range(0, h, slice_height):
        if np.random.rand() < 0.15:
            shift = np.random.randint(-max_shift, max_shift)
            output[y:y+slice_height] = np.roll(
                output[y:y+slice_height],
                shift,
                axis=1
            )
    return output
def zoom_punch(frame, strength=0.05):
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    M = cv.getRotationMatrix2D((cx, cy), 0, 1 + strength)
    return cv.warpAffine(frame, M, (w, h))
def feedback_loop(frame, feedback, decay=0.97):
    if feedback is None:
        feedback = frame.copy()

    feedback = cv.addWeighted(frame, 1 - decay, feedback, decay, 0)
    return feedback
def motion_trails(frame, motion_mask, trail_buffer, decay=0.92):
    if trail_buffer is None:
        trail_buffer = frame.copy()

    trail_buffer = cv.addWeighted(trail_buffer, decay, frame, 1 - decay, 0)
    trail_buffer[motion_mask == 0] = frame[motion_mask == 0]

    return trail_buffer, trail_buffer
def rgb_wave(frame, amount=10):
    b, g, r = cv.split(frame)

    h, w = b.shape

    # Horizontal wave offsets
    for y in range(h):
        offset = int(np.sin(y * 0.05) * amount)
        r[y] = np.roll(r[y], offset)
        b[y] = np.roll(b[y], -offset)

    return cv.merge((b, g, r))
def posterize(frame, levels=6):
    # levels = how many color steps per channel
    levels = max(2, levels)

    step = 256 // levels
    poster = (frame // step) * step

    return poster.astype(np.uint8)
def motion_smear(frame, motion_mask, strength=15):
    output = frame.copy()

    smear = np.roll(frame, strength, axis=1)
    output[motion_mask > 0] = smear[motion_mask > 0]

    return output

def change_knob(direction,targets):
    if direction == "up":
        new_values = []
        for knob in targets:
            knob += 1
            new_values.append(knob)
    else:
        new_values = []
        for knob in targets:
            knob -= 1
            new_values.append(knob)

    return new_values

def datamosh_vector(frame, prev_gray, motion_mask, strength=8):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(
        prev_gray, gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    h, w = gray.shape
    output = frame.copy()

    # grid of coordinates
    ys, xs = np.mgrid[0:h, 0:w]

    # flow vectors
    dx = flow[..., 0]
    dy = flow[..., 1]

    # source coordinates
    src_x = np.clip((xs - dx * strength).astype(np.int32), 0, w - 1)
    src_y = np.clip((ys - dy * strength).astype(np.int32), 0, h - 1)

    # apply only where motion exists
    mask = motion_mask > 0
    output[mask] = frame[src_y[mask], src_x[mask]]

    return output