import cv2 as cv
from collections import deque
import numpy as np

frame_buffer = deque(maxlen=12)


def rescale_frame(frame, scale=0.75):
    """Resize the frame by a scale factor."""
    new_width = int(frame.shape[1] * scale)
    new_height = int(frame.shape[0] * scale)
    dimensions = (new_width, new_height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def compute_motion_mask(current_gray, previous_gray, threshold=25, blur_kernel_size=5):
    """
    Compute a mask highlighting areas of motion between two grayscale frames.
    Returns a blurred binary mask where white indicates motion.
    """
    frame_difference = cv.absdiff(current_gray, previous_gray)
    _, binary_motion_mask = cv.threshold(frame_difference, threshold, 255, cv.THRESH_BINARY)
    blurred_motion_mask = cv.GaussianBlur(binary_motion_mask, (blur_kernel_size, blur_kernel_size), 0)
    return blurred_motion_mask


def show_motion_mask_effect(current_gray, previous_gray, frame):
    """
    Visualize motion by overlaying a colored heat map on areas of movement.
    Uses JET colormap to show motion intensity.
    """
    frame_difference = cv.absdiff(current_gray, previous_gray)
    normalized_motion = cv.normalize(frame_difference, None, 0, 255, cv.NORM_MINMAX)

    _, binary_motion_mask = cv.threshold(frame_difference, 25, 255, cv.THRESH_BINARY)
    blurred_motion_mask = cv.GaussianBlur(binary_motion_mask, (5, 5), 0)

    colored_motion_map = cv.applyColorMap(normalized_motion, cv.COLORMAP_JET)

    output_frame = frame.copy()
    motion_areas = normalized_motion > 25
    output_frame[motion_areas] = colored_motion_map[motion_areas]

    return output_frame


def calculate_pixel_brightness(pixel_row):
    """
    Calculate perceived brightness for a row of BGR pixels.
    Uses standard luminance weights: B=0.114, G=0.587, R=0.299
    """
    blue_channel = pixel_row[:, 0]
    green_channel = pixel_row[:, 1]
    red_channel = pixel_row[:, 2]
    return blue_channel * 0.114 + green_channel * 0.587 + red_channel * 0.299


def pixel_sort_horizontal(frame, motion_mask, brightness_threshold=40, minimum_segment_length=12):
    """
    Sort pixels horizontally within motion areas by brightness.
    Creates a glitchy, flowing effect in regions with movement.
    """
    frame_height, frame_width, _ = frame.shape
    output_frame = frame.copy()

    for row_index in range(frame_height):
        row_has_motion = motion_mask[row_index] > 0
        column_index = 0

        while column_index < frame_width:
            if not row_has_motion[column_index]:
                column_index += 1
                continue

            # found start of a motion segment
            segment_start = column_index
            while column_index < frame_width and row_has_motion[column_index]:
                column_index += 1
            segment_end = column_index

            segment_length = segment_end - segment_start
            if segment_length < minimum_segment_length:
                continue

            pixel_segment = output_frame[row_index, segment_start:segment_end]

            # filter by brightness threshold
            segment_brightness = calculate_pixel_brightness(pixel_segment)
            pixels_above_threshold = segment_brightness > brightness_threshold
            if pixels_above_threshold.sum() < minimum_segment_length:
                continue

            # sort the qualifying pixels by brightness
            sortable_pixels = pixel_segment[pixels_above_threshold]
            brightness_order = np.argsort(calculate_pixel_brightness(sortable_pixels))
            sorted_pixels = sortable_pixels[brightness_order]

            pixel_segment[pixels_above_threshold] = sorted_pixels
            output_frame[row_index, segment_start:segment_end] = pixel_segment

    return output_frame


def pixel_sort_vertical(frame, motion_mask, brightness_threshold=40, minimum_segment_length=12):
    """
    Sort pixels vertically within motion areas by brightness.
    Creates a dripping, cascading glitch effect.
    """
    frame_height, frame_width, _ = frame.shape
    output_frame = frame.copy()

    for column_index in range(frame_width):
        column_has_motion = motion_mask[:, column_index] > 0
        row_index = 0

        while row_index < frame_height:
            if not column_has_motion[row_index]:
                row_index += 1
                continue

            # found start of a motion segment
            segment_start = row_index
            while row_index < frame_height and column_has_motion[row_index]:
                row_index += 1
            segment_end = row_index

            segment_length = segment_end - segment_start
            if segment_length < minimum_segment_length:
                continue

            pixel_segment = output_frame[segment_start:segment_end, column_index]

            # filter by brightness threshold
            segment_brightness = calculate_pixel_brightness(pixel_segment)
            pixels_above_threshold = segment_brightness > brightness_threshold
            if pixels_above_threshold.sum() < minimum_segment_length:
                continue

            # sort the qualifying pixels by brightness
            sortable_pixels = pixel_segment[pixels_above_threshold]
            brightness_order = np.argsort(calculate_pixel_brightness(sortable_pixels))
            sorted_pixels = sortable_pixels[brightness_order]

            pixel_segment[pixels_above_threshold] = sorted_pixels
            output_frame[segment_start:segment_end, column_index] = pixel_segment

    return output_frame


def cycle_masks(current_gray, previous_gray, frame, preset_index_a, preset_index_b, blend_factor):
    """
    Apply color-mapped motion visualization with smooth transitions between colormap presets.
    Crossfades between two colormaps based on the blend factor (0-1).
    """
    colormap_presets = {
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

    frame_difference = cv.absdiff(current_gray, previous_gray)
    normalized_motion = cv.normalize(frame_difference, None, 0, 255, cv.NORM_MINMAX)
    normalized_motion = normalized_motion.astype(np.uint8)

    _, binary_motion_mask = cv.threshold(frame_difference, 25, 255, cv.THRESH_BINARY)

    colored_map_a = cv.applyColorMap(normalized_motion, colormap_presets[preset_index_a])
    colored_map_b = cv.applyColorMap(normalized_motion, colormap_presets[preset_index_b])

    # crossfade between the two colormaps
    blended_colormap = cv.addWeighted(colored_map_a, 1 - blend_factor, colored_map_b, blend_factor, 0)

    output_frame = frame.copy()
    motion_areas = normalized_motion > 25
    output_frame[motion_areas] = blended_colormap[motion_areas]

    return output_frame


def delay_echo(frame, echo_buffer, blend_alpha=0.6):
    """
    Create a temporal echo effect by blending with buffered past frames.
    Creates ghostly trails from movement.
    """
    if len(echo_buffer) > 0:
        oldest_frame = echo_buffer[0]
        frame = cv.addWeighted(frame, blend_alpha, oldest_frame, 1 - blend_alpha, 0)
    echo_buffer.append(frame.copy())
    return frame


def glitch_slices(frame, slice_height=12, max_horizontal_shift=40):
    """
    Randomly shift horizontal slices of the image for a glitchy look.
    Each slice has a random chance to be displaced.
    """
    frame_height, frame_width, _ = frame.shape
    output_frame = frame.copy()

    for slice_top in range(0, frame_height, slice_height):
        should_glitch = np.random.rand() < 0.15
        if should_glitch:
            random_shift = np.random.randint(-max_horizontal_shift, max_horizontal_shift)
            output_frame[slice_top:slice_top + slice_height] = np.roll(
                output_frame[slice_top:slice_top + slice_height],
                random_shift,
                axis=1
            )
    return output_frame


def zoom_punch(frame, zoom_strength=0.05):
    """
    Apply a subtle zoom from center, creating a 'punch' effect.
    """
    frame_height, frame_width, _ = frame.shape
    center_x, center_y = frame_width // 2, frame_height // 2

    zoom_matrix = cv.getRotationMatrix2D((center_x, center_y), 0, 1 + zoom_strength)
    return cv.warpAffine(frame, zoom_matrix, (frame_width, frame_height))


def feedback_loop(current_frame, previous_feedback, decay_rate=0.97):
    """
    Create a video feedback effect by blending current frame with accumulated past.
    Higher decay = longer trails, more ghosting.
    """
    if previous_feedback is None:
        previous_feedback = current_frame.copy()

    blended_feedback = cv.addWeighted(current_frame, 1 - decay_rate, previous_feedback, decay_rate, 0)
    return blended_feedback


def motion_trails(frame, motion_mask, trail_buffer, decay_rate=0.92):
    """
    Create motion trails that persist in areas of movement.
    Static areas update immediately, moving areas leave trails.
    """
    if trail_buffer is None:
        trail_buffer = frame.copy()

    trail_buffer = cv.addWeighted(trail_buffer, decay_rate, frame, 1 - decay_rate, 0)
    static_areas = motion_mask == 0
    trail_buffer[static_areas] = frame[static_areas]

    return trail_buffer, trail_buffer


def rgb_wave(frame, wave_amount=10):
    """
    Apply sinusoidal displacement to red and blue channels.
    Creates a wavy chromatic separation effect.
    """
    blue_channel, green_channel, red_channel = cv.split(frame)

    frame_height, frame_width = blue_channel.shape

    for row_index in range(frame_height):
        wave_offset = int(np.sin(row_index * 0.05) * wave_amount)
        red_channel[row_index] = np.roll(red_channel[row_index], wave_offset)
        blue_channel[row_index] = np.roll(blue_channel[row_index], -wave_offset)

    return cv.merge((blue_channel, green_channel, red_channel))


def posterize(frame, color_levels=6):
    """
    Reduce color depth by quantizing to fewer levels per channel.
    Creates a poster-like, banded color effect.
    """
    color_levels = max(2, color_levels)
    quantization_step = 256 // color_levels
    posterized_frame = (frame // quantization_step) * quantization_step
    return posterized_frame.astype(np.uint8)


def motion_smear(frame, motion_mask, smear_strength=15):
    """
    Smear moving areas horizontally by shifting pixels.
    Creates a speed blur effect in motion regions.
    """
    output_frame = frame.copy()
    smeared_frame = np.roll(frame, smear_strength, axis=1)
    motion_areas = motion_mask > 0
    output_frame[motion_areas] = smeared_frame[motion_areas]
    return output_frame


def change_knob(direction, current_values):
    """
    Increment or decrement all effect strength values.
    Used for global intensity adjustment.
    """
    adjusted_values = []
    for value in current_values:
        if direction == "up":
            adjusted_values.append(value + 1)
        else:
            adjusted_values.append(value - 1)
    return adjusted_values


def datamosh_vector(frame, previous_gray, motion_mask, displacement_strength=8):
    """
    Create a datamosh-style effect using optical flow.
    Displaces pixels based on motion vectors for a corrupted video look.
    """
    current_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    optical_flow = cv.calcOpticalFlowFarneback(
        previous_gray, current_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    frame_height, frame_width = current_gray.shape
    output_frame = frame.copy()

    # create coordinate grids
    row_coordinates, column_coordinates = np.mgrid[0:frame_height, 0:frame_width]

    # extract flow vectors
    horizontal_flow = optical_flow[..., 0]
    vertical_flow = optical_flow[..., 1]

    # calculate displaced source coordinates
    source_column = np.clip(
        (column_coordinates - horizontal_flow * displacement_strength).astype(np.int32),
        0, frame_width - 1
    )
    source_row = np.clip(
        (row_coordinates - vertical_flow * displacement_strength).astype(np.int32),
        0, frame_height - 1
    )

    # apply displacement only in motion areas
    motion_areas = motion_mask > 0
    output_frame[motion_areas] = frame[source_row[motion_areas], source_column[motion_areas]]

    return output_frame


def edge_glow(frame, glow_strength=1.5):
    """
    Detect edges and overlay them with a glowing cyan effect.
    """
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detected_edges = cv.Canny(grayscale, 50, 150)

    # dilate edges to create glow spread
    dilation_kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv.dilate(detected_edges, dilation_kernel, iterations=1)

    # create cyan glow layer
    glow_layer = np.zeros_like(frame)
    glow_layer[:, :, 0] = dilated_edges  # blue channel
    glow_layer[:, :, 1] = dilated_edges  # green channel

    # blend glow with original
    glowing_output = cv.addWeighted(frame, 1.0, glow_layer, glow_strength, 0)
    return np.clip(glowing_output, 0, 255).astype(np.uint8)


def chromatic_aberration(frame, channel_offset=5):
    """
    Shift RGB channels in opposite directions.
    Simulates lens chromatic aberration.
    """
    blue_channel, green_channel, red_channel = cv.split(frame)

    # shift red right, blue left
    red_channel = np.roll(red_channel, channel_offset, axis=1)
    blue_channel = np.roll(blue_channel, -channel_offset, axis=1)

    return cv.merge((blue_channel, green_channel, red_channel))


def vhs_effect(frame, noise_intensity=25, scanline_darkness=0.3):
    """
    Apply VHS-style distortion with analog noise and scanlines.
    Recreates the look of old VHS tapes.
    """
    frame_height, frame_width, _ = frame.shape
    output_frame = frame.copy().astype(np.float32)

    # add random noise
    analog_noise = np.random.randn(frame_height, frame_width, 3) * noise_intensity
    output_frame = output_frame + analog_noise

    # add horizontal scanlines
    for row_index in range(0, frame_height, 2):
        output_frame[row_index] = output_frame[row_index] * (1 - scanline_darkness)

    # add slight horizontal blur for color bleeding
    output_frame = cv.GaussianBlur(output_frame.astype(np.uint8), (3, 1), 0)

    return np.clip(output_frame, 0, 255).astype(np.uint8)


def mirror_effect(frame, mirror_mode='horizontal'):
    """
    Mirror the frame in various configurations.
    Modes: 'horizontal', 'vertical', 'quad' (four-way symmetry)
    """
    frame_height, frame_width, _ = frame.shape

    if mirror_mode == 'horizontal':
        left_half = frame[:, :frame_width // 2]
        frame[:, frame_width // 2:] = cv.flip(left_half, 1)

    elif mirror_mode == 'vertical':
        top_half = frame[:frame_height // 2, :]
        frame[frame_height // 2:, :] = cv.flip(top_half, 0)

    elif mirror_mode == 'quad':
        # mirror top-left quadrant into all four corners
        top_left_quadrant = frame[:frame_height // 2, :frame_width // 2]
        frame[:frame_height // 2, frame_width // 2:] = cv.flip(top_left_quadrant, 1)
        frame[frame_height // 2:, :frame_width // 2] = cv.flip(top_left_quadrant, 0)
        frame[frame_height // 2:, frame_width // 2:] = cv.flip(top_left_quadrant, -1)

    return frame


def thermal_vision(frame):
    """
    Apply thermal camera style colormap.
    Converts to grayscale with contrast enhancement, then applies INFERNO colormap.
    """
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    contrast_enhanced = cv.equalizeHist(grayscale)
    return cv.applyColorMap(contrast_enhanced, cv.COLORMAP_INFERNO)


def negative(frame):
    """Invert all colors in the frame."""
    return cv.bitwise_not(frame)


def pixelate(frame, pixel_block_size=8):
    """
    Pixelate the frame by downscaling and upscaling.
    Creates a retro, low-resolution look.
    """
    frame_height, frame_width, _ = frame.shape
    small_width = frame_width // pixel_block_size
    small_height = frame_height // pixel_block_size

    downscaled = cv.resize(frame, (small_width, small_height), interpolation=cv.INTER_LINEAR)
    pixelated = cv.resize(downscaled, (frame_width, frame_height), interpolation=cv.INTER_NEAREST)
    return pixelated


def kaleidoscope(frame, num_segments=6):
    """
    Create a kaleidoscope effect by rotating and mirroring segments.
    Produces symmetrical, mandala-like patterns.
    """
    frame_height, frame_width, _ = frame.shape
    center_x, center_y = frame_width // 2, frame_height // 2

    output_frame = np.zeros_like(frame)
    rotation_angle_step = 360 / num_segments

    for segment_index in range(num_segments):
        rotation_angle = segment_index * rotation_angle_step
        rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
        rotated_segment = cv.warpAffine(frame, rotation_matrix, (frame_width, frame_height))

        # flip every other segment for mirror symmetry
        if segment_index % 2 == 1:
            rotated_segment = cv.flip(rotated_segment, 1)

        # blend all segments together
        segment_weight = 1.0 / num_segments
        output_frame = cv.addWeighted(output_frame, 1.0, rotated_segment, segment_weight, 0)

    return output_frame


def color_channel_swap(frame, swap_mode='rgb_to_bgr'):
    """
    Swap color channels for trippy color effects.
    Different modes produce different color inversions.
    """
    blue_channel, green_channel, red_channel = cv.split(frame)

    if swap_mode == 'rgb_to_bgr':
        return cv.merge((red_channel, green_channel, blue_channel))
    elif swap_mode == 'gbr':
        return cv.merge((green_channel, blue_channel, red_channel))
    elif swap_mode == 'brg':
        return cv.merge((blue_channel, red_channel, green_channel))

    return frame


def emboss(frame, emboss_strength=1.0):
    """
    Apply an emboss effect creating a 3D relief appearance.
    """
    emboss_kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ]) * emboss_strength

    embossed_frame = cv.filter2D(frame, -1, emboss_kernel)
    # add gray offset for visibility
    return np.clip(embossed_frame + 128, 0, 255).astype(np.uint8)


def radial_blur(frame, blur_strength=0.02):
    """
    Apply a zoom/radial blur effect emanating from center.
    Creates a sense of motion or impact.
    """
    frame_height, frame_width, _ = frame.shape
    center_x, center_y = frame_width // 2, frame_height // 2

    output_frame = frame.copy().astype(np.float32)

    # blend multiple scaled versions
    for scale_iteration in range(1, 4):
        zoom_scale = 1 + blur_strength * scale_iteration
        zoom_matrix = cv.getRotationMatrix2D((center_x, center_y), 0, zoom_scale)
        scaled_frame = cv.warpAffine(frame, zoom_matrix, (frame_width, frame_height))
        output_frame = cv.addWeighted(output_frame.astype(np.uint8), 0.7, scaled_frame, 0.3, 0)

    return output_frame.astype(np.uint8)


def glitch_blocks(frame, num_glitch_blocks=8, max_displacement=30):
    """
    Randomly displace rectangular blocks for a digital glitch look.
    Some blocks also get color channel corruption.
    """
    frame_height, frame_width, _ = frame.shape
    output_frame = frame.copy()

    for _ in range(num_glitch_blocks):
        # random block dimensions
        block_height = np.random.randint(10, frame_height // 4)
        block_width = np.random.randint(20, frame_width // 3)

        # random block position
        block_top = np.random.randint(0, frame_height - block_height)
        block_left = np.random.randint(0, frame_width - block_width)

        # random displacement offset
        horizontal_offset = np.random.randint(-max_displacement, max_displacement)
        vertical_offset = np.random.randint(-max_displacement // 2, max_displacement // 2)

        # calculate source position (clamped to frame bounds)
        source_left = np.clip(block_left + horizontal_offset, 0, frame_width - block_width)
        source_top = np.clip(block_top + vertical_offset, 0, frame_height - block_height)

        # copy block from offset position
        output_frame[block_top:block_top + block_height, block_left:block_left + block_width] = \
            frame[source_top:source_top + block_height, source_left:source_left + block_width]

        # randomly add color channel corruption
        if np.random.rand() < 0.3:
            corrupted_block = output_frame[block_top:block_top + block_height, block_left:block_left + block_width]
            blue, green, red = cv.split(corrupted_block)
            channel_shift = np.random.randint(2, 8)
            red = np.roll(red, channel_shift, axis=1)
            output_frame[block_top:block_top + block_height, block_left:block_left + block_width] = cv.merge((blue, green, red))

    return output_frame


def color_drift(frame, frame_number, drift_speed=0.02, drift_intensity=50):
    """
    Smoothly shift hue over time for a psychedelic color cycling effect.
    """
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)

    # calculate hue shift based on time
    hue_shift_amount = int((frame_number * drift_speed * drift_intensity) % 180)
    hsv_frame[:, :, 0] = (hsv_frame[:, :, 0] + hue_shift_amount) % 180

    hsv_frame = hsv_frame.astype(np.uint8)
    return cv.cvtColor(hsv_frame, cv.COLOR_HSV2BGR)


def slit_scan(frame, scan_buffer, scan_slice_width=3):
    """
    Classic slit-scan effect - builds image from vertical slices over time.
    Creates time-smeared, elongated distortions.
    """
    frame_height, frame_width, _ = frame.shape

    if scan_buffer is None:
        scan_buffer = frame.copy()

    # shift existing buffer content to the left
    scan_buffer[:, :-scan_slice_width] = scan_buffer[:, scan_slice_width:]

    # capture new slice from center of current frame
    center_x = frame_width // 2
    scan_buffer[:, -scan_slice_width:] = frame[:, center_x:center_x + scan_slice_width]

    return scan_buffer.copy(), scan_buffer


def drunk_effect(frame, frame_number, wobble_intensity=15):
    """
    Wobbly, drunk-like distortion using animated sine waves.
    Creates a disorienting, swimming effect.
    """
    frame_height, frame_width, _ = frame.shape
    output_frame = np.zeros_like(frame)

    # create coordinate grids
    row_coords, column_coords = np.mgrid[0:frame_height, 0:frame_width]

    # calculate time-varying sine wave distortion
    time_factor = frame_number * 0.1
    horizontal_displacement = (np.sin(row_coords * 0.03 + time_factor) * wobble_intensity).astype(np.int32)
    vertical_displacement = (np.cos(column_coords * 0.03 + time_factor * 0.7) * wobble_intensity * 0.5).astype(np.int32)

    # calculate source coordinates with distortion
    source_column = np.clip(column_coords + horizontal_displacement, 0, frame_width - 1)
    source_row = np.clip(row_coords + vertical_displacement, 0, frame_height - 1)

    output_frame = frame[source_row, source_column]
    return output_frame


def ascii_art(frame, character_block_size=6):
    """
    Convert frame to ASCII-art style rendering.
    Renders blocks as colored squares with brightness-based intensity.
    """
    ascii_characters = " .:-=+*#%@"
    frame_height, frame_width, _ = frame.shape
    output_frame = np.zeros_like(frame)

    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    for block_top in range(0, frame_height, character_block_size):
        for block_left in range(0, frame_width, character_block_size):
            # get average brightness of this block
            brightness_block = grayscale[block_top:block_top + character_block_size,
                                        block_left:block_left + character_block_size]
            if brightness_block.size == 0:
                continue

            average_brightness = np.mean(brightness_block)
            character_index = int(average_brightness / 255 * (len(ascii_characters) - 1))

            # get average color of this block
            color_block = frame[block_top:block_top + character_block_size,
                               block_left:block_left + character_block_size]
            average_color = np.mean(color_block, axis=(0, 1))

            # scale color by character "density" (darker chars = less fill)
            density_scale = character_index / (len(ascii_characters) - 1)
            output_frame[block_top:block_top + character_block_size,
                        block_left:block_left + character_block_size] = average_color * density_scale

    return output_frame.astype(np.uint8)


def film_grain(frame, grain_intensity=30):
    """
    Add realistic film grain texture.
    Simulates analog film noise.
    """
    frame_height, frame_width, _ = frame.shape

    # generate random grain pattern
    grain_pattern = np.random.randn(frame_height, frame_width) * grain_intensity

    # apply grain to all color channels
    output_frame = frame.astype(np.float32)
    for channel_index in range(3):
        output_frame[:, :, channel_index] += grain_pattern

    return np.clip(output_frame, 0, 255).astype(np.uint8)


def tv_static(frame, static_blend=0.3):
    """
    Overlay TV static noise with occasional horizontal sync glitches.
    """
    frame_height, frame_width, _ = frame.shape

    # generate random static noise
    static_noise = np.random.randint(0, 256, (frame_height, frame_width), dtype=np.uint8)
    static_noise_colored = cv.cvtColor(static_noise, cv.COLOR_GRAY2BGR)

    # blend static with original frame
    output_frame = cv.addWeighted(frame, 1 - static_blend, static_noise_colored, static_blend, 0)

    # randomly add horizontal sync issues
    if np.random.rand() < 0.1:
        glitch_row = np.random.randint(0, frame_height)
        horizontal_shift = np.random.randint(5, 30)
        output_frame[glitch_row:glitch_row + 3] = np.roll(
            output_frame[glitch_row:glitch_row + 3], horizontal_shift, axis=1
        )

    return output_frame


def wave_distort(frame, frame_number, wave_amplitude=20, wave_frequency=0.05):
    """
    Apply flowing wave distortion like looking through water.
    """
    frame_height, frame_width, _ = frame.shape
    output_frame = np.zeros_like(frame)

    time_factor = frame_number * 0.05

    for row_index in range(frame_height):
        # calculate horizontal wave offset for this row
        horizontal_offset = int(np.sin(row_index * wave_frequency + time_factor) * wave_amplitude)
        output_frame[row_index] = np.roll(frame[row_index], horizontal_offset, axis=0)

    return output_frame


def oil_paint(frame, filter_size=5, dynamic_ratio=1):
    """
    Oil painting effect using bilateral filtering and color quantization.
    Smooths while preserving edges, then reduces color levels.
    """
    # apply bilateral filter for painterly smoothing
    smoothed_frame = cv.bilateralFilter(frame, filter_size * 2, 75, 75)

    # quantize colors to reduce palette
    color_levels = 8
    quantization_step = 256 // color_levels
    quantized_frame = (smoothed_frame // quantization_step) * quantization_step

    # extract and darken edges for painted look
    grayscale = cv.cvtColor(smoothed_frame, cv.COLOR_BGR2GRAY)
    edge_map = cv.Laplacian(grayscale, cv.CV_8U, ksize=3)
    edge_map_colored = cv.cvtColor(edge_map, cv.COLOR_GRAY2BGR)

    # subtract edges to create dark outlines
    output_frame = cv.subtract(quantized_frame, edge_map_colored // 2)

    return output_frame


def ghost_trail(frame, ghost_buffer, fade_decay=0.85, num_ghost_copies=4):
    """
    Create trailing ghost copies that fade over time.
    Produces ethereal motion trails.
    """
    if ghost_buffer is None:
        ghost_buffer = [frame.copy() for _ in range(num_ghost_copies)]

    # update buffer: remove oldest, add current
    ghost_buffer.pop()
    ghost_buffer.insert(0, frame.copy())

    output_frame = frame.copy().astype(np.float32)

    # blend in ghost copies with decreasing opacity
    for ghost_index, ghost_frame in enumerate(ghost_buffer[1:], 1):
        ghost_opacity = fade_decay ** ghost_index

        # offset each ghost slightly for trail effect
        horizontal_offset = ghost_index * 3
        vertical_offset = ghost_index * 2
        shifted_ghost = np.roll(ghost_frame, horizontal_offset, axis=1)
        shifted_ghost = np.roll(shifted_ghost, vertical_offset, axis=0)

        output_frame = cv.addWeighted(
            output_frame.astype(np.uint8), 1.0,
            shifted_ghost, ghost_opacity * 0.3, 0
        ).astype(np.float32)

    return output_frame.astype(np.uint8), ghost_buffer


def spiral_warp(frame, frame_number, warp_strength=0.5):
    """
    Warp the image in an animated spiral pattern from center.
    Creates a hypnotic, swirling distortion.
    """
    frame_height, frame_width, _ = frame.shape
    center_x, center_y = frame_width // 2, frame_height // 2

    # create coordinate grids
    row_coords, column_coords = np.mgrid[0:frame_height, 0:frame_width]

    # calculate distance and angle from center for each pixel
    delta_x = column_coords - center_x
    delta_y = row_coords - center_y
    distance_from_center = np.sqrt(delta_x ** 2 + delta_y ** 2)
    angle_from_center = np.arctan2(delta_y, delta_x)

    # apply animated rotation based on distance from center
    time_factor = frame_number * 0.02
    rotation_amount = warp_strength * np.sin(distance_from_center * 0.02 + time_factor)
    new_angle = angle_from_center + rotation_amount

    # calculate warped coordinates
    warped_x = center_x + distance_from_center * np.cos(new_angle)
    warped_y = center_y + distance_from_center * np.sin(new_angle)

    warped_x = np.clip(warped_x, 0, frame_width - 1).astype(np.int32)
    warped_y = np.clip(warped_y, 0, frame_height - 1).astype(np.int32)

    return frame[warped_y, warped_x]


def color_threshold(frame, brightness_threshold=128):
    """
    Create high-contrast color bands based on brightness threshold.
    Bright areas get saturated, dark areas get desaturated.
    """
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    output_frame = frame.copy()

    # identify bright and dark regions
    bright_areas = grayscale > brightness_threshold
    dark_areas = ~bright_areas

    # modify saturation and value in HSV space
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_frame[:, :, 1] = np.where(bright_areas, 255, hsv_frame[:, :, 1] // 2)
    hsv_frame[:, :, 2] = np.where(bright_areas, hsv_frame[:, :, 2], hsv_frame[:, :, 2] // 2)

    return cv.cvtColor(hsv_frame, cv.COLOR_HSV2BGR)


def digital_rain(frame, rain_drops=None, total_drops=200):
    """
    Matrix-style digital rain overlay.
    Animated green falling streaks over darkened frame.
    """
    frame_height, frame_width, _ = frame.shape

    if rain_drops is None:
        # initialize rain drops: [x_position, y_position, fall_speed, trail_length]
        rain_drops = []
        for _ in range(total_drops):
            x_position = np.random.randint(0, frame_width)
            y_position = np.random.randint(-frame_height, 0)
            fall_speed = np.random.randint(5, 15)
            trail_length = np.random.randint(10, 30)
            rain_drops.append([x_position, y_position, fall_speed, trail_length])

    output_frame = frame.copy()

    # darken base frame for contrast
    output_frame = (output_frame * 0.7).astype(np.uint8)

    for drop in rain_drops:
        x_position, y_position, fall_speed, trail_length = drop

        # draw the trailing streak
        for trail_index in range(trail_length):
            trail_y = y_position - trail_index * 2
            if 0 <= trail_y < frame_height and 0 <= x_position < frame_width:
                # fade from bright green at head to dark at tail
                green_intensity = int(255 * (1 - trail_index / trail_length))
                output_frame[trail_y, x_position] = [0, green_intensity, 0]

        # update drop position
        drop[1] += fall_speed

        # reset drop if it falls off screen
        if drop[1] > frame_height + trail_length * 2:
            drop[0] = np.random.randint(0, frame_width)
            drop[1] = np.random.randint(-frame_height // 2, 0)
            drop[2] = np.random.randint(5, 15)

    return output_frame, rain_drops


def tunnel_vision(frame, vignette_intensity=0.7):
    """
    Dark vignette that creates a tunnel/spotlight effect.
    Darkens edges, keeps center bright.
    """
    frame_height, frame_width, _ = frame.shape
    center_x, center_y = frame_width // 2, frame_height // 2

    # create radial gradient based on distance from center
    row_coords, column_coords = np.mgrid[0:frame_height, 0:frame_width]
    distance_from_center = np.sqrt((column_coords - center_x) ** 2 + (row_coords - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

    # create vignette mask (1 at center, fading to darker at edges)
    vignette_mask = 1 - (distance_from_center / max_distance) * vignette_intensity
    vignette_mask = np.clip(vignette_mask, 0, 1)

    output_frame = frame.astype(np.float32)
    for channel_index in range(3):
        output_frame[:, :, channel_index] *= vignette_mask

    return output_frame.astype(np.uint8)


def double_vision(frame, shift_offset=15, overlay_blend=0.5):
    """
    Overlay a shifted copy for a double vision effect.
    Simulates seeing double.
    """
    shifted_copy = np.roll(frame, shift_offset, axis=1)
    shifted_copy = np.roll(shifted_copy, shift_offset // 2, axis=0)

    return cv.addWeighted(frame, 1 - overlay_blend, shifted_copy, overlay_blend, 0)


def color_quantize(frame, num_colors=8):
    """
    Reduce image to limited color palette.
    Creates a posterized, reduced-color look.
    """
    frame_height, frame_width, _ = frame.shape
    pixels = frame.reshape(-1, 3).astype(np.float32)

    # simple quantization (faster than k-means)
    quantization_step = 256 // num_colors
    quantized_pixels = (pixels // quantization_step * quantization_step).astype(np.uint8)

    return quantized_pixels.reshape(frame_height, frame_width, 3)


def scanline_intensity(frame, scanline_height=2, scanline_darkness=0.4):
    """
    CRT-style scanlines with adjustable intensity.
    Darkens alternating horizontal bands.
    """
    frame_height, frame_width, _ = frame.shape
    output_frame = frame.copy().astype(np.float32)

    for row_index in range(0, frame_height, scanline_height * 2):
        output_frame[row_index:row_index + scanline_height] *= (1 - scanline_darkness)

    return output_frame.astype(np.uint8)


def rgb_split_radial(frame, split_strength=10):
    """
    Split RGB channels radially from center.
    Red zooms out, blue zooms in, creating color fringing.
    """
    frame_height, frame_width, _ = frame.shape
    center_x, center_y = frame_width // 2, frame_height // 2

    blue_channel, green_channel, red_channel = cv.split(frame)

    # create zoom matrices for red (larger) and blue (smaller)
    red_scale = 1 + split_strength * 0.002
    blue_scale = 1 - split_strength * 0.002

    red_transform = cv.getRotationMatrix2D((center_x, center_y), 0, red_scale)
    blue_transform = cv.getRotationMatrix2D((center_x, center_y), 0, blue_scale)

    red_channel = cv.warpAffine(red_channel, red_transform, (frame_width, frame_height))
    blue_channel = cv.warpAffine(blue_channel, blue_transform, (frame_width, frame_height))

    return cv.merge((blue_channel, green_channel, red_channel))


def motion_blur_directional(frame, blur_angle=0, blur_strength=15):
    """
    Apply motion blur in a specific direction.
    Simulates camera movement blur.
    """
    kernel_size = blur_strength
    blur_kernel = np.zeros((kernel_size, kernel_size))

    # calculate kernel line based on angle
    angle_radians = np.deg2rad(blur_angle)
    direction_x = np.cos(angle_radians)
    direction_y = np.sin(angle_radians)

    kernel_center = kernel_size // 2
    for step in range(kernel_size):
        kernel_x = int(kernel_center + (step - kernel_center) * direction_x)
        kernel_y = int(kernel_center + (step - kernel_center) * direction_y)
        if 0 <= kernel_x < kernel_size and 0 <= kernel_y < kernel_size:
            blur_kernel[kernel_y, kernel_x] = 1

    # normalize kernel
    kernel_sum = blur_kernel.sum()
    blur_kernel /= kernel_sum if kernel_sum > 0 else 1

    return cv.filter2D(frame, -1, blur_kernel)


def sketch_effect(frame):
    """
    Convert to pencil sketch style.
    Uses dodge blend technique for artistic sketch look.
    """
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    inverted_grayscale = cv.bitwise_not(grayscale)
    blurred_inverted = cv.GaussianBlur(inverted_grayscale, (21, 21), 0)
    sketch_result = cv.divide(grayscale, 255 - blurred_inverted, scale=256)

    return cv.cvtColor(sketch_result, cv.COLOR_GRAY2BGR)


def halftone(frame, halftone_dot_size=4):
    """
    Create a halftone/newspaper print effect.
    Renders brightness as variable-size dots.
    """
    frame_height, frame_width, _ = frame.shape
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    output_frame = np.zeros_like(frame)

    for block_top in range(0, frame_height, halftone_dot_size):
        for block_left in range(0, frame_width, halftone_dot_size):
            brightness_block = grayscale[block_top:block_top + halftone_dot_size,
                                        block_left:block_left + halftone_dot_size]
            if brightness_block.size == 0:
                continue

            average_brightness = np.mean(brightness_block)
            dot_radius = int((average_brightness / 255) * (halftone_dot_size // 2))

            if dot_radius > 0:
                dot_center_y = block_top + halftone_dot_size // 2
                dot_center_x = block_left + halftone_dot_size // 2
                cv.circle(output_frame, (dot_center_x, dot_center_y), dot_radius, (255, 255, 255), -1)

    return output_frame


def neon_edges(frame, glow_blur_size=5):
    """
    Glowing neon outline effect.
    Detects edges and colors them with rainbow gradient, adds glow.
    """
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detected_edges = cv.Canny(grayscale, 100, 200)

    # create rainbow-colored edges based on vertical position
    colored_edges = np.zeros_like(frame)
    edge_height, edge_width = detected_edges.shape

    for row_index in range(edge_height):
        # calculate hue based on row position (creates vertical rainbow)
        row_hue = int((row_index / edge_height) * 180)
        hsv_color = np.array([[[row_hue, 255, 255]]], dtype=np.uint8)
        bgr_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)[0, 0]

        # apply color to edge pixels in this row
        edge_pixels_in_row = detected_edges[row_index] > 0
        colored_edges[row_index, edge_pixels_in_row] = bgr_color

    # create glow by blurring colored edges
    blur_kernel_size = glow_blur_size * 2 + 1
    edge_glow = cv.GaussianBlur(colored_edges, (blur_kernel_size, blur_kernel_size), 0)

    # blend glow with darkened original, then add sharp edges on top
    output_frame = cv.addWeighted(frame, 0.3, edge_glow, 0.7, 0)
    output_frame = cv.add(output_frame, colored_edges)

    return np.clip(output_frame, 0, 255).astype(np.uint8)


def glitch_shift(frame, shift_intensity=20):
    """
    Random horizontal line shifts for digital glitch effect.
    Creates corrupted scan line appearance.
    """
    frame_height, frame_width, _ = frame.shape
    output_frame = frame.copy()

    num_glitch_lines = np.random.randint(3, 10)

    for _ in range(num_glitch_lines):
        glitch_row = np.random.randint(0, frame_height - 5)
        glitch_height = np.random.randint(1, 8)
        horizontal_shift = np.random.randint(-shift_intensity, shift_intensity)

        output_frame[glitch_row:glitch_row + glitch_height] = np.roll(
            output_frame[glitch_row:glitch_row + glitch_height], horizontal_shift, axis=1
        )

        # randomly corrupt a single color channel
        if np.random.rand() < 0.3:
            channel_to_corrupt = np.random.randint(0, 3)
            channel_shift = np.random.randint(-5, 5)
            output_frame[glitch_row:glitch_row + glitch_height, :, channel_to_corrupt] = np.roll(
                output_frame[glitch_row:glitch_row + glitch_height, :, channel_to_corrupt],
                channel_shift,
                axis=1
            )

    return output_frame


def heat_distort(frame, frame_number, distortion_intensity=8):
    """
    Heat wave distortion like hot air rising.
    Creates shimmering mirage effect.
    """
    frame_height, frame_width, _ = frame.shape
    output_frame = np.zeros_like(frame)

    time_factor = frame_number * 0.15

    row_coords, column_coords = np.mgrid[0:frame_height, 0:frame_width]

    # calculate heat wave displacement pattern
    horizontal_displacement = (
        np.sin(row_coords * 0.1 + time_factor) *
        np.sin(column_coords * 0.05) *
        distortion_intensity
    ).astype(np.int32)

    source_column = np.clip(column_coords + horizontal_displacement, 0, frame_width - 1)

    output_frame = frame[row_coords, source_column]

    return output_frame


def cross_process(frame):
    """
    Cross-processing color effect (film processing mistake look).
    Shifts colors in ways that mimic developing film in wrong chemicals.
    """
    blue_channel, green_channel, red_channel = cv.split(frame)

    # apply curves to each channel for cross-processed look
    blue_channel = np.clip(blue_channel.astype(np.float32) * 1.2 + 20, 0, 255).astype(np.uint8)
    red_channel = np.clip(red_channel.astype(np.float32) * 0.9 - 10, 0, 255).astype(np.uint8)
    green_channel = np.clip(green_channel.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)

    output_frame = cv.merge((blue_channel, green_channel, red_channel))

    # boost overall contrast
    output_frame = cv.convertScaleAbs(output_frame, alpha=1.2, beta=-20)

    return output_frame


def duotone(frame, shadow_color=(0, 50, 100), highlight_color=(200, 150, 50)):
    """
    Two-tone color effect mapping shadows and highlights to two colors.
    Creates stylized, limited palette look.
    """
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32) / 255

    output_frame = np.zeros_like(frame, dtype=np.float32)

    # interpolate between shadow and highlight colors based on brightness
    for channel_index in range(3):
        output_frame[:, :, channel_index] = (
            shadow_color[channel_index] * (1 - grayscale) +
            highlight_color[channel_index] * grayscale
        )

    return output_frame.astype(np.uint8)


def pulse_zoom(frame, frame_number, pulse_speed=0.1, pulse_amount=0.03):
    """
    Rhythmic zoom pulsing effect.
    Creates a breathing, pulsating zoom.
    """
    frame_height, frame_width, _ = frame.shape
    center_x, center_y = frame_width // 2, frame_height // 2

    # calculate zoom scale using sine wave
    zoom_scale = 1 + np.sin(frame_number * pulse_speed) * pulse_amount

    zoom_matrix = cv.getRotationMatrix2D((center_x, center_y), 0, zoom_scale)
    return cv.warpAffine(frame, zoom_matrix, (frame_width, frame_height))


def blocky_noise(frame, corruption_block_size=16, corruption_chance=0.1):
    """
    Random blocky noise corruption.
    Simulates digital video corruption with various block artifacts.
    """
    frame_height, frame_width, _ = frame.shape
    output_frame = frame.copy()

    for block_top in range(0, frame_height, corruption_block_size):
        for block_left in range(0, frame_width, corruption_block_size):
            if np.random.rand() < corruption_chance:
                corruption_type = np.random.randint(0, 4)

                if corruption_type == 0:
                    # fill with random solid color
                    random_color = np.random.randint(0, 256, 3)
                    output_frame[block_top:block_top + corruption_block_size,
                                block_left:block_left + corruption_block_size] = random_color

                elif corruption_type == 1:
                    # copy from nearby block
                    offset_x = np.random.randint(-2, 3) * corruption_block_size
                    offset_y = np.random.randint(-2, 3) * corruption_block_size
                    source_top = np.clip(block_top + offset_y, 0, frame_height - corruption_block_size)
                    source_left = np.clip(block_left + offset_x, 0, frame_width - corruption_block_size)
                    output_frame[block_top:block_top + corruption_block_size,
                                block_left:block_left + corruption_block_size] = \
                        frame[source_top:source_top + corruption_block_size,
                             source_left:source_left + corruption_block_size]

                elif corruption_type == 2:
                    # invert colors
                    output_frame[block_top:block_top + corruption_block_size,
                                block_left:block_left + corruption_block_size] = \
                        255 - frame[block_top:block_top + corruption_block_size,
                                   block_left:block_left + corruption_block_size]

                else:
                    # shift color channels
                    corrupted_block = output_frame[block_top:block_top + corruption_block_size,
                                                   block_left:block_left + corruption_block_size]
                    output_frame[block_top:block_top + corruption_block_size,
                                block_left:block_left + corruption_block_size] = np.roll(corrupted_block, 1, axis=2)

    return output_frame


def retro_crt(frame):
    """
    Complete retro CRT monitor effect.
    Combines phosphor pattern, scanlines, blur, and vignette.
    """
    frame_height, frame_width, _ = frame.shape
    output_frame = frame.copy().astype(np.float32)

    # simulate RGB phosphor stripe pattern
    for column_index in range(frame_width):
        active_channel = column_index % 3
        phosphor_mask = np.ones(3) * 0.7
        phosphor_mask[active_channel] = 1.0
        output_frame[:, column_index] *= phosphor_mask

    # add horizontal scanlines
    for row_index in range(0, frame_height, 2):
        output_frame[row_index] *= 0.8

    # add slight blur for phosphor glow
    output_frame = cv.GaussianBlur(output_frame.astype(np.uint8), (3, 3), 0)

    # apply vignette (darker edges)
    center_x, center_y = frame_width // 2, frame_height // 2
    row_coords, column_coords = np.mgrid[0:frame_height, 0:frame_width]
    distance_from_center = np.sqrt((column_coords - center_x) ** 2 + (row_coords - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    vignette_mask = 1 - (distance_from_center / max_distance) * 0.4

    output_frame = output_frame.astype(np.float32)
    for channel_index in range(3):
        output_frame[:, :, channel_index] *= vignette_mask

    return np.clip(output_frame, 0, 255).astype(np.uint8)


def time_echo(frame, echo_buffer, num_echo_frames=5, echo_decay=0.7):
    """
    Temporal echo showing past frames blended.
    Creates ghostly persistence of previous frames.
    """
    if echo_buffer is None:
        echo_buffer = []

    echo_buffer.insert(0, frame.copy())
    if len(echo_buffer) > num_echo_frames:
        echo_buffer.pop()

    output_frame = frame.copy().astype(np.float32)

    for echo_index, echo_frame in enumerate(echo_buffer[1:], 1):
        echo_opacity = echo_decay ** echo_index
        output_frame = cv.addWeighted(
            output_frame.astype(np.uint8), 1.0,
            echo_frame, echo_opacity * 0.2, 0
        ).astype(np.float32)

    return output_frame.astype(np.uint8), echo_buffer


def prism(frame, prism_offset=8):
    """
    Prismatic color splitting effect.
    Offsets RGB channels in different diagonal directions.
    """
    frame_height, frame_width, _ = frame.shape
    blue_channel, green_channel, red_channel = cv.split(frame)

    # offset red channel diagonally one way
    red_channel = np.roll(red_channel, prism_offset, axis=1)
    red_channel = np.roll(red_channel, -prism_offset // 2, axis=0)

    # offset blue channel diagonally the other way
    blue_channel = np.roll(blue_channel, -prism_offset, axis=1)
    blue_channel = np.roll(blue_channel, prism_offset // 2, axis=0)

    return cv.merge((blue_channel, green_channel, red_channel))


def rotate_frame(frame, frame_number, rotation_speed=0.5):
    """
    Continuously rotate the video frame in a circle.
    rotation_speed: degrees per frame (positive = clockwise)
    Maintains consistent size throughout rotation by scaling to fit.
    """
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width / 2, frame_height / 2

    # Calculate current rotation angle
    angle = (frame_number * rotation_speed) % 360

    # Calculate the scale factor needed to keep the rotated image fitting
    # within the original frame size without any content being cut off
    angle_rad = np.radians(angle)
    cos_val = abs(np.cos(angle_rad))
    sin_val = abs(np.sin(angle_rad))

    # The rotated bounding box dimensions
    rotated_width = frame_width * cos_val + frame_height * sin_val
    rotated_height = frame_width * sin_val + frame_height * cos_val

    # Scale factor to fit the rotated frame back into original dimensions
    scale = min(frame_width / rotated_width, frame_height / rotated_height)

    # Get rotation matrix with scale applied
    rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), angle, scale)

    # Apply rotation - output stays same size as input
    rotated = cv.warpAffine(frame, rotation_matrix, (frame_width, frame_height),
                            borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return rotated
