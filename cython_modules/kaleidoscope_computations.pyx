# kaleidoscope_computations.pyx
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, atan2, sqrt, M_PI # Removed radians, using M_PI for conversion

# Define numpy array types for efficient memoryview access
DTYPE_FLOAT = np.float32
ctypedef np.float32_t DTYPE_FLOAT_t

@cython.boundscheck(False) # Disable bounds checking for performance
@cython.wraparound(False)  # Disable negative indexing for performance
cpdef tuple fill_kaleidoscope_data_cython(
    np.ndarray[DTYPE_FLOAT_t, ndim=2, mode='c'] vertices_buffer,
    np.ndarray[DTYPE_FLOAT_t, ndim=2, mode='c'] colors_buffer,
    int start_offset,
    float rotation_angle,
    float hue_offset,
    float amplitude,
    bint is_current_pattern,
    float strobe_val,
    float base_alpha,
    int frame_count, # Pass frame_count from Python
    int oscillation_mode # Pass oscillation_mode from Python
):
    """
    Fills vertex and color data for a single kaleidoscope pattern (current or ghost)
    into pre-allocated NumPy arrays, optimized with Cython.

    Args:
        vertices_buffer (np.ndarray): The NumPy array for vertex coordinates (modified in-place).
        colors_buffer (np.ndarray): The NumPy array for vertex colors (modified in-place).
        start_offset (int): The starting index in the buffers for this pattern's data.
        rotation_angle (float): The rotation angle for this pattern.
        hue_offset (float): The hue offset for this pattern's color (0-360).
        amplitude (float): The audio amplitude for this pattern (0.0-1.0), for oscillation.
        is_current_pattern (bool): True if this is the live pattern, False for ghosts.
        strobe_val (float): The current strobing value (0.0-1.0) to apply to colors.
        base_alpha (float): The base alpha for this pattern (0.0-1.0, fading for ghosts).
        frame_count (int): Current animation frame count for temporal oscillation phase.
        oscillation_mode (int): 0 for linear oscillation, 1 for circular.

    Returns:
        tuple: A tuple (total_vertices_added, list_of_sub_draw_commands).
               total_vertices_added (int): The total number of vertices written for this pattern.
               list_of_sub_draw_commands (list): List of (relative_start_idx, num_points_in_strip, line_width) for each line.
    """
    cdef int current_write_idx = start_offset
    cdef int num_segments = 12
    
    # Dynamic grid size based on amplitude for "scatter" effect
    # Base grid size 150, expands up to 350 with max amplitude
    cdef float dynamic_grid_size = 150.0 + amplitude * 300.0 # Increased scatter range
    
    cdef int num_lines = 10
    # Amplify oscillation magnitude even more
    cdef float osc_magnitude = amplitude * 1200.0 # Even more exaggerated oscillation

    # --- STROBE / FLASH LOGIC ---
    # Introduce a fast “square wave” based on frame_count to create a momentary flash.
    # If amplitude is above a small threshold, we let “flash_mode” flip between 0 or 1 quickly.
    # You can adjust flash_speed to taste (higher => faster toggling).
    cdef float flash_threshold = 0.25  # amplitude threshold to even consider flashing
    cdef float flash_speed = 0.5       # in cycles per frame; bigger => faster flicker

    # Compute a raw “oscillator” that goes between -1 and +1.
    cdef float raw_osc = sin(frame_count * flash_speed * 2.0 * M_PI)
    # Turn that into a square wave (0 or 1)
    cdef int square_wave = 1 if raw_osc > 0.0 else 0

    # Now define a flash_factor: 
    #   if amplitude < flash_threshold, always 0. 
    #   else it’s the square_wave (so it toggles on/off when audio is loud enough).
    cdef float flash_factor = 0.0
    if amplitude >= flash_threshold:
        flash_factor = float(square_wave)

    # --- AGGRESSIVE SATURATION / VALUE BASED ON STROBE + FLASH ---
    # We still want strobe_val in [0,1] from Python. We raise it to a higher power for
    # more dramatic desaturation near max strobe. Then we optionally force a full-white
    # flash if flash_factor==1.

    # exponentiate strobe_val for sharper ramp
    cdef float strobe_pow = strobe_val ** 6.0 # bigger exponent => sharper effect
    # “desat” is how much we reduce saturation when strobing
    cdef float desaturation_factor = strobe_pow 
    cdef int target_saturation_hsv = <int>( (1.0 - desaturation_factor) * 255.0 )
    if target_saturation_hsv < 0:
        target_saturation_hsv = 0
    elif target_saturation_hsv > 255:
        target_saturation_hsv = 255

    # Now manage brightness/value:
    # If flash_factor is 1 => force full white (V=255).
    # else do a normal ramp (from 100 to 255) based on strobe_pow.
    # MODIFIED: Increased base value for target_value_hsv to make quieter colors brighter.
    cdef int target_value_hsv
    if flash_factor >= 1.0:
        target_value_hsv = 255
    else:
        # Base brightness is 100, ramps up to 255 with strobe_pow
        target_value_hsv = <int>(100 + strobe_pow * 155.0) # Range from 100 to 255
        if target_value_hsv < 100: # Ensure minimum brightness for quiet audio
            target_value_hsv = 100
        elif target_value_hsv > 255:
            target_value_hsv = 255

    # Convert HSV => RGB (0..1 floats) 
    cdef float h_norm = hue_offset / 360.0
    cdef float s_norm = target_saturation_hsv / 255.0
    cdef float v_norm = target_value_hsv / 255.0

    cdef float r, g, b
    cdef float i_h, f, p, q, t

    if s_norm == 0.0:
        r = v_norm
        g = v_norm
        b = v_norm
    else:
        i_h = h_norm * 6.0
        if i_h == 6.0: i_h = 0.0
        f = i_h - <int>i_h
        p = v_norm * (1.0 - s_norm)
        q = v_norm * (1.0 - s_norm * f)
        t = v_norm * (1.0 - s_norm * (1.0 - f))
        if <int>i_h == 0:
            r, g, b = v_norm, t, p
        elif <int>i_h == 1:
            r, g, b = q, v_norm, p
        elif <int>i_h == 2:
            r, g, b = p, v_norm, t
        elif <int>i_h == 3:
            r, g, b = p, q, v_norm
        elif <int>i_h == 4:
            r, g, b = t, p, v_norm
        else: # <int>i_h == 5
            r, g, b = v_norm, p, q

    # We’ll optionally drive alpha higher when flash_factor==1, so lines briefly max out opacity.
    cdef float base_a_final = base_alpha
    if flash_factor >= 1.0:
        # temporarily bump alpha for a very sharp flash
        base_a_final = 1.0

    # Prepare return structure for line widths. We'll override these in Python if needed.
    cdef list sub_draw_commands = []

    cdef float x, y, temp_x, temp_y, rotated_x, rotated_y, scaled_y, osc_offset
    cdef float rad_seg_rot_val, angle, dist
    cdef int i_seg, y_line_idx, x_point_idx, x_line_idx, y_point_idx
    cdef int line_start_idx
    cdef float seg_rot_angle
    cdef float scale_y

    # Define “pulse” line width: when flash_factor==1, use a much fatter line.
    cdef float ghost_line_width = 2.0
    cdef float ghost_flash_width = 5.0  # fatter on flash
    cdef float core_line_width = 4.0
    cdef float core_flash_width = 8.0

    # Spatial & temporal freq for sinusoidal oscillation
    cdef float osc_spatial_freq = 0.05 + amplitude * 0.3
    cdef float osc_temporal_freq = 0.05 + amplitude * 0.25

    for i_seg in range(num_segments): # Changed 'i' to 'i_seg' to avoid conflict with 'i_h'
        seg_rot_angle = i_seg * (360.0 / num_segments) # Assign value here
        rad_seg_rot_val = seg_rot_angle * M_PI / 180.0 # Calculate radians here
        scale_y = -1.0 if i_seg % 2 == 1 else 1.0 # Assign value here

        # Horizontal lines
        for y_line_idx in range(num_lines + 1): # Outer loop for horizontal lines
            line_start_idx = current_write_idx - start_offset # Relative start index for this line strip
            
            for x_point_idx in range(num_lines + 1): # Iterate for points along this line
                x = (x_point_idx / <float>num_lines) * dynamic_grid_size - (dynamic_grid_size / 2.0)
                y = (y_line_idx / <float>num_lines) * dynamic_grid_size - (dynamic_grid_size / 2.0)
                
                osc_offset = 0.0
                if oscillation_mode == 0:
                    # Linear mode oscillation: depends on y-position and frame count
                    osc_offset = osc_magnitude * sin(y * osc_spatial_freq + frame_count * osc_temporal_freq)
                else:
                    # Circular mode oscillation: depends on distance from center, angle, and frame count
                    angle = atan2(y, x)
                    dist = sqrt(x*x + y*y)
                    osc_offset = osc_magnitude * sin(dist * osc_spatial_freq + angle * 2.0 + frame_count * osc_temporal_freq)
                
                temp_x = x + osc_offset
                temp_y = y
                
                rotated_x = temp_x * cos(rad_seg_rot_val) - temp_y * sin(rad_seg_rot_val) # Use rad_seg_rot_val
                rotated_y = temp_x * sin(rad_seg_rot_val) + temp_y * cos(rad_seg_rot_val) # Use rad_seg_rot_val
                scaled_y = rotated_y * scale_y

                vertices_buffer[current_write_idx][0] = rotated_x
                vertices_buffer[current_write_idx][1] = scaled_y
                colors_buffer[current_write_idx][0] = r
                colors_buffer[current_write_idx][1] = g
                colors_buffer[current_write_idx][2] = b
                colors_buffer[current_write_idx][3] = base_a_final
                current_write_idx += 1
            
            # Decide which line width to use for this horizontal strip:
            if is_current_pattern:
                # core is “current pattern” → thicker
                sub_draw_commands.append((line_start_idx, num_lines + 1, 
                                          core_flash_width if flash_factor >= 1.0 else core_line_width))
            else:
                sub_draw_commands.append((line_start_idx, num_lines + 1, 
                                          ghost_flash_width if flash_factor >= 1.0 else ghost_line_width))

        # Vertical lines
        for x_line_idx in range(num_lines + 1): # Outer loop for vertical lines
            line_start_idx = current_write_idx - start_offset # Relative start index for this line strip
            for y_point_idx in range(num_lines + 1): # Iterate for points along this line
                x = (x_line_idx / <float>num_lines) * dynamic_grid_size - (dynamic_grid_size / 2.0)
                y = (y_point_idx / <float>num_lines) * dynamic_grid_size - (dynamic_grid_size / 2.0)
                
                osc_offset = 0.0
                if oscillation_mode == 0:
                    # Linear mode oscillation: depends on x-position and frame count
                    osc_offset = osc_magnitude * sin(x * osc_spatial_freq + frame_count * osc_temporal_freq)
                else:
                    # Circular mode oscillation: depends on distance from center, angle, and frame count
                    angle = atan2(y, x)
                    dist = sqrt(x*x + y*y)
                    osc_offset = osc_magnitude * sin(dist * osc_spatial_freq + angle * 2.0 + frame_count * osc_temporal_freq)
                
                temp_x = x
                temp_y = y + osc_offset

                rotated_x = temp_x * cos(rad_seg_rot_val) - temp_y * sin(rad_seg_rot_val) # Use rad_seg_rot_val
                rotated_y = temp_x * sin(rad_seg_rot_val) + temp_y * cos(rad_seg_rot_val) # Use rad_seg_rot_val
                scaled_y = rotated_y * scale_y

                vertices_buffer[current_write_idx][0] = rotated_x
                vertices_buffer[current_write_idx][1] = scaled_y
                colors_buffer[current_write_idx][0] = r
                colors_buffer[current_write_idx][1] = g
                colors_buffer[current_write_idx][2] = b
                colors_buffer[current_write_idx][3] = base_a_final
                current_write_idx += 1

            if is_current_pattern:
                sub_draw_commands.append((line_start_idx, num_lines + 1, 
                                          core_flash_width if flash_factor >= 1.0 else core_line_width))
            else:
                sub_draw_commands.append((line_start_idx, num_lines + 1, 
                                          ghost_flash_width if flash_factor >= 1.0 else ghost_line_width))

    total_vertices_added = current_write_idx - start_offset
    return total_vertices_added, sub_draw_commands
