# oscilloscope_computations.pyx
# Compiler directives for optimization:
# distutils: extra_compile_args=-O3 -ffast-math
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: cProfile=False
# cython: profile=False


import numpy as np
cimport numpy as np
import math
cimport cython

# Define the data type for NumPy arrays for efficient Cython interaction
DTYPE_FLOAT = np.float32
ctypedef np.float32_t DTYPE_FLOAT_t

@cython.boundscheck(False) # Turn off bounds-checking for speed
@cython.wraparound(False)  # Turn off negative index checking for speed
@cython.cdivision(True)    # Use C-style division for floats

def fill_trace_data_cython(
    np.ndarray[DTYPE_FLOAT_t, ndim=1] mono_data, # Input audio data for this specific trace
    np.ndarray[DTYPE_FLOAT_t, ndim=2] vertices_buffer, # Main pre-allocated vertex buffer (output)
    np.ndarray[DTYPE_FLOAT_t, ndim=2] colors_buffer,   # Main pre-allocated color buffer (output)
    int start_buffer_offset, # The starting index in the main buffers for this trace's data
    float w, float h, float center_x, float center_y, # Widget dimensions
    int points_to_draw, int sample_step, # Trace rendering details
    int current_mode, # Oscilloscope mode (LINEAR_MODE or CIRCULAR_MODE)
    int i, int num_traces, # Ghost index and total number of traces (for alpha/scroll calculations)
    float max_linear_scroll_dist, # Linear mode parameter
    float max_spiral_radius_offset, float spiral_angle_offset_per_ghost, # Circular mode parameters
    float glow_offset_x, float glow_offset_y, float glow_radius_offset_amount, # Glow effect parameters
    float r, float g, float b, float a, # RGBA color for this specific pass (pre-calculated in Python)
    bint is_glow_pass, # True if this is the glow pass, False for core pass
    bint is_current_trace # NEW PARAMETER: True if this is the current (newest) trace
):
    """
    Computes and fills vertex and color data for a single trace (ghost or current, glow or core pass)
    directly into the pre-allocated NumPy buffers.
    """
    cdef int k, sample_idx
    cdef float val, x_base, y_base, x_final, y_final
    cdef float current_scroll_dist
    cdef float current_spiral_offset_factor, current_spiral_radius_offset, current_spiral_angle_offset
    cdef float base_radius, amplitude_scale, angle, radius, radius_modifier

    # Determine line offset and radius modifier based on glow/core pass
    cdef float line_offset_x, line_offset_y
    cdef float line_radius_offset_amount # Used specifically for circular mode's glow effect
    
    if is_glow_pass:
        line_offset_x = glow_offset_x
        line_offset_y = glow_offset_y
        line_radius_offset_amount = glow_radius_offset_amount
    else:
        line_offset_x = 0.0
        line_offset_y = 0.0
        line_radius_offset_amount = 0.0

    # These constants are calculated once per function call
    base_radius = min(w, h) / 4.0
    amplitude_scale = min(w, h) / 8.0

    for k in range(points_to_draw):
        sample_idx = min(k * sample_step, mono_data.shape[0] - 1)
        val = mono_data[sample_idx]

        if current_mode == 0: # LINEAR_MODE
            # Only apply ghost scroll distance if it's NOT the current trace
            if is_current_trace:
                current_scroll_dist = 0.0 # No scroll for current trace
            else:
                current_scroll_dist = max_linear_scroll_dist * (1.0 - (<float>i / max(1.0, <float>num_traces)))
                if current_scroll_dist < 1.0: current_scroll_dist = 1.0 # Ensure minimum scroll

            x_base = k * (w / <float>points_to_draw) + current_scroll_dist
            y_base = (h / 2.0) + val * (h / 3.0) + current_scroll_dist # Apply to Y also for diagonal scroll
            
            x_final = x_base + line_offset_x
            y_final = y_base + line_offset_y
        
        else: # CIRCULAR_MODE
            # Only apply spiral offsets if it's NOT the current trace
            if is_current_trace:
                current_spiral_radius_offset = 0.0
                current_spiral_angle_offset = 0.0
            else:
                current_spiral_offset_factor = (1.0 - (<float>i / max(1.0, <float>num_traces)))
                current_spiral_radius_offset = max_spiral_radius_offset * current_spiral_offset_factor
                current_spiral_angle_offset = spiral_angle_offset_per_ghost * i
            
            angle = (<float>k / <float>points_to_draw) * (2.0 * math.pi) + current_spiral_angle_offset
            radius = base_radius + (val * amplitude_scale) + current_spiral_radius_offset
            
            # Apply glow radius offset *only if it's the glow pass*
            radius_modifier = radius + line_radius_offset_amount

            x_final = center_x + radius_modifier * math.cos(angle)
            y_final = center_y + radius_modifier * math.sin(angle)

        # Clamp values to screen bounds to prevent drawing outside widget (optional but good practice)
        if x_final < 0.0: x_final = 0.0
        if x_final > w - 1.0: x_final = w - 1.0
        if y_final < 0.0: y_final = 0.0
        if y_final > h - 1.0: y_final = h - 1.0

        vertices_buffer[start_buffer_offset + k, 0] = x_final
        vertices_buffer[start_buffer_offset + k, 1] = y_final
        
        colors_buffer[start_buffer_offset + k, 0] = r
        colors_buffer[start_buffer_offset + k, 1] = g
        colors_buffer[start_buffer_offset + k, 2] = b
        colors_buffer[start_buffer_offset + k, 3] = a

    # Return the number of points processed (useful for updating current_vertex_offset in Python)
    return points_to_draw