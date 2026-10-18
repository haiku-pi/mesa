/*
 * Copyright 2026, Haiku, Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 */
#ifndef EGL_HAIKU_HW_H
#define EGL_HAIKU_HW_H

#ifdef __cplusplus
extern "C" {
#endif

struct pipe_screen;

/* A screen on the GPU's render node, or NULL (no such device, or software
 * rendering requested with LIBGL_ALWAYS_SOFTWARE). */
struct pipe_screen *haiku_create_hw_screen(void);

#ifdef __cplusplus
}
#endif

#endif /* EGL_HAIKU_HW_H */
