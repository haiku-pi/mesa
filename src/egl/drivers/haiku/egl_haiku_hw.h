/*
 * Copyright 2026, Haiku, Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 */
#ifndef EGL_HAIKU_HW_H
#define EGL_HAIKU_HW_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "util/format/u_formats.h"

struct pipe_screen;
struct pipe_resource;

/* A screen on the GPU's render node, or NULL (no such device, or software
 * rendering requested with LIBGL_ALWAYS_SOFTWARE). */
struct pipe_screen *haiku_create_hw_screen(void);

/* The calling team's memory at address (rows of stride bytes) as a linear
 * render target the GPU writes straight into, or NULL. *handle is the
 * driver's handle of it, for haiku_hw_sync_memory(). The memory must stay
 * allocated until the resource is gone. */
struct pipe_resource *haiku_hw_import_memory(struct pipe_screen *screen,
   void *address, unsigned width, unsigned height, unsigned stride,
   enum pipe_format format, uint32_t *handle);

/* Writes back and drops the CPU's cache lines of imported memory: before the
 * GPU writes it, and after, before the CPU reads it. */
void haiku_hw_sync_memory(struct pipe_screen *screen, uint32_t handle);

/* Waits until the GPU is done with every job submitted that uses the
 * imported memory. */
void haiku_hw_wait_memory(struct pipe_screen *screen, uint32_t handle);

#ifdef __cplusplus
}
#endif

#endif /* EGL_HAIKU_HW_H */
