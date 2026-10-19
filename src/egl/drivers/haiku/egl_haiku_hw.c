/*
 * Copyright 2026, Haiku, Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 */

/*
 * Hardware screens for the Haiku EGL driver: a gallium DRM driver on the
 * render node of a Haiku kernel driver that speaks the Linux DRM render uAPI
 * (drivers/graphics/v3d publishes /dev/dri/renderD128). The node is opened by
 * path: Haiku has no DRM device enumeration.
 */

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include "pipe/p_screen.h"
#include "util/driconf.h"
#include "util/macros.h"
#include "util/u_debug.h"
#include "util/xmlconfig.h"

#include "egl_haiku_hw.h"

#ifdef GALLIUM_V3D
#include <string.h>
#include <xf86drm.h>

#include "drm-uapi/drm_fourcc.h"
#include "frontend/winsys_handle.h"
#include "pipe/p_state.h"
#include "drm-uapi/v3d_drm.h"
#include "v3d/drm/v3d_drm_public.h"

/* Haiku extensions of the v3d kernel driver (Haiku's
 * headers/private/drivers/drm/v3d_drm.h) */
struct haiku_v3d_import_memory {
   uint64_t address;
   uint64_t size;
   uint32_t handle;
   uint32_t offset;
};

struct haiku_v3d_sync_bo {
   uint32_t handle;
   uint32_t pad;
};

#define HAIKU_V3D_IOCTL_IMPORT_MEMORY \
   DRM_IOWR(DRM_COMMAND_BASE + 0x30, struct haiku_v3d_import_memory)
#define HAIKU_V3D_IOCTL_SYNC_BO \
   DRM_IOWR(DRM_COMMAND_BASE + 0x31, struct haiku_v3d_sync_bo)

static const driOptionDescription haiku_v3d_driconf[] = {
#include "pipe-loader/driinfo_gallium.h"
#include "v3d/driinfo_v3d.h"
};
#endif


struct pipe_screen *
haiku_create_hw_screen(void)
{
   if (debug_get_bool_option("LIBGL_ALWAYS_SOFTWARE", false))
      return NULL;

#ifdef GALLIUM_V3D
   int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   if (fd < 0)
      return NULL;

   /* The driver reads its options while it creates the screen. */
   driOptionCache info, options;
   driParseOptionInfo(&info, haiku_v3d_driconf, ARRAY_SIZE(haiku_v3d_driconf));
   struct pipe_screen_config config = {
      .options = &options,
      .options_info = &info,
   };

   /* The screen keeps its own duplicate of the descriptor. */
   struct pipe_screen *screen = v3d_drm_screen_create(fd, &config);
   close(fd);
   if (screen == NULL) {
      fprintf(stderr, "egl_haiku: no v3d screen on /dev/dri/renderD128, "
         "using software rendering\n");
   }

   driDestroyOptionCache(&options);
   driDestroyOptionInfo(&info);
   return screen;
#else
   return NULL;
#endif
}


struct pipe_resource *
haiku_hw_import_memory(struct pipe_screen *screen, void *address,
   unsigned width, unsigned height, unsigned stride, enum pipe_format format,
   uint32_t *handle)
{
#ifdef GALLIUM_V3D
   int fd = screen->get_screen_fd != NULL ? screen->get_screen_fd(screen) : -1;
   if (fd < 0)
      return NULL;

   struct haiku_v3d_import_memory import = {
      .address = (uintptr_t)address,
      .size = (uint64_t)stride * height,
   };
   if (drmIoctl(fd, HAIKU_V3D_IOCTL_IMPORT_MEMORY, &import) != 0)
      return NULL;

   struct winsys_handle whandle;
   memset(&whandle, 0, sizeof(whandle));
   whandle.type = WINSYS_HANDLE_TYPE_KMS;
   whandle.handle = import.handle;
   whandle.stride = stride;
   whandle.offset = (uintptr_t)address & 4095;
   whandle.modifier = DRM_FORMAT_MOD_LINEAR;

   struct pipe_resource templat;
   memset(&templat, 0, sizeof(templat));
   templat.target = PIPE_TEXTURE_2D;
   templat.format = format;
   templat.width0 = width;
   templat.height0 = height;
   templat.depth0 = 1;
   templat.array_size = 1;
   templat.bind = PIPE_BIND_RENDER_TARGET;

   struct pipe_resource *resource = screen->resource_from_handle(screen,
      &templat, &whandle, PIPE_HANDLE_USAGE_FRAMEBUFFER_WRITE);
   if (resource == NULL) {
      struct drm_gem_close close = { .handle = import.handle };
      drmIoctl(fd, DRM_IOCTL_GEM_CLOSE, &close);
      return NULL;
   }
   *handle = import.handle;
   return resource;
#else
   return NULL;
#endif
}


void
haiku_hw_sync_memory(struct pipe_screen *screen, uint32_t handle)
{
#ifdef GALLIUM_V3D
   int fd = screen->get_screen_fd != NULL ? screen->get_screen_fd(screen) : -1;
   if (fd < 0)
      return;
   struct haiku_v3d_sync_bo sync = { .handle = handle };
   drmIoctl(fd, HAIKU_V3D_IOCTL_SYNC_BO, &sync);
#endif
}


void
haiku_hw_wait_memory(struct pipe_screen *screen, uint32_t handle)
{
#ifdef GALLIUM_V3D
   int fd = screen->get_screen_fd != NULL ? screen->get_screen_fd(screen) : -1;
   if (fd < 0)
      return;
   struct drm_v3d_wait_bo wait = {
      .handle = handle,
      .timeout_ns = UINT64_MAX,
   };
   drmIoctl(fd, DRM_IOCTL_V3D_WAIT_BO, &wait);
#endif
}
