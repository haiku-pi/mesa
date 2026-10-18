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
#include "v3d/drm/v3d_drm_public.h"

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
