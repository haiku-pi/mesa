/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2014 Adrián Arroyo Calle <adrian.arroyocalle@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include "eglconfig.h"
#include "eglcontext.h"
#include "eglcurrent.h"
#include "egldevice.h"
#include "egldisplay.h"
#include "egldriver.h"
#include "eglimage.h"
#include "egllog.h"
#include "eglsurface.h"
#include "egltypedefs.h"

#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "state_tracker/st_context.h"
#include "util/u_atomic.h"
#include "util/box.h"
#include "util/u_inlines.h"
#include "util/os_time.h"
#include "mesa/glapi/glapi/glapi.h"

#include "hgl/hgl_sw_winsys.h"
#include "hgl_context.h"
#include "egl_haiku_hw.h"

#include <Application.h>
#include <Bitmap.h>
#include <Screen.h>

extern "C" {
#include "target-helpers/inline_sw_helper.h"
}

#if MESA_DEBUG
#define TRACE(x...) printf("egl_haiku: " x)
#define CALLED()    TRACE("CALLED: %s\n", __PRETTY_FUNCTION__)
#else
#define TRACE(x...)
#define CALLED()
#endif
#define ERROR(x...) printf("egl_haiku: " x)

_EGL_DRIVER_STANDARD_TYPECASTS(haiku_egl)

struct haiku_egl_display {
   int ref_count;
   struct hgl_display *disp;
   bool hardware;
};

struct haiku_egl_config {
   _EGLConfig base;
};

struct haiku_egl_context {
   _EGLContext base;
   struct hgl_context *ctx;
};

// Hardware screens: a window's bitmap as a render target of the GPU (the
// bitmap's own memory, so presenting needs no copy by the CPU)
struct imported_bitmap {
   BBitmap *bitmap;
   struct pipe_resource *resource;
   uint32_t handle;
   // overlay bitmaps: their memory is a kernel area the client may access;
   // the GPU imports a clone of it in our team (at the bitmap's Bits() then)
   area_id clone;
   void *bits;
};

// A surface's bitmaps take turns: one on screen (the view's), one with the
// presenter, one the GPU writes the next frame into. With a third, the render
// thread never waits for the presenter before it hands the GPU the next frame.
static const int kSyncBitmaps = 3;
// Unsynchronized overlay frames (swap interval 0) go through a mailbox: one
// bitmap on screen, one flipped and waiting for the firmware's update point,
// the newest finished frame waiting for the next vertical blank (replaced by
// a newer one: superseded, free at once), one being rendered.
static const int kMaxBitmaps = 4;

/* Overlay bitmaps (B_BITMAP_WILL_OVERLAY) are scanned out by the display: a
 * present flips the view's overlay to the next one. The firmware of the
 * Raspberry Pi takes a plane change at an update point ~1.6 ms before the
 * vertical blank (measured), so a bitmap replaced on screen at time T is no
 * longer scanned out after a vertical blank that was waited for from
 * T + kRetireGuard on (the retrace semaphore wakes its waiters at the blank).
 * Only then may the GPU write it again. */
static const bigtime_t kRetireGuard = 3000;

struct haiku_egl_surface {
   _EGLSurface base;
   struct hgl_buffer *fb;
   struct pipe_fence_handle *throttle_fence;
   // hardware screens: the bitmaps neither on screen nor with the presenter
   // (render thread only)
   BBitmap *free_bitmaps[kMaxBitmaps];
   int free_count;
   int bitmap_count; // all of them, on screen and presenting included
   struct imported_bitmap imported[kMaxBitmaps];
   bool no_import; // the driver cannot import them: copy
   // The presenter thread (imported bitmaps): it waits for the GPU's copy of a
   // swap and hands the bitmap to the view, while the client renders the next
   // frame. The render thread waits for it (presented_sem) before it hands it
   // the next frame, or when it has no free bitmap.
   thread_id presenter; // 0: not started yet, < 0: none (hand over in place)
   sem_id present_sem;
   sem_id presented_sem;
   bool presenting; // a frame is with the presenter (render thread only)
   bool quit;
   struct pipe_screen *present_screen;
   BBitmap *pending_bitmap;
   struct imported_bitmap *pending_slot;
   BBitmap *returned_bitmap; // the presenter's: the one the view gave back
   bigtime_t returned_time; // when it left the screen (0: it was not shown)
   int swap_interval; // the frame's (render thread)
   int pending_interval; // pending_bitmap's, for the presenter

   // the mailbox (swap interval 0, overlay bitmaps): the newest finished
   // frame, which the presenter flips after the next vertical blank, and the
   // bitmaps the flips replaced, for the render thread; under mailbox_lock
   pthread_mutex_t mailbox_lock;
   BBitmap *mailbox_bitmap;
   struct imported_bitmap *mailbox_slot;
   BBitmap *mailbox_returned[kMaxBitmaps];
   bigtime_t mailbox_returned_time[kMaxBitmaps];
   int mailbox_returned_count;
   sem_id mailbox_returned_sem; // released for each bitmap returned

   // overlay bitmaps
   bool overlay_refused; // not for this size: overlay_width x overlay_height
   unsigned overlay_width, overlay_height;
   int overlay_count;
   BBitmap *retiring[kMaxBitmaps]; // off screen at a vertical blank after
   bigtime_t retired_at[kMaxBitmaps]; // ... retired_at + kRetireGuard
   int retiring_count;
   int64 waited_from; // (atomic) a vertical blank was waited for from then on
   BScreen *screen; // its retrace semaphore
   // the presenter's flip statistics (HAIKU_EGL_FPS)
   bigtime_t last_flip; // when the last flip returned
   bigtime_t last_wake; // when the presenter woke for it (synchronized)
   bigtime_t vblank_period;
   uint32_t flips, missed, starved, late_flips, drops, superseded;
   // HAIKU_EGL_FPS: frames presented since fps_start (us), printed every 5 s,
   // with the render thread's waits for the presenter and the GPU
   uint32_t fps_frames;
   bigtime_t fps_start;
   bigtime_t presenter_wait;
   bigtime_t throttle_wait;
};

/* The next frame, as the render thread prepares it before it is presented. */
struct haiku_present {
   BBitmap *bitmap; // NULL: nothing to present
   struct imported_bitmap *slot; // NULL: the CPU copies the frame
};

static void release_import(struct haiku_egl_surface *surface, BBitmap *bitmap);
static void wait_presented(struct haiku_egl_surface *surface);
static void put_free_bitmap(struct haiku_egl_surface *surface, BBitmap *bitmap);

/* A signal interrupts a semaphore wait (B_INTERRUPTED): the program's handler
 * runs on whichever thread took it, the presenter or the render thread. Neither
 * may take that for its partner's release -- the presenter quit then, and the
 * render thread waited for it at the next swap, for ever. */
static status_t
acquire_sem_uninterrupted(sem_id sem)
{
   status_t status;
   do {
      status = acquire_sem(sem);
   } while (status == B_INTERRUPTED);
   return status;
}

/* Whether a native window shows overlay bitmaps itself, on an overlay channel
 * (BGLView's, libglvnd's libGL says): any other BitmapHook would draw them,
 * reading the display's write-combining memory, so it gets none. */
static bool
native_window_shows_overlays(void *native_window)
{
   typedef bool (*shows_overlays_func)(const void *);
   static const shows_overlays_func showsOverlays = (shows_overlays_func)
      dlsym(RTLD_DEFAULT, "hgl_native_window_shows_overlays");
   return showsOverlays != NULL && showsOverlays(native_window);
}

// #pragma mark EGLSurface

// Called via eglCreateWindowSurface(), drv->CreateWindowSurface().
static _EGLSurface *
haiku_create_window_surface(_EGLDisplay *disp, _EGLConfig *conf,
                            void *native_window, const EGLint *attrib_list)
{
   printf("haiku_create_window_surface\n");
   struct haiku_egl_display *hgl_dpy = haiku_egl_display(disp);

   struct haiku_egl_surface *wgl_surf =
      (struct haiku_egl_surface *)calloc(1, sizeof(*wgl_surf));
   if (!wgl_surf)
      return NULL;

   if (!_eglInitSurface(&wgl_surf->base, disp, EGL_WINDOW_BIT, conf,
                        attrib_list, NULL)) {
      free(wgl_surf);
      return NULL;
   }

   struct st_visual visual;
   hgl_get_st_visual(&visual, HGL_DOUBLE | HGL_DEPTH);

   wgl_surf->fb =
      hgl_create_st_framebuffer(hgl_dpy->disp, &visual, native_window);
   if (!wgl_surf->fb) {
      free(wgl_surf);
      return NULL;
   }

   // Unset and delete previously set bitmap if any.
   delete ((BitmapHook *)native_window)->SetBitmap(NULL);

   for (auto &import : wgl_surf->imported)
      import.clone = -1;
   wgl_surf->vblank_period = 16667;
   pthread_mutex_init(&wgl_surf->mailbox_lock, NULL);
   wgl_surf->mailbox_returned_sem = -1;
   if (hgl_dpy->hardware && be_app != NULL
       && native_window_shows_overlays(native_window)) {
      // for overlay frames: the screen's size and its retrace semaphore (the
      // first wait asks app_server for it)
      wgl_surf->screen = new BScreen(B_MAIN_SCREEN_ID);
      if (wgl_surf->screen->IsValid())
         wgl_surf->screen->WaitForRetrace(0);
      else {
         delete wgl_surf->screen;
         wgl_surf->screen = NULL;
      }
   }

   return &wgl_surf->base;
}

static _EGLSurface *
haiku_create_pixmap_surface(_EGLDisplay *disp, _EGLConfig *conf,
                            void *native_pixmap, const EGLint *attrib_list)
{
   printf("haiku_create_pixmap_surface\n");
   return NULL;
}

static _EGLSurface *
haiku_create_pbuffer_surface(_EGLDisplay *disp, _EGLConfig *conf,
                             const EGLint *attrib_list)
{
   printf("haiku_create_pbuffer_surface\n");
   struct haiku_egl_display *hgl_dpy = haiku_egl_display(disp);

   struct haiku_egl_surface *wgl_surf =
      (struct haiku_egl_surface *)calloc(1, sizeof(*wgl_surf));
   if (!wgl_surf)
      return NULL;

   if (!_eglInitSurface(&wgl_surf->base, disp, EGL_PBUFFER_BIT, conf,
                        attrib_list, NULL)) {
      free(wgl_surf);
      return NULL;
   }

   struct st_visual visual;
   hgl_get_st_visual(&visual, HGL_DOUBLE | HGL_DEPTH);

   wgl_surf->fb = hgl_create_st_framebuffer(hgl_dpy->disp, &visual, NULL);
   if (!wgl_surf->fb) {
      free(wgl_surf);
      return NULL;
   }

   wgl_surf->fb->newWidth = wgl_surf->base.Width;
   wgl_surf->fb->newHeight = wgl_surf->base.Height;
   p_atomic_inc(&wgl_surf->fb->base.stamp);

   return &wgl_surf->base;
}

static EGLBoolean
haiku_destroy_surface(_EGLDisplay *disp, _EGLSurface *surf)
{
   struct haiku_egl_display *hgl_dpy = haiku_egl_display(disp);
   if (_eglPutSurface(surf)) {
      struct haiku_egl_surface *hgl_surf = haiku_egl_surface(surf);
      struct pipe_screen *screen = hgl_dpy->disp->fscreen->screen;
      screen->fence_reference(screen, &hgl_surf->throttle_fence, NULL);

      if (hgl_surf->presenter > 0) {
         if (hgl_surf->presenting)
            wait_presented(hgl_surf);
         hgl_surf->quit = true;
         release_sem(hgl_surf->present_sem);
         status_t result;
         while (wait_for_thread(hgl_surf->presenter, &result) == B_INTERRUPTED)
            ; // the presenter uses the surface until it has returned
         delete_sem(hgl_surf->present_sem);
         delete_sem(hgl_surf->presented_sem);
         delete_sem(hgl_surf->mailbox_returned_sem);
      }
      // what the presenter left in the mailbox
      if (hgl_surf->mailbox_bitmap != NULL)
         put_free_bitmap(hgl_surf, hgl_surf->mailbox_bitmap);
      for (int i = 0; i < hgl_surf->mailbox_returned_count; i++)
         put_free_bitmap(hgl_surf, hgl_surf->mailbox_returned[i]);
      pthread_mutex_destroy(&hgl_surf->mailbox_lock);

      // Unset bitmap to release ownership. Bitmap will be deleted later
      // when destroying framebuffer.
      BitmapHook *bitmapHook = (BitmapHook*)hgl_surf->fb->winsysContext;
      release_import(hgl_surf, NULL);
      if (bitmapHook != NULL) {
         BBitmap *shown = bitmapHook->SetBitmap(NULL);
         if (hgl_dpy->hardware)
            delete shown;
      }
      for (int i = 0; i < hgl_surf->free_count; i++)
         delete hgl_surf->free_bitmaps[i];
      for (int i = 0; i < hgl_surf->retiring_count; i++)
         delete hgl_surf->retiring[i];
      delete hgl_surf->screen;

      hgl_destroy_st_framebuffer(hgl_surf->fb);
      free(surf);
   }
   return EGL_TRUE;
}

static void
update_size(struct hgl_buffer *buffer)
{
   if (buffer->winsysContext == NULL)
      return; // a pbuffer: its size is fixed

   uint32_t newWidth, newHeight;
   ((BitmapHook *)buffer->winsysContext)->GetSize(newWidth, newHeight);
   if (buffer->newWidth != newWidth || buffer->newHeight != newHeight) {
      buffer->newWidth = newWidth;
      buffer->newHeight = newHeight;
      p_atomic_inc(&buffer->base.stamp);
   }
}

static bool
is_overlay(const BBitmap *bitmap)
{
   return bitmap != NULL && (bitmap->Flags() & B_BITMAP_WILL_OVERLAY) != 0;
}

/* Forgets a bitmap's import (before the bitmap goes: its memory stays wired
 * while the GPU may use it). An overlay buffer goes back to the display
 * driver with its bitmap and may be handed to another client at once: the
 * GPU is done with it first. */
static void
release_import(struct haiku_egl_surface *surface, BBitmap *bitmap)
{
   for (auto &import : surface->imported) {
      if (import.resource != NULL && (bitmap == NULL || import.bitmap == bitmap)) {
         if (import.clone > 0)
            haiku_hw_wait_memory(import.resource->screen, import.handle);
         pipe_resource_reference(&import.resource, NULL);
         if (import.clone > 0)
            delete_area(import.clone);
         import.clone = -1;
         import.bits = NULL;
         import.bitmap = NULL;
      }
   }
}

static void
delete_bitmap(struct haiku_egl_surface *surface, BBitmap *bitmap)
{
   release_import(surface, bitmap);
   if (is_overlay(bitmap))
      surface->overlay_count--;
   delete bitmap;
   surface->bitmap_count--;
}

/* A bitmap is free again (render thread only). */
static void
put_free_bitmap(struct haiku_egl_surface *surface, BBitmap *bitmap)
{
   if (bitmap == NULL)
      return;
   if (surface->free_count < kMaxBitmaps) {
      surface->free_bitmaps[surface->free_count++] = bitmap;
      return;
   }
   delete_bitmap(surface, bitmap);
}

/* A bitmap the view gave back (render thread only): free now, or, an overlay
 * bitmap that left the screen at time left, once it is no longer scanned
 * out. */
static void
put_returned_bitmap(struct haiku_egl_surface *surface, BBitmap *bitmap,
                    bigtime_t left)
{
   if (bitmap == NULL)
      return;
   if (left == 0 || surface->retiring_count == kMaxBitmaps) {
      put_free_bitmap(surface, bitmap);
      return;
   }
   surface->retiring[surface->retiring_count] = bitmap;
   surface->retired_at[surface->retiring_count++] = left;
}

/* Moves the retiring bitmaps that are off screen to the free ones; first
 * takes those the presenter's mailbox flips gave back. */
static void
collect_retired(struct haiku_egl_surface *surface)
{
   if (surface->mailbox_returned_count > 0) {
      pthread_mutex_lock(&surface->mailbox_lock);
      for (int i = 0; i < surface->mailbox_returned_count; i++) {
         put_returned_bitmap(surface, surface->mailbox_returned[i],
                             surface->mailbox_returned_time[i]);
      }
      surface->mailbox_returned_count = 0;
      pthread_mutex_unlock(&surface->mailbox_lock);
   }

   const bigtime_t waitedFrom = atomic_get64(&surface->waited_from);
   for (int i = 0; i < surface->retiring_count;) {
      if (surface->retired_at[i] + kRetireGuard <= waitedFrom) {
         put_free_bitmap(surface, surface->retiring[i]);
         surface->retiring_count--;
         surface->retiring[i] = surface->retiring[surface->retiring_count];
         surface->retired_at[i] = surface->retired_at[surface->retiring_count];
      } else
         i++;
   }
}

/* Waits for the next vertical blank, from time from on; returns when it woke.
 * Both threads may wait: whoever does advances waited_from. */
static bigtime_t
wait_vblank(struct haiku_egl_surface *surface, bigtime_t from)
{
   if (system_time() < from)
      snooze_until(from, B_SYSTEM_TIMEBASE);
   const bigtime_t start = system_time();
   status_t status = B_ERROR;
   if (surface->screen != NULL) {
      do {
         status = surface->screen->WaitForRetrace(100000);
      } while (status == B_INTERRUPTED);
   }
   if (status != B_OK) {
      // no retrace semaphore, or it stopped: a frame's time holds a blank
      snooze_until(start + 20000, B_SYSTEM_TIMEBASE);
   }

   int64 old = atomic_get64(&surface->waited_from);
   while (old < start) {
      const int64 previous = atomic_test_and_set64(&surface->waited_from,
                                                   start, old);
      if (previous == old)
         break;
      old = previous;
   }
   return system_time();
}

/* Waits until the presenter has handed its frame to the view, and takes back
 * the bitmap the view gave up for it. */
static void
wait_presented(struct haiku_egl_surface *surface)
{
   acquire_sem_uninterrupted(surface->presented_sem);
   surface->presenting = false;
   put_returned_bitmap(surface, surface->returned_bitmap,
                       surface->returned_time);
   surface->returned_bitmap = NULL;
}

/* Whether frames of width x height go to overlay bitmaps (scanned out, no
 * copy): HAIKU_EGL_OVERLAY=0 never, =1 always, by default when they cover the
 * screen. Decided again when the size changes. */
static bool
want_overlay(struct haiku_egl_surface *surface, unsigned width,
             unsigned height)
{
   static const char *mode = getenv("HAIKU_EGL_OVERLAY");
   if (surface->screen == NULL || surface->no_import
       || (mode != NULL && strcmp(mode, "0") == 0))
      return false;
   if (surface->overlay_refused && width == surface->overlay_width
       && height == surface->overlay_height)
      return false;
   if (width != surface->overlay_width || height != surface->overlay_height) {
      surface->overlay_width = width;
      surface->overlay_height = height;
      BRect frame = surface->screen->Frame();
      surface->overlay_refused = !(mode != NULL && strcmp(mode, "1") == 0)
         && (width != (unsigned)frame.IntegerWidth() + 1
             || height != (unsigned)frame.IntegerHeight() + 1);
   }
   return !surface->overlay_refused;
}

/* A new bitmap for a frame: an overlay bitmap if wanted and one can be had
 * (the first takes the view's overlay channel), else an ordinary one. */
static BBitmap *
new_bitmap(struct haiku_egl_surface *surface, unsigned width, unsigned height,
           bool overlay)
{
   const BRect bounds(0, 0, width - 1, height - 1);
   if (overlay) {
      uint32 flags = B_BITMAP_WILL_OVERLAY;
      if (surface->overlay_count == 0)
         flags |= B_BITMAP_RESERVE_OVERLAY_CHANNEL;
      BBitmap *bitmap = new BBitmap(bounds, flags, B_RGB32);
      if (bitmap->InitCheck() == B_OK && bitmap->Bits() != NULL) {
         surface->overlay_count++;
         surface->bitmap_count++;
         return bitmap;
      }
      delete bitmap;
      surface->overlay_refused = true;
      ERROR("no overlay for %ux%u frames: presenting through the window\n",
            width, height);
   }

   // B_RGB32 is B, G, R, X in memory: the visual's BGRX/BGRA8888 format
   BBitmap *bitmap = new BBitmap(bounds, B_RGB32);
   if (bitmap->InitCheck() != B_OK) {
      delete bitmap;
      return NULL;
   }
   surface->bitmap_count++;
   return bitmap;
}

/* A bitmap of width x height for the next frame: a free one, a new one while
 * the surface has fewer than kMaxBitmaps, the one the presenter gets back, or
 * a retiring one once it is off screen. NULL if none can be allocated. */
static BBitmap *
take_bitmap(struct haiku_egl_surface *surface, unsigned width, unsigned height)
{
   const bool overlay = want_overlay(surface, width, height);
   const int limit = overlay && surface->swap_interval == 0
      ? kMaxBitmaps : kSyncBitmaps;
   while (true) {
      collect_retired(surface);
      if (surface->free_count > 0) {
         BBitmap *bitmap = surface->free_bitmaps[--surface->free_count];
         if ((unsigned)bitmap->Bounds().IntegerWidth() + 1 == width
             && (unsigned)bitmap->Bounds().IntegerHeight() + 1 == height
             && is_overlay(bitmap) == overlay)
            return bitmap;
         // the window was resized, or overlays came or went
         delete_bitmap(surface, bitmap);
         continue;
      }
      if (surface->bitmap_count < limit)
         return new_bitmap(surface, width, height, overlay);

      const bigtime_t start = system_time();
      if (surface->presenting)
         wait_presented(surface);
      else if (surface->retiring_count == 0 && surface->mailbox_returned_sem >= 0) {
         // the mailbox's next flip gives one back (a frame's time at most)
         acquire_sem_etc(surface->mailbox_returned_sem, 1, B_RELATIVE_TIMEOUT,
                         50000);
      } else if (surface->retiring_count > 0) {
         bigtime_t oldest = surface->retired_at[0];
         for (int i = 1; i < surface->retiring_count; i++)
            oldest = std::min(oldest, surface->retired_at[i]);
         wait_vblank(surface, oldest + kRetireGuard);
      } else
         return NULL; // all bitmaps on screen: cannot happen
      surface->presenter_wait += system_time() - start;
   }
}

/*
 * The GPU copies the frame into the bitmap's own memory, which it imported
 * once: no CPU copy, no read of the GPU's write-combining memory. Returns
 * NULL when the bitmap cannot be imported (then the CPU copies).
 *
 * Nothing on the CPU writes the bitmap after it is imported (app_server only
 * reads it), so its cache lines are written back and dropped once, then: a
 * clean line is never written back over what the GPU wrote, and the presenter
 * drops the stale ones after each copy. HAIKU_EGL_SYNC_EACH_FRAME=1 does it
 * before every copy again, as before. Overlay bitmaps are write-combining
 * everywhere: no cache to maintain.
 *
 * An overlay bitmap's memory is a kernel area every team may access; the GPU
 * imports a clone of it in our team. The display driver may move it (a mode
 * change suspends overlays): a new Bits() is imported again.
 */
static struct imported_bitmap *
haiku_present_gpu(struct pipe_context *pipe, struct pipe_resource *resource,
                  struct haiku_egl_surface *surface, BBitmap *bitmap)
{
   if (surface->no_import)
      return NULL;
   static const bool disabled = getenv("HAIKU_EGL_NO_IMPORT") != NULL;
   if (disabled)
      return NULL;
   static const bool syncEachFrame
      = getenv("HAIKU_EGL_SYNC_EACH_FRAME") != NULL;

   const bool overlay = is_overlay(bitmap);
   void *bits = bitmap->Bits();
   if (bits == NULL)
      return NULL;

   struct pipe_screen *screen = pipe->screen;
   struct imported_bitmap *slot = NULL;
   for (auto &import : surface->imported) {
      if (import.resource != NULL && import.bitmap == bitmap)
         slot = &import;
   }
   if (slot != NULL && overlay && slot->bits != bits) {
      // the overlay buffer moved
      release_import(surface, bitmap);
      slot = NULL;
   }
   bool imported = false;
   if (slot == NULL) {
      for (auto &import : surface->imported) {
         if (import.resource == NULL) {
            slot = &import;
            break;
         }
      }
      if (slot == NULL)
         return NULL; // more bitmaps than slots: cannot happen

      void *address = bits;
      area_id clone = -1;
      if (overlay) {
         area_info info;
         const area_id area = area_for(bits);
         if (area < 0 || get_area_info(area, &info) != B_OK)
            return NULL;
         void *cloneAddress;
         clone = clone_area("egl overlay bitmap", &cloneAddress, B_ANY_ADDRESS,
                            B_READ_AREA, area);
         if (clone < 0)
            return NULL;
         address = (uint8_t *)cloneAddress
            + ((uint8_t *)bits - (uint8_t *)info.address);
      }

      slot->resource = haiku_hw_import_memory(screen, address,
         resource->width0, resource->height0, bitmap->BytesPerRow(),
         resource->format, &slot->handle);
      if (slot->resource == NULL) {
         if (clone >= 0)
            delete_area(clone);
         if (overlay) {
            ERROR("cannot import an overlay bitmap: copying into it\n");
            return NULL;
         }
         ERROR("cannot import the window's bitmap: presenting by copying\n");
         surface->no_import = true;
         return NULL;
      }
      slot->bitmap = bitmap;
      slot->clone = clone;
      slot->bits = bits;
      imported = true;
   }

   // nothing the CPU cached of the bitmap may be written back over the frame
   if (!overlay && (imported || syncEachFrame))
      haiku_hw_sync_memory(screen, slot->handle);

   struct pipe_blit_info blit;
   memset(&blit, 0, sizeof(blit));
   blit.src.resource = resource;
   blit.src.format = resource->format;
   u_box_2d(0, 0, resource->width0, resource->height0, &blit.src.box);
   blit.dst.resource = slot->resource;
   blit.dst.format = slot->resource->format;
   blit.dst.box = blit.src.box;
   blit.mask = PIPE_MASK_RGBA;
   blit.filter = PIPE_TEX_FILTER_NEAREST;
   pipe->blit(pipe, &blit);
   return slot;
}

/*
 * Synchronized overlay frames (swap interval 1): the flip goes right after a
 * vertical blank, a frame before the firmware's next update point. The wait
 * starts no earlier than kRetireGuard after the last flip, so the bitmap that
 * flip replaced is off screen once it returns. Counts the blanks between
 * flips: one is on time; more than the frame needed (it was ready in time)
 * are missed flips, the others the GPU's (starved).
 */
static void
wait_flip_slot(struct haiku_egl_surface *surface, bigtime_t ready)
{
   const bigtime_t from = surface->last_flip + kRetireGuard;
   const bigtime_t wake = wait_vblank(surface, from);
   if (surface->last_wake != 0) {
      const bigtime_t period = surface->vblank_period;
      const bigtime_t interval = wake - surface->last_wake;
      const int64 blanks = std::max<int64>((interval + period / 2) / period, 1);
      if (blanks == 1)
         surface->vblank_period = (period * 15 + interval) / 16;
      // the first blank the frame could have used
      const bigtime_t earliest = std::max(ready, from);
      int64 needed = 1;
      while (surface->last_wake + needed * period - period / 4 < earliest)
         needed++;
      if (blanks > needed)
         surface->missed += blanks - needed;
      if (needed > 1)
         surface->starved += std::min(blanks, needed) - 1;
   }
   surface->last_wake = wake;
}

/*
 * The presenter thread: hands each blitted bitmap to the view once the GPU is
 * done with it, after dropping the CPU's stale cache lines (app_server reads
 * it next). An overlay bitmap is flipped onto the screen, at the next
 * vertical blank with swap interval 1; the bitmap it replaces is returned
 * with the time it left the screen.
 */
/*
 * A mailbox frame (swap interval 0): the flip waits for the next vertical
 * blank (from kRetireGuard after the last flip on, so that the bitmap the last
 * flip replaced is off screen then), and shows the newest frame finished by
 * then. One flip per blank: the frames the render thread finishes meanwhile
 * replace each other in the mailbox (superseded), never scanned out, and the
 * render thread need not wait for the display.
 */
static void
present_mailbox(struct haiku_egl_surface *surface, BitmapHook *hook)
{
   wait_vblank(surface, surface->last_flip + kRetireGuard);

   pthread_mutex_lock(&surface->mailbox_lock);
   BBitmap *bitmap = surface->mailbox_bitmap;
   struct imported_bitmap *slot = surface->mailbox_slot;
   surface->mailbox_bitmap = NULL;
   pthread_mutex_unlock(&surface->mailbox_lock);
   if (bitmap == NULL)
      return;

   haiku_hw_wait_memory(surface->present_screen, slot->handle);
   BBitmap *old = hook->SetBitmap(bitmap);
   const bigtime_t now = system_time();
   bigtime_t left = 0;
   if (old == bitmap)
      surface->drops++;
   else {
      surface->flips++;
      surface->last_flip = now;
      left = is_overlay(old) ? now : 0;
   }

   if (old != NULL) {
      pthread_mutex_lock(&surface->mailbox_lock);
      surface->mailbox_returned[surface->mailbox_returned_count] = old;
      surface->mailbox_returned_time[surface->mailbox_returned_count++] = left;
      pthread_mutex_unlock(&surface->mailbox_lock);
      release_sem(surface->mailbox_returned_sem);
   }
}

static status_t
haiku_presenter(void *data)
{
   struct haiku_egl_surface *surface = (struct haiku_egl_surface *)data;
   BitmapHook *hook = (BitmapHook *)surface->fb->winsysContext;
   while (acquire_sem_uninterrupted(surface->present_sem) == B_OK
          && !surface->quit) {
      // each release of present_sem is one frame: a mailbox frame, or then
      // the synchronous pending_bitmap
      pthread_mutex_lock(&surface->mailbox_lock);
      const bool mailbox = surface->mailbox_bitmap != NULL;
      pthread_mutex_unlock(&surface->mailbox_lock);
      if (mailbox) {
         present_mailbox(surface, hook);
         continue;
      }

      BBitmap *bitmap = surface->pending_bitmap;
      haiku_hw_wait_memory(surface->present_screen, surface->pending_slot->handle);
      if (!is_overlay(bitmap)) {
         haiku_hw_sync_memory(surface->present_screen,
                              surface->pending_slot->handle);
         surface->returned_bitmap = hook->SetBitmap(bitmap);
         surface->returned_time = 0;
         release_sem(surface->presented_sem);
         continue;
      }

      const bigtime_t ready = system_time();
      if (surface->pending_interval > 0)
         wait_flip_slot(surface, ready);
      BBitmap *old = hook->SetBitmap(bitmap);
      const bigtime_t now = system_time();
      if (old == bitmap) {
         // not shown (the window was busy): free again
         surface->drops++;
         surface->returned_time = 0;
      } else {
         surface->flips++;
         if (surface->pending_interval > 0
             && now - surface->last_wake > surface->vblank_period - kRetireGuard)
            surface->late_flips++;
         surface->last_flip = now;
         surface->returned_time = is_overlay(old) ? now : 0;
      }
      surface->returned_bitmap = old;
      release_sem(surface->presented_sem);
   }
   return B_OK;
}

/* Hands a bitmap to the view from the render thread (no presenter). */
static void
hand_to_view(struct haiku_egl_surface *surface, BitmapHook *hook,
             BBitmap *bitmap)
{
   BBitmap *old = hook->SetBitmap(bitmap);
   if (old == bitmap) {
      surface->drops++;
      put_free_bitmap(surface, bitmap);
      return;
   }
   if (is_overlay(bitmap))
      surface->flips++;
   put_returned_bitmap(surface, old, is_overlay(old) ? system_time() : 0);
}

/*
 * Hardware screens have no flush_frontbuffer: the frame is in GPU memory. It
 * goes to the view in a BBitmap, which the view draws (BitmapHook), or shows
 * on an overlay (an overlay bitmap).
 *
 * First half: takes a bitmap for the frame and has the GPU copy the frame
 * into it. Called before the frame's rendering is flushed, the copy is a
 * second store of the same render job (v3d's TLB blit), not a job of its own
 * that loads the whole frame again: HAIKU_EGL_SEPARATE_BLIT=1 calls it after
 * the flush, as before.
 */
static struct haiku_present
haiku_present_prepare(struct pipe_context *pipe, struct pipe_resource *resource,
                      struct haiku_egl_surface *surface)
{
   struct haiku_present present = { NULL, NULL };
   if (surface->fb->winsysContext == NULL)
      return present; // a pbuffer

   present.bitmap = take_bitmap(surface, resource->width0, resource->height0);
   if (present.bitmap != NULL)
      present.slot = haiku_present_gpu(pipe, resource, surface, present.bitmap);
   return present;
}

/* Second half, after the flush: hands the frame to the presenter, or copies
 * it with the CPU. */
static void
haiku_present_finish(struct pipe_context *pipe, struct pipe_resource *resource,
                     struct haiku_egl_surface *surface,
                     const struct haiku_present &present)
{
   BitmapHook *hook = (BitmapHook *)surface->fb->winsysContext;
   BBitmap *bitmap = present.bitmap;
   if (bitmap == NULL)
      return;

   struct imported_bitmap *slot = present.slot;
   if (slot != NULL && surface->presenter == 0) {
      surface->present_sem = create_sem(0, "egl present");
      surface->presented_sem = create_sem(0, "egl presented");
      surface->mailbox_returned_sem = create_sem(0, "egl mailbox returned");
      surface->present_screen = pipe->screen;
      surface->presenter = spawn_thread(haiku_presenter, "egl presenter",
                                        B_DISPLAY_PRIORITY, surface);
      if (surface->present_sem < 0 || surface->presented_sem < 0
          || surface->presenter < 0 || resume_thread(surface->presenter) != B_OK) {
         if (surface->presenter > 0)
            kill_thread(surface->presenter);
         delete_sem(surface->present_sem);
         delete_sem(surface->presented_sem);
         delete_sem(surface->mailbox_returned_sem);
         surface->mailbox_returned_sem = -1;
         surface->presenter = -1;
      }
   }

   // the frames reach the view in order: the previous one first
   if (surface->presenting) {
      const bigtime_t start = system_time();
      wait_presented(surface);
      surface->presenter_wait += system_time() - start;
   }

   if (slot != NULL && surface->presenter < 0) {
      // no presenter: hand the bitmap over here
      haiku_hw_wait_memory(pipe->screen, slot->handle);
      if (!is_overlay(bitmap))
         haiku_hw_sync_memory(pipe->screen, slot->handle);
      hand_to_view(surface, hook, bitmap);
      return;
   }
   if (slot != NULL && is_overlay(bitmap) && surface->swap_interval == 0) {
      // the mailbox: replaces a frame still waiting for its vertical blank
      pthread_mutex_lock(&surface->mailbox_lock);
      BBitmap *superseded = surface->mailbox_bitmap;
      surface->mailbox_bitmap = bitmap;
      surface->mailbox_slot = slot;
      pthread_mutex_unlock(&surface->mailbox_lock);
      if (superseded != NULL) {
         // never shown: free at once (the GPU writes it again only in a job
         // after the one that wrote it)
         surface->superseded++;
         put_free_bitmap(surface, superseded);
      } else
         release_sem(surface->present_sem);
      return;
   }
   if (slot != NULL) {
      surface->pending_bitmap = bitmap;
      surface->pending_slot = slot;
      surface->pending_interval = surface->swap_interval;
      surface->presenting = true;
      release_sem(surface->present_sem);
      return;
   }

   uint8_t *target = (uint8_t *)bitmap->Bits();
   if (target == NULL) {
      // an overlay bitmap whose buffer is suspended (a mode change)
      put_free_bitmap(surface, bitmap);
      return;
   }

   struct pipe_box box;
   u_box_2d(0, 0, resource->width0, resource->height0, &box);
   struct pipe_transfer *transfer;
   const uint8_t *source = (const uint8_t *)pipe->texture_map(pipe, resource,
      0, PIPE_MAP_READ, &box, &transfer);
   if (source == NULL) {
      put_free_bitmap(surface, bitmap);
      return;
   }

   const size_t rowLength = std::min((size_t)bitmap->BytesPerRow(),
      (size_t)resource->width0 * 4);
   for (unsigned y = 0; y < resource->height0; y++) {
      memcpy(target + y * bitmap->BytesPerRow(), source + y * transfer->stride,
         rowLength);
   }
   pipe->texture_unmap(pipe, transfer);

   hand_to_view(surface, hook, bitmap);
}

struct haiku_present_args {
   struct pipe_context *pipe;
   struct pipe_resource *resource;
   struct haiku_egl_surface *surface;
   struct haiku_present present;
};

/* st_context_flush()'s callback: the frame is complete, its job not yet
 * submitted. */
static void
haiku_present_before_flush(void *data)
{
   struct haiku_present_args *args = (struct haiku_present_args *)data;
   args->present = haiku_present_prepare(args->pipe, args->resource,
                                         args->surface);
}

static EGLBoolean
haiku_swap_buffers(_EGLDisplay *disp, _EGLSurface *surf)
{
   struct haiku_egl_display *hgl_dpy = haiku_egl_display(disp);
   struct haiku_egl_surface *hgl_surf = haiku_egl_surface(surf);
   struct haiku_egl_context *hgl_ctx = haiku_egl_context(surf->CurrentContext);
   if (hgl_ctx == NULL)
      return EGL_FALSE;

   struct st_context *st = hgl_ctx->ctx->st;
   struct pipe_screen *screen = hgl_dpy->disp->fscreen->screen;

   struct hgl_buffer *buffer = hgl_surf->fb;
   auto &frontBuffer = buffer->textures[ST_ATTACHMENT_FRONT_LEFT];
   auto &backBuffer = buffer->textures[ST_ATTACHMENT_BACK_LEFT];

   // Inform ST of a flush if double buffering is used
   if (backBuffer != NULL)
      st->pipe->flush_resource(st->pipe, backBuffer);

   _mesa_glthread_finish(st->ctx);

   // (take_bitmap() and the present need it)
   hgl_surf->swap_interval = surf->SwapInterval;

   // hardware screens: the GPU copies the frame into a bitmap as part of the
   // frame's own job, before it is flushed
   static const bool separateBlit = getenv("HAIKU_EGL_SEPARATE_BLIT") != NULL;
   struct haiku_present_args args = { st->pipe, backBuffer, hgl_surf,
                                      { NULL, NULL } };
   const bool presentFirst = hgl_dpy->hardware && backBuffer != NULL
      && !separateBlit;

   struct pipe_fence_handle *new_fence = NULL;
   st_context_flush(st, ST_FLUSH_FRONT, &new_fence,
                    presentFirst ? haiku_present_before_flush : NULL,
                    presentFirst ? &args : NULL);
   if (hgl_surf->throttle_fence) {
      const bigtime_t start = system_time();
      screen->fence_finish(screen, NULL, hgl_surf->throttle_fence,
                           OS_TIMEOUT_INFINITE);
      screen->fence_reference(screen, &hgl_surf->throttle_fence, NULL);
      hgl_surf->throttle_wait += system_time() - start;
   }
   hgl_surf->throttle_fence = new_fence;

   // flush back buffer and swap buffers if double buffering is used
   if (backBuffer != NULL) {
      if (hgl_dpy->hardware) {
         if (!presentFirst) {
            args.present = haiku_present_prepare(st->pipe, backBuffer,
                                                 hgl_surf);
            st->pipe->flush(st->pipe, NULL, 0);
         }
         haiku_present_finish(st->pipe, backBuffer, hgl_surf,
                              args.present);
      } else {
         screen->flush_frontbuffer(screen, st->pipe, backBuffer, 0, 0,
                                   buffer->winsysContext, 1, NULL);
      }
      std::swap(frontBuffer, backBuffer);
      p_atomic_inc(&buffer->base.stamp);
   }

   // XXX: right front / back if HGL_STEREO?

   static const bool reportRate = getenv("HAIKU_EGL_FPS") != NULL;
   if (reportRate) {
      const bigtime_t now = system_time();
      if (hgl_surf->fps_start == 0)
         hgl_surf->fps_start = now;
      hgl_surf->fps_frames++;
      if (now - hgl_surf->fps_start >= 5000000) {
         fprintf(stderr, "egl_haiku: %.1f frames/s; per frame %.2f ms waiting "
                 "for the GPU, %.2f ms for the presenter\n",
                 hgl_surf->fps_frames * 1e6 / (now - hgl_surf->fps_start),
                 hgl_surf->throttle_wait / 1e3 / hgl_surf->fps_frames,
                 hgl_surf->presenter_wait / 1e3 / hgl_surf->fps_frames);
         if (hgl_surf->flips + hgl_surf->drops + hgl_surf->superseded > 0) {
            // (the presenter's counts: read without its lock, for a report)
            fprintf(stderr, "egl_haiku: overlay: %u flips, %u missed, %u "
                    "starved, %u late, %u dropped, %u superseded (swap "
                    "interval %d, vblank %.3f ms)\n", hgl_surf->flips,
                    hgl_surf->missed, hgl_surf->starved, hgl_surf->late_flips,
                    hgl_surf->drops, hgl_surf->superseded, surf->SwapInterval,
                    hgl_surf->vblank_period / 1e3);
            hgl_surf->flips = hgl_surf->missed = hgl_surf->starved = 0;
            hgl_surf->late_flips = hgl_surf->drops = 0;
            hgl_surf->superseded = 0;
         }
         hgl_surf->fps_frames = 0;
         hgl_surf->fps_start = now;
         hgl_surf->throttle_wait = 0;
         hgl_surf->presenter_wait = 0;
      }
   }

   update_size(buffer);

   st_context_invalidate_state(st, ST_INVALIDATE_FB_STATE);

   return EGL_TRUE;
}

// #pragma mark EGLDisplay

static EGLBoolean
haiku_add_configs_for_visuals(_EGLDisplay *disp)
{
   CALLED();

   struct haiku_egl_config *conf;
   conf = (struct haiku_egl_config *)calloc(1, sizeof(*conf));
   if (!conf)
      return _eglError(EGL_BAD_ALLOC, "haiku_add_configs_for_visuals");

   _eglInitConfig(&conf->base, disp, 1);
   TRACE("Config inited\n");

   conf->base.RedSize = 8;
   conf->base.BlueSize = 8;
   conf->base.GreenSize = 8;
   conf->base.LuminanceSize = 0;
   conf->base.AlphaSize = 8;
   conf->base.ColorBufferType = EGL_RGB_BUFFER;
   conf->base.BufferSize = conf->base.RedSize + conf->base.GreenSize +
                           conf->base.BlueSize + conf->base.AlphaSize;
   conf->base.ConfigCaveat = EGL_NONE;
   conf->base.ConfigID = 1;
   conf->base.BindToTextureRGB = EGL_FALSE;
   conf->base.BindToTextureRGBA = EGL_FALSE;
   conf->base.StencilSize = 0;
   conf->base.TransparentType = EGL_NONE;
   conf->base.NativeRenderable = EGL_TRUE; // Let's say yes
   conf->base.NativeVisualID = 0;          // No visual
   conf->base.NativeVisualType = EGL_NONE; // No visual
   conf->base.RenderableType = disp->ClientAPIs;
   conf->base.SampleBuffers = 0; // TODO: How to get the right value ?
   conf->base.Samples = conf->base.SampleBuffers == 0 ? 0 : 0;
   conf->base.DepthSize = 24; // TODO: How to get the right value ?
   conf->base.Level = 0;
   conf->base.MaxPbufferWidth = _EGL_MAX_PBUFFER_WIDTH;
   conf->base.MaxPbufferHeight = _EGL_MAX_PBUFFER_HEIGHT;
   conf->base.MaxPbufferPixels = 0; // TODO: How to get the right value ?
   conf->base.SurfaceType = EGL_WINDOW_BIT | EGL_PIXMAP_BIT | EGL_PBUFFER_BIT;
   // overlay frames wait for the vertical blank with 1 (EGL's default)
   conf->base.MinSwapInterval = 0;
   conf->base.MaxSwapInterval = 1;

   TRACE("Config configuated\n");
   if (!_eglValidateConfig(&conf->base, EGL_FALSE)) {
      _eglLog(_EGL_DEBUG, "Haiku: failed to validate config");
      goto cleanup;
   }
   TRACE("Validated config\n");

   _eglLinkConfig(&conf->base);
   if (!_eglGetArraySize(disp->Configs)) {
      _eglLog(_EGL_WARNING, "Haiku: failed to create any config");
      goto cleanup;
   }
   TRACE("Config successful\n");

   return EGL_TRUE;

cleanup:
   free(conf);
   return EGL_FALSE;
}

static void
haiku_display_destroy(_EGLDisplay *disp)
{
   if (!disp)
      return;

   struct haiku_egl_display *hgl_dpy = haiku_egl_display(disp);

   assert(hgl_dpy->ref_count > 0);
   if (!p_atomic_dec_zero(&hgl_dpy->ref_count))
      return;

   struct pipe_screen *screen = hgl_dpy->disp->fscreen->screen;
   hgl_destroy_display(hgl_dpy->disp);
   hgl_dpy->disp = NULL;
   screen->destroy(screen); // destroy will deallocate object

   free(hgl_dpy);
}

static EGLBoolean
haiku_initialize_impl(_EGLDisplay *disp, void *platformDisplay)
{
   struct haiku_egl_display *hgl_dpy;

   hgl_dpy =
      (struct haiku_egl_display *)calloc(1, sizeof(struct haiku_egl_display));
   if (!hgl_dpy)
      return _eglError(EGL_BAD_ALLOC, "eglInitialize");

   hgl_dpy->ref_count = 1;
   disp->DriverData = (void *)hgl_dpy;

   struct pipe_screen *screen = haiku_create_hw_screen();
   hgl_dpy->hardware = screen != NULL;
   if (screen == NULL) {
      struct sw_winsys *winsys = hgl_create_sw_winsys();
      screen = sw_screen_create(winsys);
   }
   if (screen == NULL) {
      free(hgl_dpy);
      disp->DriverData = NULL;
      return _eglError(EGL_NOT_INITIALIZED, "eglInitialize");
   }
   hgl_dpy->disp = hgl_create_display(screen);

   disp->ClientAPIs = 0;
   if (_eglIsApiValid(EGL_OPENGL_API))
      disp->ClientAPIs |= EGL_OPENGL_BIT;
   if (_eglIsApiValid(EGL_OPENGL_ES_API))
      disp->ClientAPIs |=
         EGL_OPENGL_ES_BIT | EGL_OPENGL_ES2_BIT | EGL_OPENGL_ES3_BIT_KHR;

   disp->Extensions.KHR_no_config_context = EGL_TRUE;
   disp->Extensions.KHR_surfaceless_context = EGL_TRUE;
   disp->Extensions.MESA_query_driver = EGL_TRUE;

   /* Report back to EGL the bitmask of priorities supported */
   disp->Extensions.IMG_context_priority =
      hgl_dpy->disp->fscreen->screen->caps.context_priority_mask;
   disp->Extensions.NV_context_priority_realtime =
      disp->Extensions.IMG_context_priority &
      (1 << __EGL_CONTEXT_PRIORITY_REALTIME_BIT);

   disp->Extensions.EXT_pixel_format_float = EGL_TRUE;

   if (hgl_dpy->disp->fscreen->screen->is_format_supported(
          hgl_dpy->disp->fscreen->screen, PIPE_FORMAT_B8G8R8A8_SRGB,
          PIPE_TEXTURE_2D, 0, 0, PIPE_BIND_RENDER_TARGET))
      disp->Extensions.KHR_gl_colorspace = EGL_TRUE;

   disp->Extensions.KHR_create_context = EGL_TRUE;
   disp->Extensions.KHR_reusable_sync = EGL_TRUE;

   haiku_add_configs_for_visuals(disp);

   return EGL_TRUE;
}

static EGLBoolean
haiku_initialize(_EGLDisplay *disp)
{
   EGLBoolean ret = EGL_FALSE;
   struct haiku_egl_display *hgl_dpy = haiku_egl_display(disp);

   if (hgl_dpy) {
      hgl_dpy->ref_count++;
      return EGL_TRUE;
   }

   switch (disp->Platform) {
   case _EGL_PLATFORM_SURFACELESS:
   case _EGL_PLATFORM_HAIKU:
      ret = haiku_initialize_impl(disp, NULL);
      break;
   case _EGL_PLATFORM_DEVICE:
      ret = haiku_initialize_impl(disp, disp->PlatformDisplay);
      break;
   default:
      UNREACHABLE("Callers ensure we cannot get here.");
      return EGL_FALSE;
   }

   if (!ret)
      return EGL_FALSE;

   hgl_dpy = haiku_egl_display(disp);

   return EGL_TRUE;
}

static EGLBoolean
haiku_terminate(_EGLDisplay *disp)
{
   /* A context still current keeps its own reference to the display (see
    * haiku_make_current()), so the display outlives eglTerminate() until it
    * is released, as with the DRI2 driver. */
   _eglReleaseDisplayResources(disp);
   haiku_display_destroy(disp);
   return EGL_TRUE;
}

// #pragma mark EGLContext

/* The rendering API, version and flags eglCreateContext() asked for. */
static void
haiku_context_api(const _EGLContext *ctx, gl_api *api, unsigned *flags)
{
   const EGLint major = ctx->ClientMajorVersion;
   const EGLint minor = ctx->ClientMinorVersion;

   if (ctx->ClientAPI == EGL_OPENGL_ES_API)
      *api = major >= 2 ? API_OPENGLES2 : API_OPENGLES;
   else if ((major >= 4 || (major == 3 && minor >= 2))
      && ctx->Profile == EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR)
      *api = API_OPENGL_CORE;
   else if (major == 3 && minor == 1)
      *api = API_OPENGL_CORE;
   else
      *api = API_OPENGL_COMPAT;

   *flags = 0;
   if (ctx->Flags & EGL_CONTEXT_OPENGL_FORWARD_COMPATIBLE_BIT_KHR)
      *flags |= ST_CONTEXT_FLAG_FORWARD_COMPATIBLE;
   if (ctx->Flags & EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR)
      *flags |= ST_CONTEXT_FLAG_DEBUG;
}

static _EGLContext *
haiku_create_context(_EGLDisplay *disp, _EGLConfig *conf,
                     _EGLContext *share_list, const EGLint *attrib_list)
{
   CALLED();

   struct haiku_egl_display *hgl_dpy = haiku_egl_display(disp);

   struct st_visual visual;
   hgl_get_st_visual(&visual, HGL_DOUBLE | HGL_DEPTH);

   struct haiku_egl_context *context =
      (struct haiku_egl_context *)calloc(1, sizeof(*context));
   if (!context) {
      _eglError(EGL_BAD_ALLOC, "haiku_create_context");
      return NULL;
   }

   if (!_eglInitContext(&context->base, disp, conf, share_list, attrib_list))
      goto cleanup;

   gl_api api;
   unsigned flags;
   haiku_context_api(&context->base, &api, &flags);
   context->ctx = hgl_create_context(
      hgl_dpy->disp, &visual,
      share_list == NULL ? NULL : haiku_egl_context(share_list)->ctx->st,
      api, context->base.ClientMajorVersion, context->base.ClientMinorVersion,
      flags);
   if (context->ctx == NULL)
      goto cleanup;

   return &context->base;

cleanup:
   free(context);
   return NULL;
}

static EGLBoolean
haiku_destroy_context(_EGLDisplay *disp, _EGLContext *ctx)
{
   if (_eglPutContext(ctx)) {
      struct haiku_egl_context *hgl_ctx = haiku_egl_context(ctx);
      hgl_destroy_context(hgl_ctx->ctx);
      free(ctx);
      ctx = NULL;
   }
   return EGL_TRUE;
}

static EGLBoolean
haiku_make_current(_EGLDisplay *disp, _EGLSurface *dsurf, _EGLSurface *rsurf,
                   _EGLContext *ctx)
{
   CALLED();

   struct haiku_egl_context *hgl_ctx = haiku_egl_context(ctx);
   struct haiku_egl_surface *hgl_dsurf = haiku_egl_surface(dsurf);
   struct haiku_egl_surface *hgl_rsurf = haiku_egl_surface(rsurf);
   _EGLContext *old_ctx;
   _EGLSurface *old_dsurf, *old_rsurf;

   if (!_eglBindContext(ctx, dsurf, rsurf, &old_ctx, &old_dsurf, &old_rsurf))
      return EGL_FALSE;

   if (old_ctx == ctx && old_dsurf == dsurf && old_rsurf == rsurf) {
      _eglPutSurface(old_dsurf);
      _eglPutSurface(old_rsurf);
      _eglPutContext(old_ctx);
      return EGL_TRUE;
   }

   _EGLDisplay *old_disp = old_ctx != NULL ? old_ctx->Resource.Display : NULL;

   if (ctx == NULL) {
      st_api_make_current(NULL, NULL, NULL);
   } else {
      if (dsurf != NULL && dsurf != old_dsurf)
         update_size(hgl_dsurf->fb);

      st_api_make_current(hgl_ctx->ctx->st,
                          hgl_dsurf == NULL ? NULL : &hgl_dsurf->fb->base,
                          hgl_rsurf == NULL ? NULL : &hgl_rsurf->fb->base);

      /* A bound context holds a reference to its display: eglTerminate()
       * while it is current must not free what releasing it needs. */
      p_atomic_inc(&haiku_egl_display(disp)->ref_count);
   }

   if (old_dsurf != NULL)
      haiku_destroy_surface(disp, old_dsurf);
   if (old_rsurf != NULL)
      haiku_destroy_surface(disp, old_rsurf);
   if (old_ctx != NULL) {
      haiku_destroy_context(disp, old_ctx);
      haiku_display_destroy(old_disp);
   }

   return EGL_TRUE;
}

extern "C" const _EGLDriver _eglDriver = {
   .Initialize = haiku_initialize,
   .Terminate = haiku_terminate,
   .CreateContext = haiku_create_context,
   .DestroyContext = haiku_destroy_context,
   .MakeCurrent = haiku_make_current,
   .CreateWindowSurface = haiku_create_window_surface,
   .CreatePixmapSurface = haiku_create_pixmap_surface,
   .CreatePbufferSurface = haiku_create_pbuffer_surface,
   .DestroySurface = haiku_destroy_surface,
   .SwapBuffers = haiku_swap_buffers,
};
