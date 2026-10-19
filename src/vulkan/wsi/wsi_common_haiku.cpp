/*
 * Copyright 2026, Haiku, Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 */

/*
 * Haiku window system integration: swapchains on a BView (vk_haiku_surface.h).
 *
 * Swapchain images are ordinary images of the driver. Presenting one copies
 * it on the GPU (the common code's buffer blit, CPU images) into the memory
 * of a BBitmap, and a presenter thread then draws that bitmap into the view.
 * When the driver can import host memory (VK_EXT_external_memory_host), the
 * bitmap's own memory is the copy's destination: an area of ours that the
 * driver imports and app_server shares, so no CPU touches the frame. Without
 * it, the presenter copies the frame from the mapped blit buffer.
 *
 * Synchronization: the present submission signals the image's fence
 * (wsi_swapchain::fences), which the presenter waits for before it reads the
 * bitmap, and an image becomes acquirable again only after app_server has
 * drawn it (BView::Sync()). Acquisition therefore needs no GPU wait and the
 * acquire semaphore and fence are signalled at once.
 *
 * Imported memory is cacheable for the CPU and not coherent with the GPU:
 * vkFlushMappedMemoryRanges once before the GPU first writes it, and
 * vkInvalidateMappedMemoryRanges after each copy, before app_server reads it.
 *
 * Present modes: FIFO draws every presented frame in order; MAILBOX and
 * IMMEDIATE draw only the newest frame queued, the older ones are released
 * undrawn. app_server has no vertical blank synchronization for windows, so
 * FIFO is paced by app_server's drawing, not by the display's refresh.
 *
 * Locking: the presenter locks the view's window (through a BMessenger, which
 * fails safely once the window is gone) for each frame. If the window stays
 * locked by another thread for too long, the frame is dropped rather than
 * waited for, so that an application that holds the lock while it waits for
 * an image cannot deadlock. When the window's own thread presents, the frame
 * is drawn right away in that thread.
 *
 * HAIKU_VK_FPS=1 prints the rate of frames drawn every 5 seconds.
 * HAIKU_VK_NO_IMPORT=1 makes the presenter copy even if the driver could
 * import the bitmaps (to compare the two paths).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <new>

#include <Bitmap.h>
#include <Looper.h>
#include <Messenger.h>
#include <OS.h>
#include <View.h>

#include "util/cnd_monotonic.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/timespec.h"
#include "util/u_atomic.h"
#include "util/u_math.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_util.h"
#include "wsi_common_private.h"
#include "vk_haiku_surface.h"

#define VK_STRUCTURE_TYPE_HAIKU_SURFACE_CREATE_INFO_MESA_cast \
   VkHaikuSurfaceCreateInfoMESA

/* how long the presenter waits for the window's lock before it drops a frame */
static const bigtime_t kWindowLockTimeout = 100000;

struct wsi_haiku_surface {
   VkIcdSurfaceBase base;
   BView *view;
   BMessenger *window; /* the view's handler in its window: locks the window */
   uint64_t size;      /* the view's last known width << 32 | height */
};

struct wsi_haiku {
   struct wsi_interface base;
};

enum wsi_haiku_image_state {
   WSI_HAIKU_IMAGE_IDLE,     /* acquirable */
   WSI_HAIKU_IMAGE_ACQUIRED, /* the application's */
   WSI_HAIKU_IMAGE_QUEUED,   /* presented, waiting for the presenter */
   WSI_HAIKU_IMAGE_DRAWING,  /* the presenter's */
};

struct wsi_haiku_swapchain;

struct wsi_haiku_image {
   struct wsi_image base;
   struct wsi_haiku_swapchain *chain;
   uint32_t index;
   enum wsi_haiku_image_state state;
   BBitmap *bitmap;
   /* The bitmap's memory when the driver imported it (the blit's
    * destination), or -1: the presenter copies into the bitmap. */
   area_id area;
};

struct wsi_haiku_swapchain {
   struct wsi_swapchain base;
   struct wsi_haiku_surface *surface;
   VkExtent2D extent;

   mtx_t lock;
   struct u_cnd_monotonic cond; /* an image state changed, or quit */
   uint32_t *queue;             /* presented image indices, oldest first */
   uint32_t queue_start;
   uint32_t queue_count;
   bool quit;
   VkResult status; /* VK_ERROR_SURFACE_LOST_KHR once the window is gone */
   thread_id presenter;

   bool report_rate;
   uint32_t frames_drawn;
   uint32_t frames_dropped;
   bigtime_t rate_start;

   struct wsi_haiku_image images[0];
};

static inline struct wsi_haiku_surface *
haiku_surface(VkIcdSurfaceBase *surface)
{
   return (struct wsi_haiku_surface *)surface;
}

// #pragma mark - the view

/* Locks the view's window; returns it, or NULL (timeout, or the window is
 * gone). */
static BLooper *
lock_window(struct wsi_haiku_surface *surface, bigtime_t timeout)
{
   if (surface->window->LockTargetWithTimeout(timeout) != B_OK)
      return NULL;
   BLooper *looper = NULL;
   surface->window->Target(&looper);
   return looper;
}

/* The view's size, with the window locked. */
static void
update_size(struct wsi_haiku_surface *surface)
{
   const BRect bounds = surface->view->Bounds();
   const uint64_t width = bounds.IntegerWidth() + 1;
   const uint64_t height = bounds.IntegerHeight() + 1;
   p_atomic_set(&surface->size, width << 32 | height);
}

static VkExtent2D
last_size(const struct wsi_haiku_surface *surface)
{
   const uint64_t size = p_atomic_read(&surface->size);
   const VkExtent2D extent = { (uint32_t)(size >> 32), (uint32_t)size };
   return extent;
}

/* The view's current size, or VK_ERROR_SURFACE_LOST_KHR. If the window stays
 * locked by another thread, the size it had when it was last seen. */
static VkResult
get_view_size(struct wsi_haiku_surface *surface, VkExtent2D *extent)
{
   BLooper *window = lock_window(surface, 500000);
   if (window != NULL) {
      update_size(surface);
      window->Unlock();
   } else if (!surface->window->IsValid()) {
      return VK_ERROR_SURFACE_LOST_KHR;
   }
   *extent = last_size(surface);
   return VK_SUCCESS;
}

/* Draws the bitmap into the view, whose window is locked: scaled to the view
 * while the swapchain has another size than it. Returns after app_server has
 * drawn it, when the bitmap may be written again. */
static void
draw_bitmap(struct wsi_haiku_surface *surface, BBitmap *bitmap)
{
   BView *view = surface->view;
   update_size(surface);

   const BRect bounds = view->Bounds();
   const BRect source = bitmap->Bounds();
   const drawing_mode mode = view->DrawingMode();
   if (mode != B_OP_COPY)
      view->SetDrawingMode(B_OP_COPY);
   if (source.IntegerWidth() == bounds.IntegerWidth()
       && source.IntegerHeight() == bounds.IntegerHeight())
      view->DrawBitmap(bitmap, bounds.LeftTop());
   else
      view->DrawBitmap(bitmap, source, bounds);
   if (mode != B_OP_COPY)
      view->SetDrawingMode(mode);
   view->Sync();
}

// #pragma mark - presenting

/* Waits for the GPU's copy of the image and makes it visible in its bitmap. */
static void
finish_copy(struct wsi_haiku_swapchain *chain, struct wsi_haiku_image *image)
{
   const struct wsi_device *wsi = chain->base.wsi;
   VkDevice device = chain->base.device;

   VkFence fence = chain->base.fences[image->index];
   if (fence != VK_NULL_HANDLE)
      wsi->WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

   const VkMappedMemoryRange range = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = image->base.blit.memory != VK_NULL_HANDLE ?
                image->base.blit.memory : image->base.memory,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
   };
   /* the CPU's cache lines of the memory are stale now */
   wsi->InvalidateMappedMemoryRanges(device, 1, &range);

   if (image->area >= 0)
      return;

   const uint8_t *source = (const uint8_t *)image->base.cpu_map;
   uint8_t *target = (uint8_t *)image->bitmap->Bits();
   const uint32_t source_stride = image->base.row_pitches[0];
   const uint32_t target_stride = image->bitmap->BytesPerRow();
   const uint32_t row = MIN2(chain->extent.width * 4, target_stride);
   for (uint32_t y = 0; y < chain->extent.height; y++)
      memcpy(target + y * target_stride, source + y * source_stride, row);
}

static void
count_frame(struct wsi_haiku_swapchain *chain, bool drawn)
{
   if (!chain->report_rate)
      return;

   mtx_lock(&chain->lock);
   if (drawn)
      chain->frames_drawn++;
   else
      chain->frames_dropped++;

   const bigtime_t now = system_time();
   if (chain->rate_start == 0)
      chain->rate_start = now;
   if (now - chain->rate_start >= 5000000) {
      fprintf(stderr, "wsi_haiku: %.1f frames/s drawn, %u dropped\n",
              chain->frames_drawn * 1e6 / (now - chain->rate_start),
              chain->frames_dropped);
      chain->frames_drawn = 0;
      chain->frames_dropped = 0;
      chain->rate_start = now;
   }
   mtx_unlock(&chain->lock);
}

static void
release_image(struct wsi_haiku_swapchain *chain, struct wsi_haiku_image *image)
{
   mtx_lock(&chain->lock);
   image->state = WSI_HAIKU_IMAGE_IDLE;
   u_cnd_monotonic_broadcast(&chain->cond);
   mtx_unlock(&chain->lock);
}

static status_t
presenter_main(void *data)
{
   struct wsi_haiku_swapchain *chain = (struct wsi_haiku_swapchain *)data;
   struct wsi_haiku_surface *surface = chain->surface;

   mtx_lock(&chain->lock);
   while (true) {
      while (!chain->quit && chain->queue_count == 0)
         u_cnd_monotonic_wait(&chain->cond, &chain->lock);
      if (chain->quit)
         break;

      /* FIFO draws the oldest frame; the other modes the newest, the older
       * ones go undrawn */
      const bool drop = chain->base.present_mode != VK_PRESENT_MODE_FIFO_KHR
         && chain->queue_count > 1;
      struct wsi_haiku_image *image =
         &chain->images[chain->queue[chain->queue_start]];
      chain->queue_start = (chain->queue_start + 1) % chain->base.image_count;
      chain->queue_count--;
      image->state = WSI_HAIKU_IMAGE_DRAWING;
      mtx_unlock(&chain->lock);

      if (drop) {
         /* the GPU may still read the image for its copy */
         VkFence fence = chain->base.fences[image->index];
         if (fence != VK_NULL_HANDLE) {
            chain->base.wsi->WaitForFences(chain->base.device, 1, &fence,
                                           VK_TRUE, UINT64_MAX);
         }
         count_frame(chain, false);
         mtx_lock(&chain->lock);
         image->state = WSI_HAIKU_IMAGE_IDLE;
         u_cnd_monotonic_broadcast(&chain->cond);
         continue;
      }

      finish_copy(chain, image);
      BLooper *window = lock_window(surface, kWindowLockTimeout);
      if (window != NULL) {
         draw_bitmap(surface, image->bitmap);
         window->Unlock();
      } else if (!surface->window->IsValid()) {
         mtx_lock(&chain->lock);
         chain->status = VK_ERROR_SURFACE_LOST_KHR;
         mtx_unlock(&chain->lock);
      }
      count_frame(chain, window != NULL);

      mtx_lock(&chain->lock);
      image->state = WSI_HAIKU_IMAGE_IDLE;
      u_cnd_monotonic_broadcast(&chain->cond);
   }
   mtx_unlock(&chain->lock);
   return B_OK;
}

static bool
extent_is_current(const struct wsi_haiku_swapchain *chain)
{
   const VkExtent2D size = last_size(chain->surface);
   return size.width == chain->extent.width
      && size.height == chain->extent.height;
}

static VkResult
wsi_haiku_swapchain_queue_present(struct wsi_swapchain *wsi_chain,
                                  uint32_t image_index, uint64_t present_id,
                                  const VkPresentRegionKHR *damage)
{
   struct wsi_haiku_swapchain *chain =
      (struct wsi_haiku_swapchain *)wsi_chain;
   struct wsi_haiku_surface *surface = chain->surface;
   struct wsi_haiku_image *image = &chain->images[image_index];

   mtx_lock(&chain->lock);
   VkResult status = chain->status;
   mtx_unlock(&chain->lock);
   if (status != VK_SUCCESS) {
      release_image(chain, image);
      return status;
   }

   /* The window's own thread presents: draw the frame here, the presenter
    * could not lock the window before this thread returns to it. */
   BLooper *window = lock_window(surface, 0);
   if (window != NULL && window->Thread() != find_thread(NULL)) {
      window->Unlock();
      window = NULL;
   }
   if (window != NULL) {
      finish_copy(chain, image);
      draw_bitmap(surface, image->bitmap);
      window->Unlock();
      count_frame(chain, true);
      release_image(chain, image);
   } else {
      mtx_lock(&chain->lock);
      const uint32_t end =
         (chain->queue_start + chain->queue_count) % chain->base.image_count;
      chain->queue[end] = image_index;
      chain->queue_count++;
      image->state = WSI_HAIKU_IMAGE_QUEUED;
      u_cnd_monotonic_broadcast(&chain->cond);
      mtx_unlock(&chain->lock);
   }

   return extent_is_current(chain) ? VK_SUCCESS : VK_SUBOPTIMAL_KHR;
}

static VkResult
wsi_haiku_swapchain_acquire_next_image(struct wsi_swapchain *wsi_chain,
                                       const VkAcquireNextImageInfoKHR *info,
                                       uint32_t *image_index)
{
   struct wsi_haiku_swapchain *chain =
      (struct wsi_haiku_swapchain *)wsi_chain;

   const uint64_t deadline = os_time_get_absolute_timeout(info->timeout);
   struct timespec deadline_spec;
   timespec_from_nsec(&deadline_spec, deadline);

   mtx_lock(&chain->lock);
   while (true) {
      if (chain->status != VK_SUCCESS) {
         mtx_unlock(&chain->lock);
         return chain->status;
      }

      /* the idle image presented longest ago */
      struct wsi_haiku_image *found = NULL;
      for (uint32_t i = 0; i < chain->base.image_count; i++) {
         struct wsi_haiku_image *image = &chain->images[i];
         if (image->state == WSI_HAIKU_IMAGE_IDLE
             && (found == NULL
                 || image->base.present_serial < found->base.present_serial))
            found = image;
      }
      if (found != NULL) {
         found->state = WSI_HAIKU_IMAGE_ACQUIRED;
         *image_index = found->index;
         mtx_unlock(&chain->lock);
         return extent_is_current(chain) ? VK_SUCCESS : VK_SUBOPTIMAL_KHR;
      }

      if (info->timeout == 0) {
         mtx_unlock(&chain->lock);
         return VK_NOT_READY;
      }
      if (deadline == OS_TIMEOUT_INFINITE) {
         u_cnd_monotonic_wait(&chain->cond, &chain->lock);
      } else if (u_cnd_monotonic_timedwait(&chain->cond, &chain->lock,
                                           &deadline_spec) == thrd_timedout) {
         mtx_unlock(&chain->lock);
         return VK_TIMEOUT;
      }
   }
}

static VkResult
wsi_haiku_swapchain_release_images(struct wsi_swapchain *wsi_chain,
                                   uint32_t count, const uint32_t *indices)
{
   struct wsi_haiku_swapchain *chain =
      (struct wsi_haiku_swapchain *)wsi_chain;

   mtx_lock(&chain->lock);
   for (uint32_t i = 0; i < count; i++) {
      assert(indices[i] < chain->base.image_count);
      chain->images[indices[i]].state = WSI_HAIKU_IMAGE_IDLE;
   }
   u_cnd_monotonic_broadcast(&chain->cond);
   mtx_unlock(&chain->lock);
   return VK_SUCCESS;
}

static VkResult
wsi_haiku_swapchain_wait_for_present(struct wsi_swapchain *wsi_chain,
                                     uint64_t present_id, uint64_t timeout)
{
   return wsi_swapchain_wait_for_present_semaphore(wsi_chain, present_id,
                                                   timeout);
}

static struct wsi_image *
wsi_haiku_swapchain_get_wsi_image(struct wsi_swapchain *wsi_chain,
                                  uint32_t image_index)
{
   struct wsi_haiku_swapchain *chain =
      (struct wsi_haiku_swapchain *)wsi_chain;
   return &chain->images[image_index].base;
}

// #pragma mark - swapchains

/* The memory of the image's buffer blit destination: a BBitmap on an area of
 * our own (whole pages, which the driver imports), or NULL. */
static uint8_t *
alloc_bitmap_memory(struct wsi_image *wsi_image, unsigned size)
{
   struct wsi_haiku_image *image = (struct wsi_haiku_image *)wsi_image;
   struct wsi_haiku_swapchain *chain = image->chain;

   void *address;
   const area_id area = create_area("vulkan swapchain image", &address,
      B_ANY_ADDRESS, align(size, B_PAGE_SIZE), B_NO_LOCK,
      B_READ_AREA | B_WRITE_AREA | B_CLONEABLE_AREA);
   if (area < 0)
      return NULL;

   /* B_RGB32 is B, G, R, X in memory: VK_FORMAT_B8G8R8A8_* */
   const BRect bounds(0, 0, chain->extent.width - 1, chain->extent.height - 1);
   BBitmap *bitmap = new(std::nothrow) BBitmap(area, 0, bounds, 0, B_RGB32,
      chain->base.image_info.linear_stride);
   if (bitmap == NULL || bitmap->InitCheck() != B_OK) {
      delete bitmap;
      delete_area(area);
      return NULL;
   }

   image->bitmap = bitmap;
   image->area = area;
   return (uint8_t *)address;
}

/* Imported bitmap memory need not be coherent (finish_copy() invalidates it):
 * prefer the memory type the driver caches. */
static uint32_t
select_bitmap_memory_type(const struct wsi_device *wsi, uint32_t type_bits)
{
   return wsi_select_memory_type(wsi,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      0, type_bits);
}

static void
destroy_bitmap(struct wsi_haiku_image *image)
{
   delete image->bitmap;
   image->bitmap = NULL;
   if (image->area >= 0)
      delete_area(image->area);
   image->area = -1;
}

static VkResult
wsi_haiku_swapchain_destroy(struct wsi_swapchain *wsi_chain,
                            const VkAllocationCallbacks *pAllocator)
{
   struct wsi_haiku_swapchain *chain =
      (struct wsi_haiku_swapchain *)wsi_chain;
   const struct wsi_device *wsi = chain->base.wsi;

   if (chain->presenter >= 0) {
      mtx_lock(&chain->lock);
      chain->quit = true;
      u_cnd_monotonic_broadcast(&chain->cond);
      mtx_unlock(&chain->lock);
      status_t status;
      wait_for_thread(chain->presenter, &status);
   }

   /* the GPU may still copy the last frames */
   for (uint32_t i = 0; i < chain->base.image_count; i++) {
      if (chain->base.fences != NULL && chain->base.fences[i] != VK_NULL_HANDLE) {
         wsi->WaitForFences(chain->base.device, 1, &chain->base.fences[i],
                            VK_TRUE, 1000000000ull);
      }
   }

   for (uint32_t i = 0; i < chain->base.image_count; i++) {
      struct wsi_haiku_image *image = &chain->images[i];
      if (image->base.image != VK_NULL_HANDLE)
         wsi_destroy_image(&chain->base, &image->base);
      destroy_bitmap(image);
   }

   vk_free(pAllocator, chain->queue);
   u_cnd_monotonic_destroy(&chain->cond);
   mtx_destroy(&chain->lock);
   wsi_swapchain_finish(&chain->base);
   vk_free(pAllocator, chain);
   return VK_SUCCESS;
}

static VkResult
create_swapchain(struct wsi_haiku_surface *surface, VkDevice device,
                 struct wsi_device *wsi_device,
                 const VkSwapchainCreateInfoKHR *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, bool import,
                 struct wsi_swapchain **swapchain_out)
{
   const uint32_t image_count = MAX2(pCreateInfo->minImageCount, 2);
   const size_t size = sizeof(struct wsi_haiku_swapchain)
      + image_count * sizeof(struct wsi_haiku_image);
   struct wsi_haiku_swapchain *chain = (struct wsi_haiku_swapchain *)
      vk_zalloc(pAllocator, size, 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (chain == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   struct wsi_cpu_image_params params = {
      .base = { .image_type = WSI_IMAGE_TYPE_CPU },
   };
   /* the bitmaps' own memory is the blit's destination if it can be */
   if (import && wsi_cpu_image_needs_buffer_blit(wsi_device, &params))
      params.alloc_shm = alloc_bitmap_memory;

   VkResult result = wsi_swapchain_init(wsi_device, &chain->base, device,
                                        pCreateInfo, &params.base, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(pAllocator, chain);
      return result;
   }
   if (params.alloc_shm != NULL) {
      chain->base.image_info.select_blit_dst_memory_type =
         select_bitmap_memory_type;
   }
   /* the images are no dma-bufs: no implicit synchronization through them */
   if (chain->base.dma_buf_semaphore != VK_NULL_HANDLE) {
      wsi_device->DestroySemaphore(device, chain->base.dma_buf_semaphore,
                                   &chain->base.alloc);
      chain->base.dma_buf_semaphore = VK_NULL_HANDLE;
   }

   chain->base.destroy = wsi_haiku_swapchain_destroy;
   chain->base.get_wsi_image = wsi_haiku_swapchain_get_wsi_image;
   chain->base.acquire_next_image = wsi_haiku_swapchain_acquire_next_image;
   chain->base.release_images = wsi_haiku_swapchain_release_images;
   chain->base.queue_present = wsi_haiku_swapchain_queue_present;
   chain->base.wait_for_present = wsi_haiku_swapchain_wait_for_present;
   chain->base.wait_for_present2 = wsi_haiku_swapchain_wait_for_present;
   chain->base.present_mode =
      wsi_swapchain_get_present_mode(wsi_device, pCreateInfo);
   chain->base.image_count = image_count;
   chain->surface = surface;
   chain->extent = pCreateInfo->imageExtent;
   chain->presenter = -1;
   chain->report_rate = getenv("HAIKU_VK_FPS") != NULL;
   mtx_init(&chain->lock, mtx_plain);
   u_cnd_monotonic_init(&chain->cond);

   chain->queue = (uint32_t *)vk_alloc(pAllocator,
      image_count * sizeof(uint32_t), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (chain->queue == NULL) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail;
   }

   for (uint32_t i = 0; i < image_count; i++) {
      struct wsi_haiku_image *image = &chain->images[i];
      image->chain = chain;
      image->index = i;
      image->area = -1;
      image->state = WSI_HAIKU_IMAGE_IDLE;
   }

   for (uint32_t i = 0; i < image_count; i++) {
      struct wsi_haiku_image *image = &chain->images[i];
      result = wsi_create_image(&chain->base, &chain->base.image_info,
                                &image->base);
      if (result != VK_SUCCESS) {
         /* wsi_create_image() destroyed what it made */
         memset(&image->base, 0, sizeof(image->base));
         goto fail;
      }

      if (image->area >= 0) {
         /* nothing the CPU cached of the new memory may be written back
          * over the GPU's frames later */
         const VkMappedMemoryRange range = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = image->base.blit.memory,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
         };
         wsi_device->FlushMappedMemoryRanges(device, 1, &range);
      } else {
         /* the driver could not import a bitmap: the presenter copies */
         destroy_bitmap(image);
         const BRect bounds(0, 0, chain->extent.width - 1,
                            chain->extent.height - 1);
         image->bitmap = new(std::nothrow) BBitmap(bounds, B_RGB32);
         if (image->bitmap == NULL || image->bitmap->InitCheck() != B_OK) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto fail;
         }
         if (image->base.cpu_map == NULL) {
            result = VK_ERROR_MEMORY_MAP_FAILED;
            goto fail;
         }
      }
   }

   chain->presenter = spawn_thread(presenter_main, "vulkan presenter",
                                   B_DISPLAY_PRIORITY, chain);
   if (chain->presenter < 0 || resume_thread(chain->presenter) != B_OK) {
      if (chain->presenter >= 0)
         kill_thread(chain->presenter);
      chain->presenter = -1;
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail;
   }

   *swapchain_out = &chain->base;
   return VK_SUCCESS;

fail:
   wsi_haiku_swapchain_destroy(&chain->base, pAllocator);
   return result;
}

static VkResult
wsi_haiku_surface_create_swapchain(VkIcdSurfaceBase *icd_surface,
                                   VkDevice device,
                                   struct wsi_device *wsi_device,
                                   const VkSwapchainCreateInfoKHR *pCreateInfo,
                                   const VkAllocationCallbacks *pAllocator,
                                   struct wsi_swapchain **swapchain_out)
{
   struct wsi_haiku_surface *surface = haiku_surface(icd_surface);
   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR);

   static const bool no_import = getenv("HAIKU_VK_NO_IMPORT") != NULL;
   if (wsi_device->has_import_memory_host && !no_import) {
      VkResult result = create_swapchain(surface, device, wsi_device,
                                         pCreateInfo, pAllocator, true,
                                         swapchain_out);
      if (result == VK_SUCCESS)
         return result;
      mesa_logw("wsi_haiku: the driver cannot import the window's bitmaps "
                "(%d): presenting by copying", result);
   }
   return create_swapchain(surface, device, wsi_device, pCreateInfo,
                           pAllocator, false, swapchain_out);
}

// #pragma mark - surfaces

static VkResult
wsi_haiku_surface_get_support(VkIcdSurfaceBase *surface,
                              struct wsi_device *wsi_device,
                              uint32_t queueFamilyIndex,
                              VkBool32 *pSupported)
{
   *pSupported = VK_TRUE;
   return VK_SUCCESS;
}

static const VkPresentModeKHR present_modes[] = {
   VK_PRESENT_MODE_FIFO_KHR,
   VK_PRESENT_MODE_MAILBOX_KHR,
   VK_PRESENT_MODE_IMMEDIATE_KHR,
};

static VkResult
wsi_haiku_surface_get_capabilities(VkIcdSurfaceBase *icd_surface,
                                   struct wsi_device *wsi_device,
                                   VkSurfaceCapabilitiesKHR *caps)
{
   VkExtent2D extent;
   VkResult result = get_view_size(haiku_surface(icd_surface), &extent);
   if (result != VK_SUCCESS)
      return result;

   /* one image on screen (being drawn), one to render into */
   caps->minImageCount = 2;
   caps->maxImageCount = 0;
   caps->currentExtent = extent;
   caps->minImageExtent.width = 1;
   caps->minImageExtent.height = 1;
   caps->maxImageExtent.width = wsi_device->maxImageDimension2D;
   caps->maxImageExtent.height = wsi_device->maxImageDimension2D;
   caps->supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
   caps->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
   caps->maxImageArrayLayers = 1;
   caps->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
   caps->supportedUsageFlags = wsi_caps_get_image_usage();

   VK_FROM_HANDLE(vk_physical_device, pdevice, wsi_device->pdevice);
   if (pdevice->supported_extensions.EXT_attachment_feedback_loop_layout)
      caps->supportedUsageFlags |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;

   return VK_SUCCESS;
}

static VkResult
wsi_haiku_surface_get_capabilities2(VkIcdSurfaceBase *surface,
                                    struct wsi_device *wsi_device,
                                    const void *info_next,
                                    VkSurfaceCapabilities2KHR *caps)
{
   assert(caps->sType == VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR);

   const VkSurfacePresentModeKHR *present_mode =
      vk_find_struct_const(info_next, SURFACE_PRESENT_MODE_EXT);

   VkResult result = wsi_haiku_surface_get_capabilities(surface, wsi_device,
      &caps->surfaceCapabilities);

   vk_foreach_struct(ext, caps->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_SURFACE_PROTECTED_CAPABILITIES_KHR: {
         VkSurfaceProtectedCapabilitiesKHR *prot =
            (VkSurfaceProtectedCapabilitiesKHR *)ext;
         prot->supportsProtected = VK_FALSE;
         break;
      }
      case VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_KHR: {
         /* (a swapchain of another size than the view is scaled to it, but
          * not as one of the scaling modes of the extension) */
         VkSurfacePresentScalingCapabilitiesKHR *scaling =
            (VkSurfacePresentScalingCapabilitiesKHR *)ext;
         scaling->supportedPresentScaling = 0;
         scaling->supportedPresentGravityX = 0;
         scaling->supportedPresentGravityY = 0;
         scaling->minScaledImageExtent =
            caps->surfaceCapabilities.minImageExtent;
         scaling->maxScaledImageExtent =
            caps->surfaceCapabilities.maxImageExtent;
         break;
      }
      case VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_COMPATIBILITY_KHR: {
         VkSurfacePresentModeCompatibilityKHR *compat =
            (VkSurfacePresentModeCompatibilityKHR *)ext;
         if (compat->pPresentModes == NULL) {
            compat->presentModeCount = 1;
         } else if (compat->presentModeCount > 0 && present_mode != NULL) {
            compat->presentModeCount = 1;
            compat->pPresentModes[0] = present_mode->presentMode;
         }
         break;
      }
      case VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT: {
         VkPresentTimingSurfaceCapabilitiesEXT *timing =
            (VkPresentTimingSurfaceCapabilitiesEXT *)ext;
         timing->presentStageQueries = 0;
         timing->presentTimingSupported = VK_FALSE;
         timing->presentAtAbsoluteTimeSupported = VK_FALSE;
         timing->presentAtRelativeTimeSupported = VK_FALSE;
         break;
      }
      default:
         break;
      }
   }

   return result;
}

static const VkFormat formats[] = {
   VK_FORMAT_B8G8R8A8_UNORM,
   VK_FORMAT_B8G8R8A8_SRGB,
};

static VkResult
wsi_haiku_surface_get_formats(VkIcdSurfaceBase *surface,
                              struct wsi_device *wsi_device,
                              uint32_t *pSurfaceFormatCount,
                              VkSurfaceFormatKHR *pSurfaceFormats)
{
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormatKHR, out, pSurfaceFormats,
                          pSurfaceFormatCount);
   for (unsigned i = 0; i < ARRAY_SIZE(formats); i++) {
      vk_outarray_append_typed(VkSurfaceFormatKHR, &out, format) {
         format->format = formats[i];
         format->colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      }
   }
   return vk_outarray_status(&out);
}

static VkResult
wsi_haiku_surface_get_formats2(VkIcdSurfaceBase *surface,
                               struct wsi_device *wsi_device,
                               const void *info_next,
                               uint32_t *pSurfaceFormatCount,
                               VkSurfaceFormat2KHR *pSurfaceFormats)
{
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormat2KHR, out, pSurfaceFormats,
                          pSurfaceFormatCount);
   for (unsigned i = 0; i < ARRAY_SIZE(formats); i++) {
      vk_outarray_append_typed(VkSurfaceFormat2KHR, &out, format) {
         format->surfaceFormat.format = formats[i];
         format->surfaceFormat.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      }
   }
   return vk_outarray_status(&out);
}

static VkResult
wsi_haiku_surface_get_present_modes(VkIcdSurfaceBase *surface,
                                    struct wsi_device *wsi_device,
                                    uint32_t *pPresentModeCount,
                                    VkPresentModeKHR *pPresentModes)
{
   if (pPresentModes == NULL) {
      *pPresentModeCount = ARRAY_SIZE(present_modes);
      return VK_SUCCESS;
   }

   *pPresentModeCount = MIN2(*pPresentModeCount, ARRAY_SIZE(present_modes));
   typed_memcpy(pPresentModes, present_modes, *pPresentModeCount);
   return *pPresentModeCount < ARRAY_SIZE(present_modes) ?
      VK_INCOMPLETE : VK_SUCCESS;
}

static VkResult
wsi_haiku_surface_get_present_rectangles(VkIcdSurfaceBase *icd_surface,
                                         struct wsi_device *wsi_device,
                                         uint32_t *pRectCount,
                                         VkRect2D *pRects)
{
   VkExtent2D extent;
   VkResult result = get_view_size(haiku_surface(icd_surface), &extent);
   if (result != VK_SUCCESS)
      return result;

   VK_OUTARRAY_MAKE_TYPED(VkRect2D, out, pRects, pRectCount);
   vk_outarray_append_typed(VkRect2D, &out, rect) {
      rect->offset.x = 0;
      rect->offset.y = 0;
      rect->extent = extent;
   }
   return vk_outarray_status(&out);
}

bool
wsi_haiku_is_view_surface(const VkHeadlessSurfaceCreateInfoEXT *info)
{
   /* (the loader passes no create info at all when it could not copy it) */
   return info != NULL
      && vk_find_struct_const(info->pNext, HAIKU_SURFACE_CREATE_INFO_MESA)
         != NULL;
}

VkResult
wsi_haiku_create_surface(VkInstance _instance,
                         const VkHeadlessSurfaceCreateInfoEXT *info,
                         const VkAllocationCallbacks *pAllocator,
                         VkSurfaceKHR *pSurface)
{
   VK_FROM_HANDLE(vk_instance, instance, _instance);
   const VkHaikuSurfaceCreateInfoMESA *haiku_info =
      vk_find_struct_const(info->pNext, HAIKU_SURFACE_CREATE_INFO_MESA);

   BView *view = (BView *)haiku_info->view;
   if (view == NULL || view->Looper() == NULL) {
      mesa_loge("wsi_haiku: the surface's view is not attached to a window");
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   struct wsi_haiku_surface *surface = (struct wsi_haiku_surface *)
      vk_zalloc2(&instance->alloc, pAllocator, sizeof(*surface), 8,
                 VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (surface == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   surface->base.platform = VK_ICD_WSI_PLATFORM_HAIKU_MESA;
   surface->view = view;
   surface->window = new(std::nothrow) BMessenger(view);
   if (surface->window == NULL) {
      vk_free2(&instance->alloc, pAllocator, surface);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   VkExtent2D extent;
   get_view_size(surface, &extent);

   *pSurface = VkIcdSurfaceBase_to_handle(&surface->base);
   return VK_SUCCESS;
}

void
wsi_haiku_surface_destroy(VkIcdSurfaceBase *icd_surface, VkInstance _instance,
                          const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vk_instance, instance, _instance);
   struct wsi_haiku_surface *surface = haiku_surface(icd_surface);
   delete surface->window;
   vk_free2(&instance->alloc, pAllocator, surface);
}

VkResult
wsi_haiku_init_wsi(struct wsi_device *wsi_device,
                   const VkAllocationCallbacks *alloc,
                   VkPhysicalDevice physical_device)
{
   struct wsi_haiku *wsi = (struct wsi_haiku *)vk_zalloc(alloc, sizeof(*wsi),
      8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (wsi == NULL) {
      wsi_device->wsi[VK_ICD_WSI_PLATFORM_HAIKU_MESA] = NULL;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   wsi->base.get_support = wsi_haiku_surface_get_support;
   wsi->base.get_capabilities2 = wsi_haiku_surface_get_capabilities2;
   wsi->base.get_formats = wsi_haiku_surface_get_formats;
   wsi->base.get_formats2 = wsi_haiku_surface_get_formats2;
   wsi->base.get_present_modes = wsi_haiku_surface_get_present_modes;
   wsi->base.get_present_rectangles = wsi_haiku_surface_get_present_rectangles;
   wsi->base.create_swapchain = wsi_haiku_surface_create_swapchain;

   wsi_device->wsi[VK_ICD_WSI_PLATFORM_HAIKU_MESA] = &wsi->base;
   return VK_SUCCESS;
}

void
wsi_haiku_finish_wsi(struct wsi_device *wsi_device,
                     const VkAllocationCallbacks *alloc)
{
   vk_free(alloc, wsi_device->wsi[VK_ICD_WSI_PLATFORM_HAIKU_MESA]);
   wsi_device->wsi[VK_ICD_WSI_PLATFORM_HAIKU_MESA] = NULL;
}
