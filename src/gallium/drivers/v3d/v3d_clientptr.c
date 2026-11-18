/*
 * Copyright 2026, Haiku, Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 */

/*
 * GL_MESA_map_buffer_client_pointer: a buffer's BO mapped into memory the
 * client provides (Wine's WoW64 thunks give memory below 4 GiB, so that 32-bit
 * code can use the pointer directly instead of a copy).
 *
 * Every v3d buffer has a BO of its own, and the kernel can map a BO any number
 * of times, so a placed BO simply has its one CPU mapping (bo->map) in the
 * client's range: internal maps use the same pointer and nothing is staged.
 * The placement stays for the BO's life; a client-pointer map of a placed BO
 * costs nothing. A range goes back to the heap's free list, as plain memory,
 * when its BO dies (before the BO enters the BO cache) or when the client
 * asks for all of its ranges (glReleaseClientPointerRangeMESA with
 * GL_CLIENT_POINTER_RELEASE_ALL_MESA) and no client map is open on the BO.
 */

#include <sys/mman.h>

#include "util/u_memory.h"

#include "v3d_context.h"
#include "v3d_resource.h"

struct v3d_clientptr_range {
        struct pipe_clientptr_range base;
        struct list_head link;
};

static inline struct v3d_clientptr_heap *
v3d_clientptr_heap(struct pipe_clientptr_heap *heap)
{
        return (struct v3d_clientptr_heap *)heap;
}

static struct pipe_clientptr_heap *
v3d_clientptr_heap_create(struct pipe_screen *pscreen)
{
        struct v3d_clientptr_heap *heap = CALLOC_STRUCT(v3d_clientptr_heap);
        if (!heap)
                return NULL;

        pipe_reference_init(&heap->base.reference, 1);
        mtx_init(&heap->lock, mtx_plain);
        list_inithead(&heap->placed_bos);
        list_inithead(&heap->free_ranges);
        return &heap->base;
}

static void
v3d_clientptr_heap_destroy(struct pipe_screen *pscreen,
                           struct pipe_clientptr_heap *pheap)
{
        struct v3d_clientptr_heap *heap = v3d_clientptr_heap(pheap);

        /* Placed BOs hold a reference. */
        assert(list_is_empty(&heap->placed_bos));

        /* Ranges the client never took back stay as they are: plain memory
         * the client still counts as its own. Unmapping them would leave
         * holes in its address space bookkeeping.
         */
        list_for_each_entry_safe(struct v3d_clientptr_range, range,
                                 &heap->free_ranges, link) {
                mesa_logw("v3d: client pointer range %p (%u bytes) never "
                          "released", range->base.start, range->base.size);
                list_del(&range->link);
                FREE(range);
        }

        mtx_destroy(&heap->lock);
        FREE(heap);
}

/**
 * Unplaces \p bo and takes it off its heap's list; called with the heap
 * locked. The caller drops the BO's reference on the heap. Returns the range
 * in \p range, or false if no memory could be put back over it: then the BO's
 * pages are unmapped from it all the same (a hole in the client's memory is
 * the lesser evil) and the range must not go back to the client.
 */
static bool
unplace_bo_locked(struct v3d_bo *bo, struct pipe_clientptr_range *range)
{
        range->start = bo->map;
        range->size = bo->size;

        list_del(&bo->clientptr_link);
        bo->clientptr_heap = NULL;

        if (!v3d_bo_unmap_placed(bo)) {
                munmap(bo->map, bo->size);
                bo->map = NULL;
                return false;
        }
        return true;
}

static bool
v3d_clientptr_range_reclaim(struct pipe_screen *pscreen,
                            struct pipe_clientptr_heap *pheap,
                            bool release_all,
                            struct pipe_clientptr_range *range)
{
        struct v3d_clientptr_heap *heap = v3d_clientptr_heap(pheap);
        bool found = false;

        mtx_lock(&heap->lock);
        if (!list_is_empty(&heap->free_ranges)) {
                struct v3d_clientptr_range *free_range =
                        list_first_entry(&heap->free_ranges,
                                         struct v3d_clientptr_range, link);
                list_del(&free_range->link);
                *range = free_range->base;
                FREE(free_range);
                mtx_unlock(&heap->lock);
                return true;
        }

        unsigned unplaced = 0;
        if (release_all) {
                /* A BO with a client map open keeps its range. */
                list_for_each_entry_safe(struct v3d_bo, bo, &heap->placed_bos,
                                         clientptr_link) {
                        if (bo->clientptr_maps != 0)
                                continue;
                        unplaced++;
                        if (unplace_bo_locked(bo, range)) {
                                found = true;
                                break;
                        }
                }
        }
        mtx_unlock(&heap->lock);

        /* The unplaced BOs' references on the heap (the caller has one too,
         * so this never destroys it).
         */
        while (unplaced-- > 0) {
                struct pipe_clientptr_heap *ref = pheap;
                pipe_clientptr_heap_reference(pscreen, &ref, NULL);
        }
        return found;
}

void
v3d_clientptr_release_bo(struct v3d_bo *bo)
{
        struct v3d_clientptr_heap *heap = bo->clientptr_heap;
        struct pipe_clientptr_heap *pheap = &heap->base;
        struct pipe_clientptr_range range;

        assert(bo->clientptr_maps == 0);

        struct v3d_clientptr_range *free_range =
                CALLOC_STRUCT(v3d_clientptr_range);

        mtx_lock(&heap->lock);
        if (unplace_bo_locked(bo, &range) && free_range) {
                free_range->base = range;
                list_addtail(&free_range->link, &heap->free_ranges);
                free_range = NULL;
        }
        mtx_unlock(&heap->lock);

        FREE(free_range);
        pipe_clientptr_heap_reference(&bo->screen->base, &pheap, NULL);
}

static void *
v3d_transfer_map_clientptr_range(struct pipe_context *pctx,
                                 struct pipe_transfer *ptrans,
                                 struct pipe_clientptr_range *range,
                                 uint32_t *size)
{
        struct v3d_transfer *trans = v3d_transfer(ptrans);
        struct v3d_resource *rsc = v3d_resource(ptrans->resource);
        struct v3d_clientptr_heap *heap =
                v3d_clientptr_heap(rsc->base.clientptr_heap);
        struct v3d_bo *bo = rsc->bo;

        assert(ptrans->usage & PIPE_MAP_CLIENT_POINTER);
        assert(rsc->base.target == PIPE_BUFFER && heap);

        if (!trans->clientptr_bo) {
                if (!bo->clientptr_heap) {
                        /* The frontend passes a range only of the size we
                         * asked for: the BO's (its padding included).
                         */
                        if (!range || range->size != bo->size) {
                                if (size)
                                        *size = bo->size;
                                return NULL;
                        }

                        if (!v3d_bo_map_placed(bo, range->start)) {
                                if (size)
                                        *size = 0;
                                return NULL;
                        }
                        range->start = NULL;
                        range->size = 0;

                        pipe_reference(NULL, &heap->base.reference);
                        mtx_lock(&heap->lock);
                        bo->clientptr_heap = heap;
                        list_addtail(&bo->clientptr_link, &heap->placed_bos);
                        mtx_unlock(&heap->lock);
                }
                assert(bo->clientptr_heap == heap);

                trans->clientptr_bo = v3d_bo_reference(bo);
                bo->clientptr_maps++;
        }

        return (uint8_t *)trans->clientptr_bo->map + ptrans->box.x;
}

static void
v3d_transfer_unmap_clientptr_range(struct pipe_context *pctx,
                                   struct pipe_transfer *ptrans)
{
        struct v3d_transfer *trans = v3d_transfer(ptrans);

        assert(trans->clientptr_bo && trans->clientptr_bo->clientptr_maps);
        trans->clientptr_bo->clientptr_maps--;
        v3d_bo_unreference(&trans->clientptr_bo);
}

void
v3d_clientptr_screen_init(struct pipe_screen *pscreen)
{
        struct v3d_screen *screen = v3d_screen(pscreen);

        if (!screen->has_placed_map)
                return;

        pscreen->clientptr_heap_create = v3d_clientptr_heap_create;
        pscreen->clientptr_heap_destroy = v3d_clientptr_heap_destroy;
        pscreen->clientptr_range_reclaim = v3d_clientptr_range_reclaim;
}

void
v3d_clientptr_context_init(struct pipe_context *pctx)
{
        pctx->transfer_map_clientptr_range = v3d_transfer_map_clientptr_range;
        pctx->transfer_unmap_clientptr_range =
                v3d_transfer_unmap_clientptr_range;
}
