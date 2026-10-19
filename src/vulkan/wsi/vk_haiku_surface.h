/*
 * Copyright 2026, Haiku, Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 */
#ifndef VK_HAIKU_SURFACE_H
#define VK_HAIKU_SURFACE_H

/*
 * Vulkan surfaces on Haiku views (Mesa's Haiku window system integration).
 *
 * Vulkan has no registered Haiku surface extension, so Mesa's drivers take a
 * BView through VK_EXT_headless_surface: chain a VkHaikuSurfaceCreateInfoMESA
 * to VkHeadlessSurfaceCreateInfoEXT::pNext and the surface is that view. Its
 * size is the surface's current extent, and each presented image is drawn
 * into it. Without the struct the surface is an ordinary headless one.
 *
 *    instance extensions: VK_KHR_surface, VK_EXT_headless_surface
 *
 *    VkHaikuSurfaceCreateInfoMESA haiku = {
 *       .sType = VK_STRUCTURE_TYPE_HAIKU_SURFACE_CREATE_INFO_MESA,
 *       .view = view,
 *    };
 *    VkHeadlessSurfaceCreateInfoEXT info = {
 *       .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
 *       .pNext = &haiku,
 *    };
 *    vkCreateHeadlessSurfaceEXT(instance, &info, NULL, &surface);
 *
 * The view must be attached to a window when the surface is created, and it
 * must stay there until the surface is destroyed. The application should not
 * draw into it itself (a view color of B_TRANSPARENT_COLOR keeps app_server
 * from erasing the frames). Images are presented by a thread of the driver,
 * which locks the window for each frame; when the window's own thread
 * presents, the image is drawn right away in that thread instead.
 *
 * The Vulkan loader copies the create info of a surface and drops structs it
 * does not know: it must know this one (the Haiku build of the Khronos loader
 * has a patch for it).
 */

#include <vulkan/vulkan_core.h>

#ifdef __cplusplus
extern "C" {
#endif

/* In the range of structure types of VK_EXT_headless_surface (extension 257),
 * which does not use this value. */
#define VK_STRUCTURE_TYPE_HAIKU_SURFACE_CREATE_INFO_MESA \
   ((VkStructureType)1000256900)

typedef struct VkHaikuSurfaceCreateInfoMESA {
   VkStructureType sType;
   const void *pNext;
   VkFlags flags; /* reserved: 0 */
   void *view;    /* the BView to present into */
} VkHaikuSurfaceCreateInfoMESA;

#ifdef __cplusplus
}
#endif

#endif /* VK_HAIKU_SURFACE_H */
