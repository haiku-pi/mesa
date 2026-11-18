/*
 * Copyright 2026, Haiku, Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 */

/*
 * Depth clamping (GL_ARB_depth_clamp) for V3D 4.x, which can neither turn
 * off clipping against the near and far planes nor clamp the depth: V3D 7.1
 * has both in the CFG_BITS packet.
 *
 * The last geometry stage passes its clip-space Z to the fragment shader in
 * a varying of its own (V3D_DEPTH_CLAMP_SLOT) and gives the fixed-function
 * hardware a Z of 0, which lies between the near and far planes of every
 * vertex in front of the eye, so that nothing is clipped against them
 * (v3d_nir_lower_io.c). The fragment shader computes the window-space depth
 * from that varying the way the hardware would have, clamps it to the depth
 * range and writes it (this file). The clip-space Z is interpolated with
 * perspective correction and divided by the fragment's interpolated W, which
 * gives the depth the hardware interpolates linearly in screen space, also
 * where the clipper cut the primitive. Reads of gl_FragCoord.z see the
 * unclamped depth, as the hardware's Z would be.
 *
 * The polygon offset, which the hardware applies to its own Z only, is added
 * the same way: factor × the depth's largest screen-space slope + units × the
 * depth buffer's resolution (for a float depth buffer, of the depth's
 * exponent), then limited by the clamp.
 *
 * Writing the depth from the shader turns off early Z tests while depth
 * clamping is on. A fragment shader that already uses every input slot, or
 * forces early fragment tests, is left alone; its last geometry stage then
 * sees no use of the varying and keeps its Z (the primitive is clipped, not
 * clamped).
 */

#include "compiler/v3d_compiler.h"
#include "compiler/nir/nir_builder.h"

static unsigned
next_driver_location(nir_shader *s, nir_variable_mode mode)
{
        unsigned next = 0;
        nir_foreach_variable_with_modes(var, s, mode) {
                unsigned slots = glsl_count_attribute_slots(var->type, false);
                next = MAX2(next, var->data.driver_location + slots);
        }
        return next;
}

static bool
is_depth_store(nir_intrinsic_instr *intr)
{
        return intr->intrinsic == nir_intrinsic_store_output &&
               nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_DEPTH;
}

static bool
is_frag_coord_load(nir_intrinsic_instr *intr)
{
        return (intr->intrinsic == nir_intrinsic_load_input ||
                intr->intrinsic == nir_intrinsic_load_interpolated_input) &&
               nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_POS;
}

static nir_def *
clamp_to_depth_range(nir_builder *b, nir_def *depth)
{
        /* Window Z is clip-space Z / W * scale + offset: the depth range is
         * offset - |scale| .. offset + |scale| (V3D 4.x has no
         * GL_ARB_clip_control, so Z is always -1..1 in clip space).
         */
        nir_def *offset = nir_load_viewport_z_offset(b);
        nir_def *half_range = nir_fabs(b, nir_load_viewport_z_scale(b));
        return nir_fmin(b, nir_fmax(b, depth, nir_fsub(b, offset, half_range)),
                        nir_fadd(b, offset, half_range));
}

static nir_def *
polygon_offset(nir_builder *b, nir_def *depth, bool float_depth)
{
        nir_def *params = nir_load_depth_offset_v3d(b, 32);
        nir_def *units = nir_channel(b, params, 0);
        nir_def *factor = nir_channel(b, params, 1);
        nir_def *limit = nir_channel(b, params, 2);

        if (float_depth) {
                /* the resolution at the depth's exponent: 2^(e - 23) */
                nir_def *e = nir_ffloor(b, nir_flog2(b, nir_fabs(b, depth)));
                units = nir_fmul(b, units,
                                 nir_fexp2(b, nir_fadd_imm(b, e, -23.0f)));
        }
        nir_def *slope = nir_fmax(b, nir_fabs(b, nir_ddx(b, depth)),
                                  nir_fabs(b, nir_ddy(b, depth)));
        nir_def *offset = nir_ffma(b, factor, slope, units);
        offset = nir_bcsel(b, nir_fgt_imm(b, limit, 0.0f),
                           nir_fmin(b, offset, limit),
                           nir_bcsel(b, nir_flt_imm(b, limit, 0.0f),
                                     nir_fmax(b, offset, limit), offset));
        return nir_fadd(b, depth, offset);
}

bool
v3d_nir_lower_depth_clamp_fs(nir_shader *s, struct v3d_compile *c)
{
        assert(s->info.stage == MESA_SHADER_FRAGMENT);

        if (s->info.fs.early_fragment_tests)
                return false;

        unsigned input_location = next_driver_location(s, nir_var_shader_in);
        if ((input_location + 1) * 4 > V3D_MAX_FS_INPUTS)
                return false;

        nir_variable *clip_z = nir_variable_create(s, nir_var_shader_in,
                                                   glsl_float_type(),
                                                   "v3d_clip_z");
        clip_z->data.location = V3D_DEPTH_CLAMP_SLOT;
        clip_z->data.driver_location = input_location;
        clip_z->data.interpolation = INTERP_MODE_SMOOTH;
        clip_z->data.how_declared = nir_var_hidden;
        s->info.inputs_read |= BITFIELD64_BIT(V3D_DEPTH_CLAMP_SLOT);

        nir_variable *depth_var =
                nir_find_variable_with_location(s, nir_var_shader_out,
                                                FRAG_RESULT_DEPTH);

        nir_function_impl *impl = nir_shader_get_entrypoint(s);
        nir_builder b = nir_builder_at(nir_before_impl(impl));

        /* The depth the hardware would have interpolated */
        nir_def *z = nir_load_input(&b, 1, 32, nir_imm_int(&b, 0),
                                    .base = input_location,
                                    .component = 0,
                                    .dest_type = nir_type_float32,
                                    .io_semantics.location = V3D_DEPTH_CLAMP_SLOT,
                                    .io_semantics.num_slots = 1);
        z = nir_fdiv(&b, z, nir_load_fep_w_v3d(&b, 32));
        nir_def *depth = nir_ffma(&b, z, nir_load_viewport_z_scale(&b),
                                  nir_load_viewport_z_offset(&b));
        if (c->fs_key->depth_offset)
                depth = polygon_offset(&b, depth, c->fs_key->depth_offset_float);

        nir_foreach_block(block, impl) {
                nir_foreach_instr_safe(instr, block) {
                        if (instr->type != nir_instr_type_intrinsic)
                                continue;
                        nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

                        if (is_frag_coord_load(intr)) {
                                /* gl_FragCoord.z: the computed depth */
                                unsigned first = nir_intrinsic_component(intr);
                                unsigned count = intr->def.num_components;
                                if (first > 2 || first + count <= 2)
                                        continue;
                                b.cursor = nir_after_instr(instr);
                                nir_def *value =
                                        nir_vector_insert_imm(&b, &intr->def,
                                                              depth, 2 - first);
                                nir_def_rewrite_uses_after(&intr->def, value);
                        } else if (is_depth_store(intr)) {
                                /* gl_FragDepth: clamped as the hardware's
                                 * would be
                                 */
                                b.cursor = nir_before_instr(instr);
                                nir_def *value = nir_channel(&b, intr->src[0].ssa, 0);
                                nir_src_rewrite(&intr->src[0],
                                                clamp_to_depth_range(&b, value));
                        }
                }
        }

        if (depth_var == NULL) {
                depth_var = nir_variable_create(s, nir_var_shader_out,
                                                glsl_float_type(),
                                                "v3d_clamped_depth");
                depth_var->data.location = FRAG_RESULT_DEPTH;
                depth_var->data.driver_location =
                        next_driver_location(s, nir_var_shader_out);
                depth_var->data.how_declared = nir_var_hidden;
                s->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);

                b.cursor = nir_after_impl(impl);
                nir_store_output(&b, clamp_to_depth_range(&b, depth),
                                 nir_imm_int(&b, 0),
                                 .base = depth_var->data.driver_location,
                                 .component = 0,
                                 .src_type = nir_type_float32,
                                 .io_semantics.location = FRAG_RESULT_DEPTH,
                                 .io_semantics.num_slots = 1);
        }

        return nir_progress(true, impl, nir_metadata_control_flow);
}
