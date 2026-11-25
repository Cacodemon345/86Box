/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Native parameter-draw spans for the Vérité pixel engine on AArch64, after
   pixel_jit_x64.c.

   A repeated parameter draw (rdrawp) runs one pixel_full() per pixel with a
   render state that stays fixed for the whole span. This compiles that state
   into a native loop and caches it. The key holds everything the code depends
   on; values that may change between spans (bases, scissors, colours, masks)
   are read from the engine at run time. States outside the supported subset
   return to the C path. */

#include "internal.h"

#ifdef VERITE_JIT_ARM64

#    include <stdlib.h>
#    include "a64.h"

enum {
    SPAN_CACHE      = 64,
    SPAN_ARENA      = 1024 * 1024,
    SPAN_MAX_CODE   = 16 * 1024,
    PREFETCH_SHIFT  = 2, /* look 1 << PREFETCH_SHIFT pixels ahead */
    DST_RAW_ZWRITE  = 1 << 16,
    DST_RAW_YUV     = 1 << 18,
    DST_RAW_BLEND   = 1 << 19,
    DST_RAW_DITHER  = 1 << 20,
    DST_RAW_FOG     = 1 << 21,
    DST_RAW_BGR     = 1 << 24,
    DST_RAW_NO_PAD  = 1 << 22,
    DST_RAW_READ_DISABLE = 1 << 23,
    DST_RAW_TREJECT = 1 << 25,
};

/* The span's stack frame at SP. The values and deltas step four at a time
   with Q loads, which need offsets that are multiples of 16. */
typedef struct {
    uint32_t          values[12], deltas[12];
    int32_t           c[4], tap[4], sum[4];
    uint16_t          fg[4]; /* argb(FgColor) as halfwords R, G, B, A */
    verite_core      *core;
    verite_draw_site *site;
    const uint16_t   *dither_row; /* the dither vectors for a fixed Y */
    int32_t           cnt;
    uint32_t          x_step, dst_stride, z_stride, tex_stride;
    uint32_t          z_addr, z;
    uint32_t          prev_u, prev_v; /* the last pixel's texel coordinates */
    int32_t           scissor[4];     /* x lo, x hi, y lo, y hi */
    uint32_t          dst_row, z_row; /* row addresses when Y is no parameter */
    uint32_t          dither_x, dither_y, thres;
    uint32_t          prefetch; /* whether the texture outgrows the L1 cache */
} span_ctx;
typedef char span_ctx_aligned[offsetof(span_ctx, deltas) % 16 == 0 && offsetof(span_ctx, c) % 16 == 0
                                      && offsetof(span_ctx, sum) % 16 == 0 && offsetof(span_ctx, fg) % 8 == 0
                                  ? 1
                                  : -1];

/* Per-state constants, placed before the span's code and loaded into V16-V22
   before its loop. Vectors hold halfwords R, G, B, A. */
typedef struct {
    uint16_t k127[4], k8081[4], k255[4];
    int16_t  down[4]; /* x >> (8 - n) as a USHL count, n the field width */
    int16_t  up[4];   /* the field's position as a USHL count */
    uint16_t rgb[4], alpha[4];
    uint16_t dither[16][4]; /* (dither_table[i] << 4) >> n */
} span_consts;

/* unsigned fn(verite_core *core, verite_draw_site *site): draws with the
   site's value and delta registers and returns whether a pixel was written. */
typedef unsigned (*span_fn)(verite_core *core, verite_draw_site *site);

typedef struct {
    uint32_t op, drawctl, src_mode, dst_mode, dst_fmt;
} span_key;

typedef struct {
    span_key key;
    span_fn  fn;
    bool     used;
} span_entry;

struct verite_pixel_jit {
    uint8_t   *arena;
    size_t     used;
    unsigned   next;
    span_entry entries[SPAN_CACHE];
};

/* Parameter layouts; these mirror pixel.c. */
typedef struct {
    pixel_param even, odd;
} jit_pair;
static const struct {
    jit_pair pair;
    bool     shadow;
} jit_pair1[] = {
    { { P_X, P_NONE }, false }, { { P_Q, P_NONE }, true }, { { P_X, P_Y }, false }, { { P_Q, P_Y }, true },
    { { P_X, P_U }, false }, { { P_Q, P_U }, true }, { { P_X, P_R }, false }, { { P_Q, P_R }, true },
    { { P_X, P_A }, false }, { { P_Q, P_A }, true }, { { P_Z, P_X }, false }, { { P_NONE, P_NONE }, true },
    { { P_X, P_F }, false }, { { P_Q, P_F }, true }, { { P_Y, P_U }, true }, { { P_Q, P_U }, true },
};
static const jit_pair jit_pair2[] = {
    { P_X, P_V }, { P_Y, P_V }, { P_R, P_V }, { P_NONE, P_V }, { P_I, P_V }, { P_Z, P_V }, { P_A, P_V },
    { P_Z, P_U }, { P_G, P_B }, { P_Z, P_A }, { P_Z, P_NONE }, { P_NONE, P_F }, { P_Z, P_F }, { P_F, P_V },
    { P_R, P_F }, { P_R, P_A }, { P_R, P_NONE }, { P_Z, P_R }, { P_NONE, P_A }, { P_F, P_A }, { P_I, P_NONE },
    { P_Z, P_I }, { P_I, P_A }, { P_I, P_F },
};
static const jit_pair jit_pair3[] = {
    { P_G, P_B }, { P_Z, P_A }, { P_Z, P_NONE }, { P_NONE, P_F }, { P_Z, P_F }, { P_NONE, P_NONE },
    { P_R, P_F }, { P_R, P_A }, { P_R, P_NONE }, { P_Z, P_R }, { P_NONE, P_A }, { P_F, P_A },
    { P_I, P_NONE }, { P_Z, P_I }, { P_I, P_A }, { P_I, P_F },
};

static uint8_t widen_table[3][64]; /* 4, 5 and 6 bits, padded */
static const uint32_t stride_high[] = { 0, 16, 32, 64, 128, 1024, 2048, 4096 };
static const uint32_t stride_low[]  = { 0, 256, 512, 1024, 4, 768, 576, 1152 };
static const uint8_t dither_table[16] = { 0, 12, 3, 15, 7, 11, 4, 8, 13, 1, 14, 2, 10, 6, 9, 5 };

static void
init_tables(void)
{
    static const unsigned bits[] = { 4, 5, 6 };
    for (unsigned t = 0; t < 3; t++) {
        unsigned max = (1U << bits[t]) - 1;
        for (unsigned v = 0; v <= max; v++)
            widen_table[t][v] = (v * 255 + max / 2) / max;
    }
}

/* Every 16-bit texel of a 565 (fmt 4) or 1555 (fmt 6) texture, padded or not,
   decoded as texel() does into R | G << 8 | B << 16 | A << 24. Built on first
   use. */
static uint32_t *texel_tables[2][2];

static const uint32_t *
texel_table(unsigned fmt, bool no_pad)
{
    uint32_t **table = &texel_tables[fmt == 6][no_pad];
    if (*table)
        return *table;
    uint32_t *t = malloc(65536 * sizeof(uint32_t));
    if (!t)
        abort();
    static const unsigned shift565[] = { 11, 5, 0 }, bits565[] = { 5, 6, 5 }, shift1555[] = { 10, 5, 0 };
    for (unsigned texel = 0; texel < 65536; texel++) {
        uint32_t ch[4];
        for (unsigned k = 0; k < 3; k++) {
            unsigned shift = fmt == 4 ? shift565[k] : shift1555[k];
            unsigned bits  = fmt == 4 ? bits565[k] : 5;
            unsigned v     = (texel >> shift) & ((1U << bits) - 1);
            ch[k]          = no_pad ? v << (8 - bits) : widen_table[bits - 4][v];
        }
        ch[3]    = fmt == 4 || (texel & 0x8000) ? 255 : 0;
        t[texel] = ch[0] | ch[1] << 8 | ch[2] << 16 | ch[3] << 24;
    }
    return *table = t;
}

/* ---- Registers ------------------------------------------------------------ */

/* W0-W2 and W5-W6 are scratch as RAX, RCX, RDX, R8 and R9 are in the x64
   spans; W3 and W4 hold x and y, W7 and W8 u and v. X9 is the VRAM base, W10
   the VRAM mask with bit 0 clear, X11 the engine, W12 the pixel count, X13
   the texel table, W14 the drew flag and X15 the constants. W19 and W20, the
   only callee-saved registers, keep the lowest and highest address written.
   V16-V22 hold the constants and V24-V27 the bilinear weights. */
enum { AX = 0, CX = 1, DX = 2, XR = 3, YR = 4, R8 = 5, R9 = 6, UR = 7, VR = 8,
       VRAM = 9, MASK = 10, PE = 11, COUNT = 12, TABLE = 13, DREW = 14, KC = 15,
       LO = 19, HI = 20, CTX = A64_SP };
enum { K127 = 16, K8081, K255, KDOWN, KUP, KRGB, KALPHA, WU0 = 24, WU1, WV0, WV1 };

typedef a64 as;

#    define CTXF(f)     ((int32_t) offsetof(span_ctx, f))
#    define PEF(f)      ((int32_t) offsetof(pixel_engine, f))
#    define KF(f)       ((int32_t) offsetof(span_consts, f))
#    define LD(r, b, o) a64_ldst(a, A64_LDR_W, r, b, o)
#    define ST(r, b, o) a64_ldst(a, A64_STR_W, r, b, o)
#    define MOV(d, s)   a64_mov(a, false, d, s)

/* dst = 16-bit VRAM word at the address in addr (clobbers addr). */
static void
emit_load16(as *a, unsigned dst, unsigned addr)
{
    a64_rrr(a, A64_AND, false, addr, addr, MASK);
    a64_alu_imm(a, A64_EOR, false, addr, addr, 2);
    a64_ldst_reg(a, A64_LDRH, dst, VRAM, addr, A64_UXTW, false);
}

/* Store the low 16 bits of val at the address in addr (clobbers addr),
   widening the written range in W19 and W20. */
static void
emit_store16(as *a, unsigned val, unsigned addr)
{
    a64_rrr(a, A64_AND, false, addr, addr, MASK);
    a64_cmp(a, false, addr, LO);
    a64_csel(a, false, LO, addr, LO, A64_LO);
    a64_cmp(a, false, addr, HI);
    a64_csel(a, false, HI, addr, HI, A64_HI);
    a64_alu_imm(a, A64_EOR, false, addr, addr, 2);
    a64_ldst_reg(a, A64_STRH, val, VRAM, addr, A64_UXTW, false);
}

/* reg = clamp(reg >> 16, 0, 255). Clobbers W16. */
static void
emit_sat_reg(as *a, unsigned reg)
{
    a64_shift_imm(a, A64_ASR, false, reg, reg, 16);
    a64_alu(a, A64_BIC, false, reg, reg, reg, A64_ASR, 31);
    a64_mov_imm(a, false, A64_IP0, 255);
    a64_cmp(a, false, reg, A64_IP0);
    a64_csel(a, false, reg, A64_IP0, reg, A64_GT);
}

static void
emit_sat(as *a)
{
    emit_sat_reg(a, AX);
}

/* fg = (fg & ~mask) | (W0 & mask). Clobbers W1. */
static void
emit_fg_merge(as *a, uint32_t mask)
{
    a64_alu_imm(a, A64_AND, false, AX, AX, mask);
    LD(CX, PE, PEF(fg));
    a64_alu_imm(a, A64_AND, false, CX, CX, ~mask);
    a64_rrr(a, A64_ORR, false, AX, AX, CX);
    ST(AX, PE, PEF(fg));
}

/* pixel_set() for one parameter, the value in W0. */
static void
emit_set(as *a, pixel_param param, uint32_t src_mode)
{
    static const int8_t shift[] = { [P_R] = 16, [P_G] = 8, [P_B] = 0, [P_A] = 24 };
    switch (param) {
        case P_X:
            ST(AX, PE, PEF(x));
            break;
        case P_Y:
            ST(AX, PE, PEF(y));
            break;
        case P_F:
            ST(AX, PE, PEF(f));
            break;
        case P_Z:
            ST(AX, PE, PEF(z));
            if (src_mode & SRC_Z_SCISSOR) {
                a64_tst_imm(a, false, AX, 0xf8000000);
                a64_cset(a, false, CX, A64_NE);
                a64_ldst(a, A64_STRB, CX, PE, PEF(z_ovf));
            } else
                a64_ldst(a, A64_STRB, A64_ZR, PE, PEF(z_ovf));
            break;
        case P_U:
        case P_V:
            {
                bool u = (param == P_U) != !!(src_mode & SRC_SWAP_UV);
                /* emit_uv_mask() sets the mask once, before the loop. */
                ST(AX, PE, u ? PEF(u) : PEF(v));
                break;
            }
        case P_R:
        case P_G:
        case P_B:
        case P_A:
            ST(AX, PE, param == P_R ? PEF(r) : param == P_G ? PEF(g) : param == P_B ? PEF(b) : PEF(a));
            emit_sat(a);
            a64_shift_imm(a, A64_LSL, false, AX, AX, shift[param]);
            emit_fg_merge(a, 0xffU << shift[param]);
            break;
        case P_I:
            ST(AX, PE, PEF(r));
            ST(AX, PE, PEF(g));
            ST(AX, PE, PEF(b));
            emit_sat(a);
            a64_mov_imm(a, false, CX, 0x010101);
            a64_mul(a, false, AX, AX, CX);
            emit_fg_merge(a, 0xffffff);
            break;
        default:
            break;
    }
}

/* The mask pixel_set() takes from SrcMask with U or V. SrcMask is fixed for
   the span, so this runs once before its first pixel. Clobbers W1. */
static void
emit_uv_mask(as *a, pixel_param param, uint32_t src_mode)
{
    bool u = (param == P_U) != !!(src_mode & SRC_SWAP_UV);
    LD(CX, PE, PEF(src_mask));
    a64_ubfx(a, false, CX, CX, u ? 0 : 16, 11);
    ST(CX, PE, u ? PEF(u_mask) : PEF(v_mask));
}

/* reg = texture_coord(reg, mask from the engine). Clobbers W16. */
static void
emit_coord_reg(as *a, unsigned reg, int32_t mask_field, bool clamped)
{
    LD(A64_IP0, PE, mask_field);
    if (clamped) {
        a64_cmp(a, false, reg, A64_IP0);
        a64_csel(a, false, reg, A64_IP0, reg, A64_GT);
        a64_alu(a, A64_BIC, false, reg, reg, reg, A64_ASR, 31); /* below 0 to 0 */
    } else
        a64_rrr(a, A64_AND, false, reg, reg, A64_IP0);
}

/* W1 = widened channel of W0: (W0 >> shift) & ((1 << bits) - 1), expanded to 8 bits. */
static void
emit_widen(as *a, unsigned shift, unsigned bits, bool no_pad)
{
    a64_ubfx(a, false, CX, AX, shift, bits);
    if (no_pad)
        a64_shift_imm(a, A64_LSL, false, CX, CX, 8 - bits);
    else {
        a64_mov_imm(a, true, A64_IP0, (uint64_t) (uintptr_t) widen_table[bits - 4]);
        a64_ldst_reg(a, A64_LDRB, CX, A64_IP0, CX, A64_UXTW, false);
    }
}

/* Vv = the decoded texel at row + 2 * x (both registers) as bytes R, G, B, A
   or, keyed, black or YUV black (128, 16, 128, 0). Clobbers W6, W16, W17. */
static void
emit_tap(as *a, uint32_t src_mode, unsigned v, unsigned row, unsigned x)
{
    a64_alu(a, A64_ADD, false, R9, row, x, A64_LSL, 1);
    emit_load16(a, R9, R9);
    a64_ldst_reg(a, A64_LDR_S, v, TABLE, R9, A64_UXTW, true);
    if (src_mode & SRC_CHROMA) {
        a64_mov_imm(a, false, A64_IP1, 0x10001);
        a64_mul(a, false, A64_IP0, R9, A64_IP1);
        LD(A64_IP1, PE, PEF(chroma_color));
        a64_rrr(a, A64_EOR, false, A64_IP0, A64_IP0, A64_IP1);
        LD(A64_IP1, PE, PEF(chroma_mask));
        a64_rrr(a, A64_ANDS, false, A64_IP0, A64_IP0, A64_IP1);
        uint8_t *opaque = a64_bcond(a, A64_NE);
        a64_mov_imm(a, false, A64_IP0, src_mode & SRC_CHROMA_BLACK_YUV ? 128 | 16 << 8 | 128 << 16 : 0);
        a64_fmov_from(a, v, A64_IP0);
        a64_land(opaque, a->p);
    }
}

/* Prefetch the texel 1 << PREFETCH_SHIFT pixels on, extrapolated from how far u
   and v moved since the last pixel. Perspective keeps that step close to
   constant over a few pixels, and a wrong guess only costs the hint.
   W7 = u, W8 = v. Clobbers W0, W1, W16. */
static void
emit_prefetch(as *a)
{
    for (unsigned axis = 0; axis < 2; axis++) {
        unsigned reg  = axis ? CX : AX, coord = axis ? VR : UR;
        int32_t  prev = axis ? CTXF(prev_v) : CTXF(prev_u);
        LD(A64_IP0, CTX, prev);
        a64_rrr(a, A64_SUB, false, reg, coord, A64_IP0);
        a64_alu(a, A64_ADD, false, reg, coord, reg, A64_LSL, PREFETCH_SHIFT);
        ST(coord, CTX, prev);
        a64_shift_imm(a, A64_ASR, false, reg, reg, 16);
        LD(A64_IP0, PE, axis ? PEF(v_mask) : PEF(u_mask));
        a64_rrr(a, A64_AND, false, reg, reg, A64_IP0);
    }
    LD(A64_IP0, CTX, CTXF(tex_stride));
    a64_mul(a, false, CX, CX, A64_IP0);
    LD(A64_IP0, PE, PEF(src_base));
    a64_rrr(a, A64_ADD, false, CX, CX, A64_IP0);
    a64_alu(a, A64_ADD, false, CX, CX, AX, A64_LSL, 1);
    a64_rrr(a, A64_AND, false, CX, CX, MASK);
    a64_prefetch(a, VRAM, CX);
}

/* sample() into V3 as halfwords R, G, B, A. W7 = u, W8 = v after any
   perspective division. Clobbers W0-W2, W5, W6, W16, W17, V0, V1, V4, V5 and
   the weights.

   The bilinear weights (256 - fu, fu) x (256 - fv, fv) separate. Each row
   sums its two taps exactly in 16 bits, h = (256 - fu) t0 + fu t1 <= 65280,
   and the rows then sum in 32 bits; RSHRN gives (sum + 32768) >> 16 as in
   sample(). For an unpadded 565 texel this also matches
   sample_bilinear_565(), as the x64 spans note. */
static void
emit_sample(as *a, uint32_t src_mode)
{
    bool uc = !!(src_mode & SRC_U_CLAMP), vc = !!(src_mode & SRC_V_CLAMP);
    bool filter = !!(src_mode & SRC_FILTER);
    LD(A64_IP0, CTX, CTXF(prefetch));
    uint8_t *cached = a64_cbz(a, false, false, A64_IP0);
    emit_prefetch(a);
    a64_land(cached, a->p);

    /* W0 = x0, W1 = x1, W2 = row0, W5 = row1 */
    a64_shift_imm(a, A64_ASR, false, AX, UR, 16);
    a64_shift_imm(a, A64_ASR, false, DX, VR, 16);
    if (filter) {
        a64_alu_imm(a, A64_ADD, false, CX, AX, 1);
        a64_alu_imm(a, A64_ADD, false, R8, DX, 1);
        emit_coord_reg(a, CX, PEF(u_mask), uc);
        emit_coord_reg(a, R8, PEF(v_mask), vc);
        LD(A64_IP0, CTX, CTXF(tex_stride));
        LD(A64_IP1, PE, PEF(src_base));
        a64_madd(a, false, R8, R8, A64_IP0, A64_IP1);
    }
    emit_coord_reg(a, AX, PEF(u_mask), uc);
    emit_coord_reg(a, DX, PEF(v_mask), vc);
    LD(A64_IP0, CTX, CTXF(tex_stride));
    LD(A64_IP1, PE, PEF(src_base));
    a64_madd(a, false, DX, DX, A64_IP0, A64_IP1);

    if (!filter) {
        emit_tap(a, src_mode, 3, DX, AX);
        a64_uxtl(a, false, 3, 3);
        return;
    }

    /* The weights 256 - f and f, from bits 8-15 of u and v. */
    for (unsigned axis = 0; axis < 2; axis++) {
        a64_ubfx(a, false, A64_IP0, axis ? VR : UR, 8, 8);
        a64_dup(a, false, axis ? WV1 : WU1, A64_IP0);
        a64_mov_imm(a, false, A64_IP1, 256);
        a64_rrr(a, A64_SUB, false, A64_IP0, A64_IP1, A64_IP0);
        a64_dup(a, false, axis ? WV0 : WU0, A64_IP0);
    }

    emit_tap(a, src_mode, 0, DX, AX);
    emit_tap(a, src_mode, 1, DX, CX);
    emit_tap(a, src_mode, 4, R8, AX);
    emit_tap(a, src_mode, 5, R8, CX);
    a64_uxtl(a, false, 0, 0);
    a64_uxtl(a, false, 1, 1);
    a64_uxtl(a, false, 4, 4);
    a64_uxtl(a, false, 5, 5);
    A64_VMUL(a, V4H, 0, 0, WU0);
    A64_VMLA(a, V4H, 0, 1, WU1); /* h0 */
    A64_VMUL(a, V4H, 4, 4, WU0);
    A64_VMLA(a, V4H, 4, 5, WU1); /* h1 */
    a64_umull(a, 0, 0, WV0);
    a64_umlal(a, 0, 4, WV1);
    a64_shrn(a, true, 3, 0, 16);
}

/* W7, W8 = u, v * 2^24 / q unless Q is 0, with the exact truncating division
   of perspective_uv(). Clobbers W5, X16. */
static void
emit_divide_uv(as *a)
{
    LD(R8, PE, PEF(q));
    uint8_t *flat = a64_cbz(a, false, false, R8);
    for (unsigned axis = 0; axis < 2; axis++) {
        unsigned coord = axis ? VR : UR;
        a64_sxtw(a, A64_IP0, coord);
        a64_shift_imm(a, A64_LSL, true, A64_IP0, A64_IP0, 24);
        a64_sdiv(a, true, A64_IP0, A64_IP0, R8); /* q zero-extended by the load */
        MOV(coord, A64_IP0);
    }
    a64_land(flat, a->p);
}

/* W0 = (W0 + 127) / 255 for W0 + 127 < 65536. */
static void
emit_div255(as *a)
{
    a64_alu_imm(a, A64_ADD, false, AX, AX, 127);
    a64_mov_imm(a, false, A64_IP0, 0x8081);
    a64_mul(a, false, AX, AX, A64_IP0);
    a64_shift_imm(a, A64_LSR, false, AX, AX, 23);
}

/* W1 = 255 - [SP + field] */
static void
emit_inverse(as *a, int32_t field)
{
    LD(A64_IP0, CTX, field);
    a64_mov_imm(a, false, CX, 255);
    a64_rrr(a, A64_SUB, false, CX, CX, A64_IP0);
}

/* W1 = factor(func, source, k, s = ctx.c, d = ctx.tap), as in pixel.c. */
static void
emit_factor(as *a, unsigned func, bool source, unsigned k)
{
    int32_t s = CTXF(c[0]) + 4 * (int32_t) k, d = CTXF(tap[0]) + 4 * (int32_t) k;
    switch (func) {
        case 0:
            LD(CX, CTX, source ? d : s);
            break;
        case 1:
            emit_inverse(a, source ? d : s);
            break;
        case 2:
            LD(CX, CTX, CTXF(c[3]));
            break;
        case 3:
            emit_inverse(a, CTXF(c[3]));
            break;
        case 4:
            LD(CX, CTX, CTXF(tap[3]));
            break;
        case 5:
            emit_inverse(a, CTXF(tap[3]));
            break;
        case 6:
            a64_mov_imm(a, false, CX, 0);
            break;
        case 8:
        case 9:
            if (k == 3)
                a64_mov_imm(a, false, CX, 255);
            else {
                /* min(s.a, 255 - d.a); both are bytes. */
                emit_inverse(a, CTXF(tap[3]));
                LD(DX, CTX, CTXF(c[3]));
                a64_cmp(a, false, DX, CX);
                a64_csel(a, false, CX, DX, CX, A64_LT);
            }
            if (func == 9) {
                a64_mov_imm(a, false, A64_IP0, 255);
                a64_rrr(a, A64_SUB, false, CX, A64_IP0, CX);
            }
            break;
        case 10:
            LD(CX, CTX, source ? s : d);
            break;
        case 11:
            emit_inverse(a, source ? s : d);
            break;
        default:
            a64_mov_imm(a, false, CX, 255);
            break;
    }
}

/* W0 = one side of blend(): the channel scaled by its factor, or less 128. */
static void
emit_blend_term(as *a, unsigned func, bool source, unsigned k)
{
    int32_t own = source ? CTXF(c[0]) + 4 * (int32_t) k : CTXF(tap[0]) + 4 * (int32_t) k;
    if (func == 15) {
        LD(AX, CTX, own);
        a64_alu_imm(a, A64_SUB, false, AX, AX, 128);
        return;
    }
    emit_factor(a, func, source, k);
    LD(AX, CTX, own);
    a64_mul(a, false, AX, AX, CX);
    emit_div255(a);
}

/* c = blend(c, d), d from DstColor or the decoded destination pixel. W3 = x,
   W4 = y. Clobbers W0-W2, W6. */
static void
emit_blend(as *a, const span_key *k)
{
    if (k->dst_mode & DST_RAW_READ_DISABLE) {
        LD(AX, PE, PEF(dst_color));
        for (unsigned ch = 0; ch < 4; ch++) {
            static const unsigned sh[] = { 16, 8, 0, 24 };
            a64_ubfx(a, false, CX, AX, sh[ch], 8);
            ST(CX, CTX, CTXF(tap[0]) + 4 * (int32_t) ch);
        }
    } else {
        bool no_pad = !!(k->dst_mode & DST_RAW_NO_PAD);
        LD(A64_IP0, CTX, CTXF(dst_stride));
        LD(A64_IP1, PE, PEF(dst_base));
        a64_madd(a, false, AX, YR, A64_IP0, A64_IP1);
        a64_alu(a, A64_ADD, false, AX, AX, XR, A64_LSL, 1);
        emit_load16(a, AX, AX);
        static const unsigned shift565[] = { 11, 5, 0 }, bits565[] = { 5, 6, 5 }, shift1555[] = { 10, 5, 0 };
        for (unsigned ch = 0; ch < 3; ch++) {
            if (k->dst_fmt == 4)
                emit_widen(a, shift565[ch], bits565[ch], no_pad);
            else
                emit_widen(a, shift1555[ch], 5, no_pad);
            ST(CX, CTX, CTXF(tap[0]) + 4 * (int32_t) ch);
        }
        if (k->dst_fmt == 4)
            a64_mov_imm(a, false, CX, 255);
        else {
            /* 255 with the alpha bit, else 0 */
            a64_bfm(a, A64_SBFM, false, CX, AX, 15, 15);
            a64_alu_imm(a, A64_AND, false, CX, CX, 255);
        }
        ST(CX, CTX, CTXF(tap[3]));
    }
    unsigned sf = (k->dst_mode >> 4) & 15, df = (k->dst_mode >> 8) & 15;
    for (unsigned ch = 0; ch < 4; ch++) {
        emit_blend_term(a, sf, true, ch);
        MOV(R9, AX);
        emit_blend_term(a, df, false, ch);
        a64_rrr(a, A64_ADD, false, AX, AX, R9);
        a64_alu(a, A64_BIC, false, AX, AX, AX, A64_ASR, 31); /* below 0 to 0 */
        a64_mov_imm(a, false, CX, 255);
        a64_cmp(a, false, AX, CX);
        a64_csel(a, false, AX, CX, AX, A64_GT);
        ST(AX, CTX, CTXF(sum[0]) + 4 * (int32_t) ch);
    }
    a64_ldst(a, A64_LDR_Q, 0, CTX, CTXF(sum[0]));
    a64_ldst(a, A64_STR_Q, 0, CTX, CTXF(c[0]));
}

/* c = f * (c - fog) / 255 + fog for red, green and blue, with f = sat(F)
   and C's truncating division, F at [base + f]. Clobbers W0, W5, W7, W8, W16,
   W17. */
static void
emit_fog(as *a, unsigned base, int32_t f)
{
    LD(UR, base, f);
    emit_sat_reg(a, UR);
    LD(VR, PE, PEF(fog_color));
    for (unsigned ch = 0; ch < 3; ch++) {
        static const unsigned sh[] = { 16, 8, 0 };
        int32_t c = CTXF(c[0]) + 4 * (int32_t) ch;
        a64_ubfx(a, false, R8, VR, sh[ch], 8);
        LD(AX, CTX, c);
        a64_rrr(a, A64_SUB, false, AX, AX, R8);
        a64_mul(a, false, AX, AX, UR);
        a64_mov_imm(a, false, A64_IP1, 255);
        a64_sdiv(a, false, AX, AX, A64_IP1);
        a64_rrr(a, A64_ADD, false, AX, AX, R8);
        ST(AX, CTX, c);
    }
}

static bool
key_supported(const span_key *k)
{
    unsigned func = (k->src_mode >> 4) & 7;
    unsigned fmt  = k->src_mode & 15 ? k->src_mode & 15 : k->dst_fmt;
    if ((k->dst_fmt != 4 && k->dst_fmt != 6) || (k->dst_mode & 15) != 12)
        return false;
    if (k->dst_mode & (DST_RAW_YUV | DST_RAW_BGR))
        return false;
    if (func > 3)
        return false;
    if (func && ((fmt != 4 && fmt != 6) || (k->src_mode & SRC_BGR)))
        return false;
    return true;
}

/* ctx.field = line_stride(pe->stride >> shift). Clobbers W0-W2, X16. */
static void
emit_stride(as *a, unsigned shift, int32_t field)
{
    LD(AX, PE, PEF(stride));
    a64_ubfx(a, false, CX, AX, shift + 4, 3);
    a64_ubfx(a, false, AX, AX, shift, 3);
    a64_mov_imm(a, true, A64_IP0, (uint64_t) (uintptr_t) stride_high);
    a64_ldst_reg(a, A64_LDR_W, CX, A64_IP0, CX, A64_UXTW, true);
    a64_mov_imm(a, true, A64_IP0, (uint64_t) (uintptr_t) stride_low);
    a64_ldst_reg(a, A64_LDR_W, DX, A64_IP0, AX, A64_UXTW, true);
    a64_rrr(a, A64_ADD, false, CX, CX, DX);
    ST(CX, CTX, field);
}

static unsigned
draw_count(uint8_t op)
{
    switch (op & 0xfe) {
        case 0xf0:
            return 2;
        case 0xf2:
            return 4;
        case 0xf8:
            return 6;
        case 0xfa:
            return 8;
        default:
            return 10;
    }
}

/* reg = the parameter register idx this pixel or, without one, the engine
   field, which then stays fixed for the span. */
static void
emit_source(as *a, unsigned reg, int idx, int32_t field)
{
    if (idx >= 0)
        LD(reg, CTX, CTXF(values[0]) + 4 * idx);
    else
        LD(reg, PE, field);
}

/* The constants for this state; see span_consts. */
static void
build_consts(span_consts *c, const span_key *k)
{
    /* Field width and position for R, G, B, A; a 565 target drops alpha. */
    static const uint8_t n565[] = { 5, 6, 5, 0 }, pos565[] = { 11, 5, 0, 0 };
    static const uint8_t n1555[] = { 5, 5, 5, 1 }, pos1555[] = { 10, 5, 0, 15 };
    bool is565 = k->dst_fmt == 4;
    memset(c, 0, sizeof(*c));
    for (unsigned ch = 0; ch < 4; ch++) {
        unsigned n = is565 ? n565[ch] : n1555[ch], pos = is565 ? pos565[ch] : pos1555[ch];
        bool     on = n != 0;
        c->k127[ch]  = 127;
        c->k8081[ch] = 0x8081;
        c->k255[ch]  = 255;
        /* A shift by -16 clears a lane the target has no field for. */
        c->down[ch]  = on ? -(int16_t) (8 - n) : -16;
        c->up[ch]    = on ? (int16_t) pos : 0;
        c->rgb[ch]   = ch == 3 ? 0 : 0xffff;
        c->alpha[ch] = ch == 3 ? 0xffff : 0;
        for (unsigned i = 0; i < 16; i++)
            c->dither[i][ch] = on ? (dither_table[i] << 4) >> n : 0;
    }
}

/* Generates a span_fn. op holds the register count, plus 0x100 for a single
   draw that writes back its stepped values and decrements Cnt.

   Within the loop the parameters feed the pixel straight from ctx.values;
   the engine fields pixel_set() would write are written once after it, from
   the last pixel's values. The colour stays in V2 as halfwords R, G, B, A. */
static uint8_t *
compile(as *a, const span_key *k)
{
    unsigned     count = k->op & 0xff;
    bool         single = !!(k->op & 0x100);
    unsigned     ctl   = k->drawctl;
    unsigned     i2    = (ctl >> 4) & 31;
    jit_pair     p2    = i2 <= 7 ? jit_pair3[i2] : jit_pair2[i2 - 8];
    jit_pair     p3 = jit_pair3[(ctl >> 9) & 15], p4 = jit_pair3[(ctl >> 13) & 7];
    pixel_param  params[] = { jit_pair1[ctl & 15].pair.even, jit_pair1[ctl & 15].pair.odd, p2.even, p2.odd,
                              p3.even, p3.odd, p4.even, p4.odd, P_G, P_B };
    bool         shadow   = jit_pair1[ctl & 15].shadow;
    int          q_index  = -1;
    /* The last register to set each value this pixel, or -1. */
    int          src_x = -1, src_y = -1, src_z = -1, src_u = -1, src_v = -1, src_f = -1;
    int          src_ch[4] = { -1, -1, -1, -1 };
    for (unsigned i = 0; i < count; i++) {
        switch (params[i]) {
            case P_Q:
                q_index = (int) i;
                break;
            case P_X:
                shadow = false;
                src_x  = (int) i;
                break;
            case P_Y:
                src_y = (int) i;
                break;
            case P_Z:
                src_z = (int) i;
                break;
            case P_F:
                src_f = (int) i;
                break;
            case P_U:
            case P_V:
                if ((params[i] == P_U) != !!(k->src_mode & SRC_SWAP_UV))
                    src_u = (int) i;
                else
                    src_v = (int) i;
                break;
            case P_R:
            case P_G:
            case P_B:
            case P_A:
                src_ch[params[i] == P_R ? 0 : params[i] == P_G ? 1 : params[i] == P_B ? 2 : 3] = (int) i;
                break;
            case P_I:
                src_ch[0] = src_ch[1] = src_ch[2] = (int) i;
                break;
            default:
                break;
        }
    }
    unsigned func       = (k->src_mode >> 4) & 7;
    unsigned fmt        = k->src_mode & 15 ? k->src_mode & 15 : k->dst_fmt;
    unsigned depth_mode = (k->dst_mode >> 12) & 7;
    bool     zwrite     = !!(k->dst_mode & DST_RAW_ZWRITE);
    bool     dither     = !!(k->dst_mode & DST_RAW_DITHER);
    /* An unpadded 565 texel survives replace, dither and encode unchanged:
       expansion leaves multiples of each channel's quantization step, which
       the dither term never reaches, and a keyed texel encodes as 0. Copy the
       texel itself, as pixel_copy_565() does. */
    bool raw_copy = func == 1 && fmt == 4 && k->dst_fmt == 4 && (k->src_mode & SRC_NO_PAD)
        && !(k->src_mode & (SRC_FILTER | SRC_BGR)) && !(k->dst_mode & (DST_RAW_BLEND | DST_RAW_FOG));

    /* Constants first, then the code. */
    while ((uintptr_t) a->p & 15)
        *a->p++ = 0;
    span_consts *kc = (span_consts *) a->p;
    build_consts(kc, k);
    a->p += (sizeof(span_consts) + 15) & ~(size_t) 15;
    uint8_t *start = a->p;

    /* Spans call nothing; only X19 and X20 need saving. */
    uint32_t frame = (sizeof(span_ctx) + 15) & ~15U;
    a64_pair(a, false, 3, LO, HI, A64_SP, -16);
    a64_alu_imm(a, A64_SUB, true, A64_SP, A64_SP, frame);
    a64_ldst(a, A64_STR_X, 0, CTX, CTXF(core));
    a64_ldst(a, A64_STR_X, 1, CTX, CTXF(site));
    a64_alu_imm(a, A64_ADD, true, PE, 0, offsetof(verite_core, pe));
    a64_ldst(a, A64_LDR_X, A64_IP1, 0, offsetof(verite_core, jit_mem));
    a64_ldst(a, A64_LDR_X, VRAM, A64_IP1, offsetof(verite_jit_memory, vram));
    a64_ldst(a, A64_LDR_W, MASK, A64_IP1, offsetof(verite_jit_memory, mask));
    a64_alu_imm(a, A64_AND, false, MASK, MASK, ~1U);
    a64_mov_imm(a, false, DREW, 0);
    a64_mov_imm(a, false, LO, UINT32_MAX);
    a64_mov_imm(a, false, HI, 0);
    a64_adr(a, KC, kc);

    /* cnt = (int16_t) Cnt; nothing happens unless it is positive. */
    a64_ldst(a, A64_LDRSH, AX, PE, PEF(cnt));
    ST(AX, CTX, CTXF(cnt));
    a64_cmp_imm(a, false, AX, 0);
    uint8_t *empty = a64_bcond(a, A64_LE);
    if (single)
        a64_mov_imm(a, false, COUNT, 1);
    else
        MOV(COUNT, AX);

    a64_ldst(a, A64_LDR_X, CX, CTX, CTXF(site));
    a64_ldst(a, A64_LDR_X, DX, CX, offsetof(verite_draw_site, values));
    a64_ldst(a, A64_LDR_X, CX, CX, offsetof(verite_draw_site, deltas));
    for (unsigned i = 0; i < count; i++) {
        LD(AX, DX, 4 * (int32_t) i);
        ST(AX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
        LD(AX, CX, 4 * (int32_t) i);
        ST(AX, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
    }

    if (shadow) {
        /* x_step = x_dir ? -1.0 : 1.0 in 16.16 */
        a64_ldst(a, A64_LDRB, AX, PE, PEF(x_dir));
        a64_mov_imm(a, false, CX, 0x10000);
        a64_cmp_imm(a, false, AX, 0);
        a64_cneg(a, false, CX, CX, A64_NE);
        ST(CX, CTX, CTXF(x_step));
    }
    emit_stride(a, 8, CTXF(dst_stride));
    if (depth_mode || zwrite)
        emit_stride(a, 16, CTXF(z_stride));
    if (func) {
        emit_stride(a, 0, CTXF(tex_stride));
        /* The first pixel has no step to extrapolate; its prefetch is a guess. */
        ST(A64_ZR, CTX, CTXF(prev_u));
        ST(A64_ZR, CTX, CTXF(prev_v));
    }

    /* Loop-invariant state: U and V masks, the scissor bounds [lo, min(hi,
       2048)) for each axis, row addresses for a fixed Y, the dither offset,
       the alpha threshold, argb(FgColor) and the texel table. */
    for (unsigned i = 0; i < count; i++)
        if (params[i] == P_U || params[i] == P_V)
            emit_uv_mask(a, params[i], k->src_mode);
    for (unsigned axis = 0; axis < 2; axis++) {
        LD(AX, PE, axis ? PEF(scissor_y) : PEF(scissor_x));
        a64_ubfx(a, false, CX, AX, 16, 12);
        ST(CX, CTX, CTXF(scissor[0]) + 8 * (int32_t) axis);
        a64_ubfx(a, false, AX, AX, 0, 12);
        a64_mov_imm(a, false, CX, 2048);
        a64_cmp(a, false, AX, CX);
        a64_csel(a, false, AX, CX, AX, A64_GT);
        ST(AX, CTX, CTXF(scissor[1]) + 8 * (int32_t) axis);
    }
    /* A fixed Y outside the scissor, or a fixed Z overflow under the Z
       scissor, rejects every pixel; so does an empty X range. */
    uint8_t *rejects[3];
    unsigned nreject = 0;
    if (src_y < 0) {
        LD(DX, PE, PEF(y));
        a64_shift_imm(a, A64_ASR, false, DX, DX, 16);
        LD(A64_IP0, CTX, CTXF(scissor[2]));
        a64_cmp(a, false, DX, A64_IP0);
        rejects[nreject++] = a64_bcond(a, A64_LT);
        LD(A64_IP0, CTX, CTXF(scissor[3]));
        a64_cmp(a, false, DX, A64_IP0);
        rejects[nreject++] = a64_bcond(a, A64_GE);
    }
    if ((k->src_mode & SRC_Z_SCISSOR) && src_z < 0) {
        a64_ldst(a, A64_LDRB, A64_IP0, PE, PEF(z_ovf));
        rejects[nreject++] = a64_cbz(a, true, false, A64_IP0);
    }
    if (nreject) {
        uint8_t *accept = a64_b(a);
        for (unsigned i = 0; i < nreject; i++)
            a64_land(rejects[i], a->p);
        ST(A64_ZR, CTX, CTXF(scissor[0]));
        ST(A64_ZR, CTX, CTXF(scissor[1]));
        a64_land(accept, a->p);
    }
    if (src_y < 0) {
        LD(A64_IP0, CTX, CTXF(dst_stride));
        LD(A64_IP1, PE, PEF(dst_base));
        a64_madd(a, false, AX, DX, A64_IP0, A64_IP1);
        ST(AX, CTX, CTXF(dst_row));
        if (depth_mode || zwrite) {
            LD(A64_IP0, CTX, CTXF(z_stride));
            LD(A64_IP1, PE, PEF(z_base));
            a64_madd(a, false, AX, DX, A64_IP0, A64_IP1);
            ST(AX, CTX, CTXF(z_row));
        }
    }
    if (dither && !raw_copy) {
        LD(DX, PE, PEF(dither_offset));
        a64_ubfx(a, false, AX, DX, 0, 2);
        ST(AX, CTX, CTXF(dither_x));
        a64_ubfx(a, false, DX, DX, 4, 2);
        ST(DX, CTX, CTXF(dither_y));
        if (src_y < 0) {
            /* dither_row = the four vectors for ((y + dither_y) & 3) */
            LD(AX, PE, PEF(y));
            a64_shift_imm(a, A64_ASR, false, AX, AX, 16);
            a64_rrr(a, A64_ADD, false, AX, AX, DX);
            a64_ubfx(a, false, AX, AX, 0, 2);
            a64_alu_imm(a, A64_ADD, true, CX, KC, KF(dither));
            /* rows of 4 vectors of 8 bytes; the 32-bit ops left X0's top clear */
            a64_alu(a, A64_ADD, true, CX, CX, AX, A64_LSL, 5);
            a64_ldst(a, A64_STR_X, CX, CTX, CTXF(dither_row));
        }
    }
    if ((k->dst_mode & DST_RAW_TREJECT) && !raw_copy) {
        LD(CX, PE, PEF(alpha_thres));
        a64_alu_imm(a, A64_AND, false, CX, CX, 255);
        ST(CX, CTX, CTXF(thres));
    }
    if (func != 1 && !raw_copy) {
        /* fg = argb(FgColor) as R, G, B, A with the lanes parameters fill 0 */
        static const unsigned byte[] = { 16, 8, 0, 24 };
        LD(AX, PE, PEF(fg));
        a64_mov_imm(a, true, CX, 0);
        for (unsigned lane = 0; lane < 4; lane++) {
            if (src_ch[lane] >= 0)
                continue;
            a64_ubfx(a, true, DX, AX, byte[lane], 8);
            a64_alu(a, A64_ORR, true, CX, CX, DX, A64_LSL, 16 * lane);
        }
        a64_ldst(a, A64_STR_X, CX, CTX, CTXF(fg));
    }
    if (func && !raw_copy) {
        a64_mov_imm(a, true, TABLE, (uint64_t) (uintptr_t) texel_table(fmt, !!(k->src_mode & SRC_NO_PAD)));
        /* Prefetch once the texels the masks reach, each row rounded up to
           cache lines, pass 16 KiB; smaller textures stay in the L1 cache,
           where the hint only costs time. */
        LD(AX, PE, PEF(u_mask));
        a64_rrr(a, A64_ADD, false, AX, AX, AX);
        a64_alu_imm(a, A64_ORR, false, AX, AX, 63);
        a64_alu_imm(a, A64_ADD, false, AX, AX, 1);
        LD(CX, PE, PEF(v_mask));
        a64_alu_imm(a, A64_ADD, false, CX, CX, 1);
        a64_mul(a, false, AX, AX, CX);
        a64_cmp_imm(a, false, AX, 16384);
        a64_cset(a, false, CX, A64_HI);
        ST(CX, CTX, CTXF(prefetch));
    }
    {
        static const int32_t konst[] = { KF(k127), KF(k8081), KF(k255), KF(down), KF(up), KF(rgb), KF(alpha) };
        for (unsigned i = 0; i < 7; i++)
            a64_ldst(a, A64_LDR_D, K127 + i, KC, konst[i]);
    }

    uint8_t *loop = a->p;

    /* W3 = x, W4 = y; scissor. */
    uint8_t *skips[16];
    unsigned nskip = 0;
    emit_source(a, XR, src_x, shadow ? PEF(x_shadow) : PEF(x));
    a64_shift_imm(a, A64_ASR, false, XR, XR, 16);
    /* A fixed Y only addresses a read destination. */
    if (src_y >= 0 || ((k->dst_mode & DST_RAW_BLEND) && !(k->dst_mode & DST_RAW_READ_DISABLE))) {
        emit_source(a, YR, src_y, PEF(y));
        a64_shift_imm(a, A64_ASR, false, YR, YR, 16);
    }
    for (unsigned axis = 0; axis < (src_y >= 0 ? 2U : 1U); axis++) {
        unsigned r = axis ? YR : XR;
        LD(A64_IP0, CTX, CTXF(scissor[0]) + 8 * (int32_t) axis);
        a64_cmp(a, false, r, A64_IP0);
        skips[nskip++] = a64_bcond(a, A64_LT);
        LD(A64_IP0, CTX, CTXF(scissor[1]) + 8 * (int32_t) axis);
        a64_cmp(a, false, r, A64_IP0);
        skips[nskip++] = a64_bcond(a, A64_GE);
    }
    if ((k->src_mode & SRC_Z_SCISSOR) && src_z >= 0) {
        LD(A64_IP0, CTX, CTXF(values[0]) + 4 * src_z);
        a64_tst_imm(a, false, A64_IP0, 0xf8000000);
        skips[nskip++] = a64_bcond(a, A64_NE);
    }

    if (depth_mode || zwrite) {
        if (src_y >= 0) {
            LD(A64_IP0, CTX, CTXF(z_stride));
            LD(A64_IP1, PE, PEF(z_base));
            a64_madd(a, false, AX, YR, A64_IP0, A64_IP1);
        } else
            LD(AX, CTX, CTXF(z_row));
        a64_alu(a, A64_ADD, false, AX, AX, XR, A64_LSL, 1);
        ST(AX, CTX, CTXF(z_addr));
        emit_source(a, CX, src_z, PEF(z));
        a64_ubfx(a, false, CX, CX, 11, 16);
        ST(CX, CTX, CTXF(z));
        if (depth_mode == 7)
            skips[nskip++] = a64_b(a);
        else if (depth_mode) {
            /* conditions that fail the test of z against the stored depth */
            static const uint8_t fail[] = { 0, A64_HS, A64_NE, A64_HI, A64_LS, A64_EQ, A64_LO };
            emit_load16(a, DX, AX);
            a64_cmp(a, false, CX, DX);
            skips[nskip++] = a64_bcond(a, fail[depth_mode]);
        }
    }

    if (func) {
        emit_source(a, UR, src_u, PEF(u));
        emit_source(a, VR, src_v, PEF(v));
        if (q_index >= 0)
            emit_divide_uv(a);
    }
    if (raw_copy) {
        bool uc = !!(k->src_mode & SRC_U_CLAMP), vc = !!(k->src_mode & SRC_V_CLAMP);
        a64_shift_imm(a, A64_ASR, false, AX, VR, 16);
        emit_coord_reg(a, AX, PEF(v_mask), vc);
        LD(A64_IP0, CTX, CTXF(tex_stride));
        LD(A64_IP1, PE, PEF(src_base));
        a64_madd(a, false, R9, AX, A64_IP0, A64_IP1);
        a64_shift_imm(a, A64_ASR, false, AX, UR, 16);
        emit_coord_reg(a, AX, PEF(u_mask), uc);
        a64_alu(a, A64_ADD, false, AX, R9, AX, A64_LSL, 1);
        emit_load16(a, R8, AX);
        uint8_t *opaque = NULL;
        if (k->src_mode & SRC_CHROMA) {
            a64_mov_imm(a, false, A64_IP1, 0x10001);
            a64_mul(a, false, CX, R8, A64_IP1);
            LD(A64_IP1, PE, PEF(chroma_color));
            a64_rrr(a, A64_EOR, false, CX, CX, A64_IP1);
            LD(A64_IP1, PE, PEF(chroma_mask));
            a64_rrr(a, A64_ANDS, false, CX, CX, A64_IP1);
            opaque = a64_bcond(a, A64_NE);
            /* Keyed: alpha 0 is always rejected; otherwise it encodes as 0,
               or YUV black (128, 16, 128) as 0x8090, which dither never
               changes either. */
            if (k->dst_mode & DST_RAW_TREJECT)
                skips[nskip++] = a64_b(a);
            else
                a64_mov_imm(a, false, R8, k->src_mode & SRC_CHROMA_BLACK_YUV ? 0x8090 : 0);
            uint8_t *keyed = a64_b(a);
            a64_land(opaque, a->p);
            opaque = keyed;
        }
        if (k->dst_mode & DST_RAW_TREJECT) {
            /* Alpha 255 is rejected only by a threshold of 255. */
            LD(CX, PE, PEF(alpha_thres));
            a64_alu_imm(a, A64_AND, false, CX, CX, 255);
            a64_cmp_imm(a, false, CX, 255);
            skips[nskip++] = a64_bcond(a, A64_EQ);
        }
        if (opaque)
            a64_land(opaque, a->p);
    } else {
        /* V3 = t; V2 = c = argb(fg) with the colour parameters set. */
        if (func)
            emit_sample(a, k->src_mode);
        if (func != 1) {
            a64_ldst(a, A64_LDR_D, 2, CTX, CTXF(fg));
            for (unsigned ch = 0; ch < 4; ch++) {
                int  idx   = src_ch[ch];
                bool first = idx >= 0;
                for (unsigned prev = 0; prev < ch; prev++)
                    first = first && src_ch[prev] != idx;
                if (!first)
                    continue;
                /* sat(value) into each lane it sets; those lanes of fg are 0. */
                LD(AX, CTX, CTXF(values[0]) + 4 * idx);
                emit_sat(a);
                for (unsigned lane = 0; lane < 4; lane++)
                    if (src_ch[lane] == idx)
                        a64_ins(a, 2, lane, AX);
            }
        }
        if (func == 1)
            a64_vmov(a, 2, 3);
        else if (func == 3) {
            /* (c t + 127) / 255 for c t + 127 < 65536: (x * 0x8081) >> 23. */
            A64_VMUL(a, V4H, 2, 2, 3);
            A64_VADD(a, V4H, 2, 2, K127);
            a64_umull(a, 0, 2, K8081);
            a64_shrn(a, false, 2, 0, 16);
            a64_ushr4h(a, 2, 2, 7);
        } else if (func == 2) {
            /* Decal: (c (255 - ta) + t ta + 127) / 255; alpha stays c's. */
            a64_dup_lane(a, 0, 3, 3);
            A64_VSUB(a, V4H, 1, K255, 0);
            A64_VMUL(a, V4H, 0, 0, 3);
            A64_VMLA(a, V4H, 0, 1, 2);
            A64_VADD(a, V4H, 0, 0, K127);
            a64_umull(a, 1, 0, K8081);
            a64_shrn(a, false, 0, 1, 16);
            a64_ushr4h(a, 0, 0, 7);
            A64_VAND(a, 0, 0, KRGB);
            A64_VAND(a, 2, 2, KALPHA);
            A64_VORR(a, 2, 2, 0);
        }

        if (k->dst_mode & DST_RAW_TREJECT) {
            a64_umov(a, AX, 2, 3);
            LD(A64_IP0, CTX, CTXF(thres));
            a64_cmp(a, false, AX, A64_IP0);
            skips[nskip++] = a64_bcond(a, A64_LE);
        }

        if (k->dst_mode & (DST_RAW_FOG | DST_RAW_BLEND)) {
            /* ctx.c = V2 as words, for the scalar stages; and back. */
            a64_uxtl(a, true, 0, 2);
            a64_ldst(a, A64_STR_Q, 0, CTX, CTXF(c[0]));
            if (k->dst_mode & DST_RAW_FOG) {
                if (src_f >= 0)
                    emit_fog(a, CTX, CTXF(values[0]) + 4 * src_f);
                else
                    emit_fog(a, PE, PEF(f));
            }
            if (k->dst_mode & DST_RAW_BLEND)
                emit_blend(a, k);
            a64_ldst(a, A64_LDR_Q, 2, CTX, CTXF(c[0]));
            a64_xtn(a, 2, 2);
        }

        /* Encode into W5: per field, min(c + (d << 4 >> n), 255) >> (8 - n)
           << pos, the fields summed across the vector. */
        if (dither && src_y < 0) {
            LD(AX, CTX, CTXF(dither_x));
            a64_rrr(a, A64_ADD, false, AX, AX, XR);
            a64_ubfx(a, false, AX, AX, 0, 2);
            a64_ldst(a, A64_LDR_X, A64_IP0, CTX, CTXF(dither_row));
            a64_ldst_reg(a, A64_LDR_D, 0, A64_IP0, AX, A64_UXTW, true);
            A64_VADD(a, V4H, 2, 2, 0);
            A64_VSMIN(a, V4H, 2, 2, K255);
        } else if (dither) {
            LD(AX, CTX, CTXF(dither_x));
            a64_rrr(a, A64_ADD, false, AX, AX, XR);
            a64_ubfx(a, false, AX, AX, 0, 2);
            LD(CX, CTX, CTXF(dither_y));
            a64_rrr(a, A64_ADD, false, CX, CX, YR);
            a64_ubfx(a, false, CX, CX, 0, 2);
            a64_alu(a, A64_ADD, false, AX, AX, CX, A64_LSL, 2);
            a64_alu_imm(a, A64_ADD, true, A64_IP0, KC, KF(dither));
            a64_ldst_reg(a, A64_LDR_D, 0, A64_IP0, AX, A64_UXTW, true);
            A64_VADD(a, V4H, 2, 2, 0);
            A64_VSMIN(a, V4H, 2, 2, K255);
        }
        A64_VUSHL(a, V4H, 2, 2, KDOWN);
        A64_VUSHL(a, V4H, 2, 2, KUP);
        a64_addv4h(a, 0, 2);
        a64_umov(a, R8, 0, 0);
    }

    /* addr = dst_base + dst_stride * y + 2x */
    if (src_y >= 0) {
        LD(A64_IP0, CTX, CTXF(dst_stride));
        LD(A64_IP1, PE, PEF(dst_base));
        a64_madd(a, false, AX, YR, A64_IP0, A64_IP1);
    } else
        LD(AX, CTX, CTXF(dst_row));
    a64_alu(a, A64_ADD, false, AX, AX, XR, A64_LSL, 1);
    emit_store16(a, R8, AX);
    if (zwrite) {
        LD(AX, CTX, CTXF(z_addr));
        LD(CX, CTX, CTXF(z));
        emit_store16(a, CX, AX);
    }

    for (unsigned i = 0; i < nskip; i++)
        a64_land(skips[i], a->p);
    if (q_index >= 0) {
        LD(AX, CTX, CTXF(values[0]) + 4 * q_index);
        ST(AX, PE, PEF(q));
    }
    if (shadow) {
        LD(AX, CTX, CTXF(x_step));
        LD(CX, PE, PEF(x_shadow));
        a64_rrr(a, A64_ADD, false, CX, CX, AX);
        ST(CX, PE, PEF(x_shadow));
    }
    for (unsigned i = 0; i < count; i += 4) {
        a64_ldst(a, A64_LDR_Q, 0, CTX, CTXF(values[0]) + 4 * (int32_t) i);
        a64_ldst(a, A64_LDR_Q, 1, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
        A64_VADD(a, V4S, 0, 0, 1);
        a64_ldst(a, A64_STR_Q, 0, CTX, CTXF(values[0]) + 4 * (int32_t) i);
    }
    a64_alu_imm(a, A64_SUBS, false, COUNT, COUNT, 1);
    a64_bcond_to(a, A64_NE, loop);

    /* The engine as pixel_set() left it for the last pixel. */
    for (unsigned i = 0; i < count; i++) {
        if (params[i] == P_NONE || params[i] == P_Q)
            continue;
        LD(AX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
        LD(CX, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
        a64_rrr(a, A64_SUB, false, AX, AX, CX);
        emit_set(a, params[i], k->src_mode);
    }
    /* Every written pixel lowered lo. */
    a64_cmp_imm(a, false, LO, UINT32_MAX);
    a64_cset(a, false, DREW, A64_NE);

    if (single) {
        /* dest[k] = value + delta (already stepped); Cnt decrements. */
        a64_ldst(a, A64_LDR_X, CX, CTX, CTXF(site));
        a64_ldst(a, A64_LDR_X, CX, CX, offsetof(verite_draw_site, dest));
        uint8_t *no_dest = a64_cbz(a, false, true, CX);
        for (unsigned i = 0; i < count; i++) {
            LD(AX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
            ST(AX, CX, 4 * (int32_t) i);
        }
        a64_land(no_dest, a->p);
        LD(AX, CTX, CTXF(cnt));
        a64_alu_imm(a, A64_SUB, false, AX, AX, 1);
        ST(AX, PE, PEF(cnt));
    } else
        ST(A64_ZR, PE, PEF(cnt));
    uint8_t *none = a64_cbz(a, false, false, DREW);
    a64_mov_imm(a, false, AX, 1);
    a64_ldst(a, A64_STRB, AX, PE, PEF(pick_hit));
    a64_ldst(a, A64_LDR_X, CX, CTX, CTXF(core));
    a64_ldst(a, A64_STRB, AX, CX, offsetof(verite_core, drew));
    a64_land(none, a->p);

    a64_land(empty, a->p);
    a64_ldst(a, A64_LDR_X, CX, CTX, CTXF(site));
    ST(LO, CX, offsetof(verite_draw_site, lo));
    ST(HI, CX, offsetof(verite_draw_site, hi));
    MOV(AX, DREW);
    a64_alu_imm(a, A64_ADD, true, A64_SP, A64_SP, frame);
    a64_pair(a, true, 1, LO, HI, A64_SP, 16);
    a64_ret(a);
    return start;
}

void
pixel_jit_free(verite_core *c)
{
    if (c->pixel_jit) {
        a64_free_code(c->pixel_jit->arena, SPAN_ARENA);
        free(c->pixel_jit);
        c->pixel_jit = NULL;
    }
}

static span_fn
lookup(verite_core *c, const span_key *key)
{
    struct verite_pixel_jit *jit = c->pixel_jit;
    if (!jit) {
        static bool tables;
        if (!tables) {
            init_tables();
            tables = true;
        }
        jit = calloc(1, sizeof(*jit));
        if (!jit || !(jit->arena = a64_alloc_code(SPAN_ARENA))) {
            free(jit);
            c->jit_unavailable = true;
            return NULL;
        }
        c->pixel_jit = jit;
    }
    for (unsigned i = 0; i < SPAN_CACHE; i++) {
        span_entry *e = &jit->entries[i];
        if (e->used && !memcmp(&e->key, key, sizeof(*key)))
            return e->fn;
    }
    if (jit->used + SPAN_MAX_CODE > SPAN_ARENA) {
        /* Draw sites in RISC blocks point into the arena. */
        memset(jit->entries, 0, sizeof(jit->entries));
        jit->used = 0;
        risc_jit_reset_sites(c);
    }
    /* Build the texel table outside the write window. */
    unsigned fmt = key->src_mode & 15 ? key->src_mode & 15 : key->dst_fmt;
    if ((key->src_mode >> 4) & 7)
        texel_table(fmt, !!(key->src_mode & SRC_NO_PAD));
    uint8_t *base = jit->arena + jit->used;
    as       a    = { base };
    a64_write_begin();
    uint8_t *code = compile(&a, key); /* after its constants */
    size_t   size = (size_t) (a.p - base);
    a64_write_end(base, size);
    if (size > SPAN_MAX_CODE)
        abort();
    jit->used += (size + 15) & ~(size_t) 15;
    span_entry *e = &jit->entries[jit->next];
    jit->next     = (jit->next + 1) % SPAN_CACHE;
    *e            = (span_entry) { *key, (span_fn) (void *) code, true };
    return e->fn;
}

/* The compiled span for the current render state, or NULL to use C. */
static span_fn
state_span(verite_core *c, uint8_t op)
{
    pixel_engine *pe = &c->pe;
    if (c->jit_disabled || c->jit_unavailable || (pe->pat_mode & 0x10000) || (pe->pmask & 0xffff) != 0xffff)
        return NULL;
    span_key key = {
        .op       = draw_count(op) | (op & 1 ? 0 : 0x100),
        .drawctl  = pe->drawctl & 0xffff,
        .src_mode = pe->src_mode & 0x1ffff,
        .dst_mode = pe->dst_mode & 0x03ff7fff,
        .dst_fmt  = pe->dst_fmt & 15,
    };
    return key_supported(&key) ? lookup(c, &key) : NULL;
}

/* From pixel_draw_param(): the caller writes back a single draw's registers. */
bool
pixel_jit_span(verite_core *c, uint8_t op, const uint32_t values[], const uint32_t deltas[], verite_memory mem)
{
    if (mem.mask > UINT32_MAX)
        return false;
    span_fn fn = state_span(c, op);
    if (!fn)
        return false;
    verite_jit_memory        memory = { mem.data, (uint32_t) mem.mask };
    verite_draw_site         site   = { .values = values, .deltas = deltas };
    const verite_jit_memory *outer  = c->jit_mem;
    c->jit_mem                      = &memory;
    fn(c, &site);
    c->jit_mem = outer;
    return true;
}

/* A draw site's key did not match: compile or find the span for this state
   and remember it, or run the C pixel engine. A C draw reports the whole of
   VRAM as written, which ends the block. */
unsigned
pixel_jit_site_miss(verite_core *c, verite_draw_site *s)
{
    pixel_engine *pe = &c->pe;
    uint8_t       op = s->ir >> 24;
    span_fn       fn = state_span(c, op);
    if (fn) {
        s->drawctl  = pe->drawctl;
        s->src_mode = pe->src_mode;
        s->dst_mode = pe->dst_mode;
        s->dst_fmt  = pe->dst_fmt;
        s->pmask    = pe->pmask;
        s->pat_mode = pe->pat_mode;
        s->fn       = (void *) fn;
        return fn(c, s);
    }
    verite_memory mem = { c->jit_mem->vram, c->jit_mem->mask };
    pixel_draw_param(c, op, s->ir >> 16, s->ir >> 8, s->ir, mem);
    s->lo = 0;
    s->hi = UINT32_MAX;
    return 0;
}

#endif
