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
    SPAN_CACHE           = 512,
    SPAN_ARENA           = 2 * 1024 * 1024,
    SPAN_MAX_CODE        = 16 * 1024,
    PREFETCH_SHIFT       = 2, /* look 1 << PREFETCH_SHIFT pixels ahead */
    DST_RAW_ZWRITE       = 1 << 16,
    DST_RAW_YUV          = 1 << 18,
    DST_RAW_BLEND        = 1 << 19,
    DST_RAW_DITHER       = 1 << 20,
    DST_RAW_FOG          = 1 << 21,
    DST_RAW_BGR          = 1 << 24,
    DST_RAW_NO_PAD       = 1 << 22,
    DST_RAW_READ_DISABLE = 1 << 23,
    DST_RAW_TREJECT      = 1 << 25,
};

/* The span's stack frame at SP. */
typedef struct {
    uint32_t          values[12], deltas[12];
    uint32_t          pal[16];
    uint16_t          fg[4], fog[4], fog_f[4], dst_c[4];
    verite_core      *core;
    verite_draw_site *site;
    const uint16_t   *dither_row; /* the dither vectors for a fixed Y */
    const uint32_t   *dst_table;
    int32_t           cnt;
    uint32_t          x_step, dst_stride, z_stride, tex_stride;
    uint32_t          z_addr, z, dst_addr, byte_mask;
    uint32_t          zlo, zhi, tail, x_dir_step;
    int32_t           x_first;
    uint32_t          prev_u, prev_v; /* the last pixel's texel coordinates */
    int32_t           scissor[4];     /* x lo, x hi, y lo, y hi */
    uint32_t          dst_row, z_row; /* row addresses when Y is no parameter */
    uint32_t          dither_x, dither_y, thres;
    uint32_t          prefetch; /* whether the texture outgrows the L1 cache */
} span_ctx;
typedef char span_ctx_aligned[offsetof(span_ctx, deltas) % 16 == 0 && offsetof(span_ctx, fg) % 8 == 0 ? 1 : -1];

/* Per-state constants, placed before the span's code and loaded into V16-V22
   before its loop. Vectors hold halfwords R, G, B, A. */
typedef struct {
    uint16_t k127[4], k128[4], k8081[4], k255[4];
    int16_t  down[4]; /* x >> (8 - n) as a USHL count, n the field width */
    int16_t  up[4];   /* the field's position as a USHL count */
    uint16_t rgb[4], alpha[4], alpha255[4];
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
    unsigned   count;
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

static uint8_t        widen_table[3][64]; /* 4, 5 and 6 bits, padded */
static uint16_t       stride_table[128];
static const uint8_t  dither_table[16] = { 0, 12, 3, 15, 7, 11, 4, 8, 13, 1, 14, 2, 10, 6, 9, 5 };

static void
init_tables(void)
{
    static const unsigned bits[] = { 4, 5, 6 };
    for (unsigned t = 0; t < 3; t++) {
        unsigned max = (1U << bits[t]) - 1;
        for (unsigned v = 0; v <= max; v++)
            widen_table[t][v] = (v * 255 + max / 2) / max;
    }
    for (unsigned i = 0; i < 128; i++)
        stride_table[i] = (uint16_t) verite_line_stride(i);
}

/* Every 16-bit texel of a 565 (fmt 4), 4444 (fmt 5), or 1555 (fmt 6) texture,
   padded or not, decoded as texel() does into R | G << 8 | B << 16 | A << 24.
   Built on first use. */
static uint32_t *texel_tables[3][2];

static const uint32_t *
texel_table(unsigned fmt, bool no_pad)
{
    unsigned   slot  = fmt == 4 ? 0 : fmt == 5 ? 1 : 2;
    uint32_t **table = &texel_tables[slot][no_pad];
    if (*table)
        return *table;
    uint32_t *t = malloc(65536 * sizeof(uint32_t));
    if (!t)
        abort();
    static const unsigned shift565[]  = { 11, 5, 0 }, bits565[] = { 5, 6, 5 };
    static const unsigned shift4444[] = { 8, 4, 0 }, shift1555[] = { 10, 5, 0 };
    for (unsigned texel = 0; texel < 65536; texel++) {
        uint32_t ch[4];
        for (unsigned k = 0; k < 3; k++) {
            unsigned shift = fmt == 4 ? shift565[k] : fmt == 5 ? shift4444[k] : shift1555[k];
            unsigned bits  = fmt == 4 ? bits565[k] : fmt == 5 ? 4 : 5;
            unsigned v     = (texel >> shift) & ((1U << bits) - 1);
            ch[k]          = no_pad ? v << (8 - bits) : widen_table[bits - 4][v];
        }
        if (fmt == 5) {
            unsigned a = (texel >> 12) & 15;
            ch[3]      = no_pad ? a << 4 : widen_table[0][a];
        } else {
            ch[3] = fmt == 4 || (texel & 0x8000) ? 255 : 0;
        }
        t[texel] = ch[0] | ch[1] << 8 | ch[2] << 16 | ch[3] << 24;
    }
    return *table = t;
}

/* ---- Registers ------------------------------------------------------------ */

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

/* dst = 8-bit VRAM byte at addr (clobbers addr, W16). */
static void
emit_load8(as *a, unsigned dst, unsigned addr)
{
    LD(A64_IP0, CTX, CTXF(byte_mask));
    a64_rrr(a, A64_AND, false, addr, addr, A64_IP0);
    a64_alu_imm(a, A64_EOR, false, addr, addr, 3);
    a64_ldst_reg(a, A64_LDRB, dst, VRAM, addr, A64_UXTW, false);
}

/* dst = 16-bit VRAM word at the address in addr (clobbers addr). */
static void
emit_load16(as *a, unsigned dst, unsigned addr)
{
    a64_rrr(a, A64_AND, false, addr, addr, MASK);
    a64_alu_imm(a, A64_EOR, false, addr, addr, 2);
    a64_ldst_reg(a, A64_LDRH, dst, VRAM, addr, A64_UXTW, false);
}

/* dst = 32-bit VRAM dword at addr (clobbers addr). */
static void
emit_load32(as *a, unsigned dst, unsigned addr)
{
    a64_rrr(a, A64_AND, false, addr, addr, MASK);
    a64_alu_imm(a, A64_AND, false, addr, addr, ~3U);
    a64_ldst_reg(a, A64_LDR_W, dst, VRAM, addr, A64_UXTW, false);
}

/* Store the low 16 bits of val at the address in addr (clobbers addr, A64_IP0). */
static void
emit_store16(as *a, unsigned val, unsigned addr, bool track_bounds, bool is_z)
{
    a64_rrr(a, A64_AND, false, addr, addr, MASK);
    if (track_bounds && is_z) {
        LD(A64_IP0, CTX, CTXF(zlo));
        a64_cmp(a, false, addr, A64_IP0);
        a64_csel(a, false, A64_IP0, addr, A64_IP0, A64_LO);
        ST(A64_IP0, CTX, CTXF(zlo));
        LD(A64_IP0, CTX, CTXF(zhi));
        a64_cmp(a, false, addr, A64_IP0);
        a64_csel(a, false, A64_IP0, addr, A64_IP0, A64_HI);
        ST(A64_IP0, CTX, CTXF(zhi));
    } else if (track_bounds) {
        a64_cmp(a, false, addr, LO);
        a64_csel(a, false, LO, addr, LO, A64_LO);
        a64_cmp(a, false, addr, HI);
        a64_csel(a, false, HI, addr, HI, A64_HI);
    }
    a64_alu_imm(a, A64_EOR, false, addr, addr, 2);
    a64_ldst_reg(a, A64_STRH, val, VRAM, addr, A64_UXTW, false);
}

/* Store 32-bit val at addr (clobbers addr, W16). */
static void
emit_store32(as *a, unsigned val, unsigned addr, bool track_bounds)
{
    a64_rrr(a, A64_AND, false, addr, addr, MASK);
    a64_alu_imm(a, A64_AND, false, addr, addr, ~3U);
    if (track_bounds) {
        a64_cmp(a, false, addr, LO);
        a64_csel(a, false, LO, addr, LO, A64_LO);
        a64_alu_imm(a, A64_ADD, false, A64_IP0, addr, 2);
        a64_cmp(a, false, A64_IP0, HI);
        a64_csel(a, false, HI, A64_IP0, HI, A64_HI);
    }
    a64_ldst_reg(a, A64_STR_W, val, VRAM, addr, A64_UXTW, false);
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

static void
emit_uv_mask(as *a, pixel_param param, uint32_t src_mode)
{
    bool u = (param == P_U) != !!(src_mode & SRC_SWAP_UV);
    LD(CX, PE, PEF(src_mask));
    a64_ubfx(a, false, CX, CX, u ? 0 : 16, 11);
    ST(CX, PE, u ? PEF(u_mask) : PEF(v_mask));
}

static void
emit_coord_reg(as *a, unsigned reg, int32_t mask_field, bool clamped)
{
    LD(A64_IP0, PE, mask_field);
    if (clamped) {
        a64_cmp(a, false, reg, A64_IP0);
        a64_csel(a, false, reg, A64_IP0, reg, A64_GT);
        a64_alu(a, A64_BIC, false, reg, reg, reg, A64_ASR, 31);
    } else
        a64_rrr(a, A64_AND, false, reg, reg, A64_IP0);
}

/* Vv = decoded texel at (row, x) as bytes R, G, B, A (or B, G, R, A for
   fmt == 12). Clobbers W6, W16, W17. */
static void
emit_tap(as *a, unsigned fmt, uint32_t src_mode, unsigned v, unsigned row, unsigned x)
{
    if (fmt >= 8 && fmt <= 10) {
        a64_alu(a, A64_ADD, false, R9, row, x, A64_LSR, 1);
        emit_load8(a, R9, R9);
        a64_alu_imm(a, A64_AND, false, A64_IP0, R9, 15);
        a64_shift_imm(a, A64_LSR, false, R9, R9, 4);
        a64_tst_imm(a, false, x, 1);
        a64_csel(a, false, R9, A64_IP0, R9, A64_NE);
        a64_alu_imm(a, A64_ADD, true, A64_IP0, CTX, CTXF(pal[0]));
        a64_ldst_reg(a, A64_LDR_S, v, A64_IP0, R9, A64_UXTW, true);
        return;
    }
    if (fmt == 2 || fmt == 3) {
        a64_rrr(a, A64_ADD, false, R9, row, x);
        emit_load8(a, R9, R9);
        uint8_t *opaque = NULL;
        if (src_mode & SRC_CHROMA) {
            a64_mov_imm(a, false, A64_IP1, 0x01010101);
            a64_mul(a, false, A64_IP0, R9, A64_IP1);
            LD(A64_IP1, PE, PEF(chroma_color));
            a64_rrr(a, A64_EOR, false, A64_IP0, A64_IP0, A64_IP1);
            LD(A64_IP1, PE, PEF(chroma_mask));
            a64_rrr(a, A64_ANDS, false, A64_IP0, A64_IP0, A64_IP1);
            opaque = a64_bcond(a, A64_NE);
            a64_mov_imm(a, false, A64_IP0, src_mode & SRC_CHROMA_BLACK_YUV ? 128 | 16 << 8 | 128 << 16 : 0);
            a64_fmov_from(a, v, A64_IP0);
            uint8_t *keyed = a64_b(a);
            a64_land(opaque, a->p);
            opaque = keyed;
        }
        if (fmt == 2) {
            a64_mov_imm(a, false, A64_IP0, 0x010101);
            a64_mul(a, false, R9, R9, A64_IP0);
            a64_alu_imm(a, A64_ORR, false, R9, R9, 0xff000000U);
        } else {
            a64_shift_imm(a, A64_LSL, false, R9, R9, 24);
            a64_alu_imm(a, A64_ORR, false, R9, R9, 0x00ffffffU);
        }
        a64_fmov_from(a, v, R9);
        if (opaque)
            a64_land(opaque, a->p);
        return;
    }
    if (fmt == 12) {
        a64_alu(a, A64_ADD, false, R9, row, x, A64_LSL, 2);
        emit_load32(a, R9, R9);
        a64_fmov_from(a, v, R9);
        if (src_mode & SRC_CHROMA) {
            LD(A64_IP1, PE, PEF(chroma_color));
            a64_rrr(a, A64_EOR, false, A64_IP0, R9, A64_IP1);
            LD(A64_IP1, PE, PEF(chroma_mask));
            a64_rrr(a, A64_ANDS, false, A64_IP0, A64_IP0, A64_IP1);
            uint8_t *opaque = a64_bcond(a, A64_NE);
            a64_mov_imm(a, false, A64_IP0, src_mode & SRC_CHROMA_BLACK_YUV ? 128 | 16 << 8 | 128 << 16 : 0);
            a64_fmov_from(a, v, A64_IP0);
            a64_land(opaque, a->p);
        }
        return;
    }
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

static void
emit_prefetch(as *a, unsigned fmt)
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
    if (fmt >= 8 && fmt <= 10)
        a64_alu(a, A64_ADD, false, CX, CX, AX, A64_LSR, 1);
    else if (fmt == 12)
        a64_alu(a, A64_ADD, false, CX, CX, AX, A64_LSL, 2);
    else if (fmt == 2 || fmt == 3)
        a64_rrr(a, A64_ADD, false, CX, CX, AX);
    else
        a64_alu(a, A64_ADD, false, CX, CX, AX, A64_LSL, 1);
    a64_rrr(a, A64_AND, false, CX, CX, MASK);
    a64_prefetch(a, VRAM, CX);
}

static void
emit_sample(as *a, unsigned fmt, uint32_t src_mode, bool single)
{
    bool uc = !!(src_mode & SRC_U_CLAMP), vc = !!(src_mode & SRC_V_CLAMP);
    bool filter = !!(src_mode & SRC_FILTER);
    if (!single) {
        LD(A64_IP0, CTX, CTXF(prefetch));
        uint8_t *cached = a64_cbz(a, false, false, A64_IP0);
        emit_prefetch(a, fmt);
        a64_land(cached, a->p);
    }

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
        emit_tap(a, fmt, src_mode, 3, DX, AX);
        a64_uxtl(a, false, 3, 3);
    } else {
        for (unsigned axis = 0; axis < 2; axis++) {
            a64_ubfx(a, false, A64_IP0, axis ? VR : UR, 8, 8);
            a64_dup(a, false, axis ? WV1 : WU1, A64_IP0);
            a64_mov_imm(a, false, A64_IP1, 256);
            a64_rrr(a, A64_SUB, false, A64_IP0, A64_IP1, A64_IP0);
            a64_dup(a, false, axis ? WV0 : WU0, A64_IP0);
        }

        emit_tap(a, fmt, src_mode, 0, DX, AX);
        emit_tap(a, fmt, src_mode, 1, DX, CX);
        emit_tap(a, fmt, src_mode, 4, R8, AX);
        emit_tap(a, fmt, src_mode, 5, R8, CX);
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
    if (fmt == 12) {
        a64_umov(a, AX, 3, 0);
        a64_umov(a, CX, 3, 2);
        a64_ins(a, 3, 0, CX);
        a64_ins(a, 3, 2, AX);
    }
}

/* W7, W8 = u, v * 2^24 / q unless Q is 0 (exact truncating division).
   Uses double-precision reciprocal + branchless remainder fixup for q >= 64.
   Clobbers W0, W1, W5, X16, D0-D1. Preserves X1 (for vid_verite_test.c:1240). */
static void
emit_divide_uv(as *a)
{
    LD(R8, PE, PEF(q));
    uint8_t *flat = a64_cbz(a, false, false, R8);
    a64_cmp_imm(a, false, R8, 64);
    uint8_t *fast = a64_bcond(a, A64_HS);
    for (unsigned axis = 0; axis < 2; axis++) {
        unsigned coord = axis ? VR : UR;
        a64_sxtw(a, A64_IP0, coord);
        a64_shift_imm(a, A64_LSL, true, A64_IP0, A64_IP0, 24);
        a64_sdiv(a, true, A64_IP0, A64_IP0, R8);
        MOV(coord, A64_IP0);
    }
    uint8_t *done = a64_b(a);

    a64_land(fast, a->p);
    /* D0 = 2^24 / q (R8 is zero-extended 32-bit unsigned) */
    a64_scvtf_d(a, true, 0, R8);
    a64_mov_imm(a, true, A64_IP0, UINT64_C(0x4170000000000000)); /* 16777216.0 */
    a64_fmov_d_x(a, 1, A64_IP0);
    a64_fdiv_d(a, 0, 1, 0);
    for (unsigned axis = 0; axis < 2; axis++) {
        unsigned coord = axis ? VR : UR;
        a64_scvtf_d(a, false, 1, coord);
        a64_fmul_d(a, 1, 1, 0);
        a64_fcvtzs_x_d(a, A64_IP1, 1); /* X17 = estimate */
        a64_sxtw(a, AX, coord);
        a64_shift_imm(a, A64_LSL, true, AX, AX, 24);
        a64_msub(a, true, AX, A64_IP1, R8, AX); /* X0 = rem = num - est * q */

        a64_mov_imm(a, true, A64_IP0, 1);
        a64_cmp_imm(a, false, coord, 0);
        a64_csel(a, true, A64_IP0, R8, A64_IP0, A64_GE);
        a64_cmp(a, true, AX, A64_IP0);
        a64_csinc(a, true, A64_IP1, A64_IP1, A64_IP1, A64_LT);

        a64_rrr(a, A64_ADD, true, A64_IP0, AX, R8);
        a64_alu_imm(a, A64_SUB, true, A64_IP0, A64_IP0, 1);
        a64_cmp_imm(a, false, coord, 0);
        a64_csel(a, true, AX, A64_IP0, AX, A64_LT);
        a64_cmp_imm(a, true, AX, 0);
        a64_csinc(a, true, A64_IP0, A64_ZR, A64_ZR, A64_GE);
        a64_rrr(a, A64_SUB, true, A64_IP1, A64_IP1, A64_IP0);
        MOV(coord, A64_IP1);
    }
    a64_land(done, a->p);
    a64_land(flat, a->p);
}

/* Vectorized NEON fog in V2. Clobbers W0, W16, V0, V1, V3, V4. */
static void
emit_fog(as *a, int src_f)
{
    if (src_f >= 0) {
        LD(AX, CTX, CTXF(values[0]) + 4 * src_f);
        emit_sat(a);
        a64_dup(a, true, 0, AX);
    } else {
        a64_ldst(a, A64_LDR_D, 0, CTX, CTXF(fog_f));
    }
    a64_ldst(a, A64_LDR_D, 1, CTX, CTXF(fog));
    A64_VUABD(a, V4H, 3, 2, 1);
    A64_VCMGT(a, V4H, 4, 1, 2);
    A64_VMUL(a, V4H, 3, 3, 0);
    a64_umull(a, 0, 3, K8081);
    a64_shrn(a, false, 3, 0, 16);
    a64_ushr4h(a, 3, 3, 7);
    A64_VEOR(a, 3, 3, 4);
    A64_VSUB(a, V4H, 3, 3, 4);
    A64_VADD(a, V4H, 3, 3, 1);
    A64_VAND(a, 3, 3, KRGB);
    A64_VAND(a, 2, 2, KALPHA);
    A64_VORR(a, 2, 2, 3);
}

/* dst_v = blend term for V2 (s) and V3 (d). Clobbers tmp_v, V6. */
static void
emit_blend_side(as *a, unsigned func, bool source, unsigned dst_v, unsigned tmp_v)
{
    unsigned own   = source ? 2 : 3;
    unsigned other = source ? 3 : 2;
    if (func == 6) {
        A64_VEOR(a, dst_v, dst_v, dst_v);
        return;
    }
    if (func == 7 || func == 12 || func == 13 || func == 14) {
        a64_vmov(a, dst_v, own);
        return;
    }
    if (func == 15) {
        a64_ldst(a, A64_LDR_D, tmp_v, KC, KF(k128));
        A64_VSUB(a, V4H, dst_v, own, tmp_v);
        return;
    }
    switch (func) {
        case 0:
            a64_vmov(a, dst_v, other);
            break;
        case 1:
            A64_VSUB(a, V4H, dst_v, K255, other);
            break;
        case 2:
            a64_dup_lane(a, dst_v, 2, 3);
            break;
        case 3:
            a64_dup_lane(a, tmp_v, 2, 3);
            A64_VSUB(a, V4H, dst_v, K255, tmp_v);
            break;
        case 4:
            a64_dup_lane(a, dst_v, 3, 3);
            break;
        case 5:
            a64_dup_lane(a, tmp_v, 3, 3);
            A64_VSUB(a, V4H, dst_v, K255, tmp_v);
            break;
        case 8:
        case 9:
            a64_dup_lane(a, tmp_v, 3, 3);
            A64_VSUB(a, V4H, dst_v, K255, tmp_v);
            a64_dup_lane(a, tmp_v, 2, 3);
            A64_VSMIN(a, V4H, dst_v, dst_v, tmp_v);
            A64_VAND(a, dst_v, dst_v, KRGB);
            a64_ldst(a, A64_LDR_D, tmp_v, KC, KF(alpha255));
            A64_VORR(a, dst_v, dst_v, tmp_v);
            if (func == 9)
                A64_VSUB(a, V4H, dst_v, K255, dst_v);
            break;
        case 10:
            a64_vmov(a, dst_v, own);
            break;
        case 11:
            A64_VSUB(a, V4H, dst_v, K255, own);
            break;
    }
    A64_VMUL(a, V4H, dst_v, dst_v, own);
    A64_VADD(a, V4H, dst_v, dst_v, K127);
    a64_umull(a, 6, dst_v, K8081);
    a64_shrn(a, false, dst_v, 6, 16);
    a64_ushr4h(a, dst_v, dst_v, 7);
}

/* Vectorized NEON blend in V2. Computes and saves ctx.dst_addr when reading dst. */
static void
emit_blend(as *a, const span_key *k, int src_y, bool same_table)
{
    if (k->dst_mode & DST_RAW_READ_DISABLE) {
        a64_ldst(a, A64_LDR_D, 3, CTX, CTXF(dst_c));
    } else {
        if (src_y >= 0) {
            LD(A64_IP0, CTX, CTXF(dst_stride));
            LD(A64_IP1, PE, PEF(dst_base));
            a64_madd(a, false, AX, YR, A64_IP0, A64_IP1);
        } else
            LD(AX, CTX, CTXF(dst_row));
        a64_alu(a, A64_ADD, false, AX, AX, XR, A64_LSL, k->dst_fmt == 12 ? 2 : 1);
        ST(AX, CTX, CTXF(dst_addr));
        if (k->dst_fmt == 12) {
            emit_load32(a, AX, AX);
            a64_fmov_from(a, 3, AX);
            a64_uxtl(a, false, 3, 3);
            a64_umov(a, AX, 3, 0);
            a64_umov(a, CX, 3, 2);
            a64_ins(a, 3, 0, CX);
            a64_ins(a, 3, 2, AX);
        } else {
            emit_load16(a, AX, AX);
            if (same_table) {
                a64_ldst_reg(a, A64_LDR_S, 3, TABLE, AX, A64_UXTW, true);
            } else {
                a64_ldst(a, A64_LDR_X, A64_IP0, CTX, CTXF(dst_table));
                a64_ldst_reg(a, A64_LDR_S, 3, A64_IP0, AX, A64_UXTW, true);
            }
            a64_uxtl(a, false, 3, 3);
        }
    }
    unsigned sf = (k->dst_mode >> 4) & 15, df = (k->dst_mode >> 8) & 15;
    emit_blend_side(a, sf, true, 0, 4);
    emit_blend_side(a, df, false, 1, 4);
    A64_VADD(a, V4H, 2, 0, 1);
    if (sf == 15 || df == 15) {
        A64_VEOR(a, 0, 0, 0);
        A64_VSMAX(a, V4H, 2, 2, 0);
    }
    A64_VSMIN(a, V4H, 2, 2, K255);
}

static bool
key_supported(const span_key *k)
{
    unsigned func = (k->src_mode >> 4) & 7;
    unsigned fmt  = k->src_mode & 15 ? k->src_mode & 15 : k->dst_fmt;
    if ((k->dst_fmt != 4 && k->dst_fmt != 5 && k->dst_fmt != 6 && k->dst_fmt != 12) || (k->dst_mode & 15) != 12)
        return false;
    if (k->dst_mode & (DST_RAW_YUV | DST_RAW_BGR))
        return false;
    if (func > 3)
        return false;
    if (func) {
        if (k->src_mode & SRC_BGR)
            return false;
        if (fmt != 2 && fmt != 3 && fmt != 4 && fmt != 5 && fmt != 6 && (fmt < 8 || fmt > 10) && fmt != 12)
            return false;
    }
    return true;
}

static void
emit_stride(as *a, unsigned shift, int32_t field)
{
    LD(AX, PE, PEF(stride));
    a64_ubfx(a, false, AX, AX, shift, 7);
    a64_mov_imm(a, true, A64_IP0, (uint64_t) (uintptr_t) stride_table);
    a64_ldst_reg(a, A64_LDRH, AX, A64_IP0, AX, A64_UXTW, true);
    ST(AX, CTX, field);
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

static void
emit_source(as *a, unsigned reg, int idx, int32_t field)
{
    if (idx >= 0)
        LD(reg, CTX, CTXF(values[0]) + 4 * idx);
    else
        LD(reg, PE, field);
}

static void
build_consts(span_consts *c, const span_key *k)
{
    static const uint8_t n565[]  = { 5, 6, 5, 0 }, pos565[] = { 11, 5, 0, 0 };
    static const uint8_t n4444[] = { 4, 4, 4, 4 }, pos4444[] = { 8, 4, 0, 12 };
    static const uint8_t n1555[] = { 5, 5, 5, 1 }, pos1555[] = { 10, 5, 0, 15 };
    const uint8_t       *ns  = k->dst_fmt == 4 ? n565 : k->dst_fmt == 5 ? n4444 : n1555;
    const uint8_t       *pos = k->dst_fmt == 4 ? pos565 : k->dst_fmt == 5 ? pos4444 : pos1555;
    memset(c, 0, sizeof(*c));
    for (unsigned ch = 0; ch < 4; ch++) {
        unsigned n       = k->dst_fmt == 12 ? 8 : ns[ch];
        unsigned p       = k->dst_fmt == 12 ? 0 : pos[ch];
        bool     on      = n != 0;
        c->k127[ch]      = 127;
        c->k128[ch]      = 128;
        c->k8081[ch]     = 0x8081;
        c->k255[ch]      = 255;
        c->down[ch]      = on ? -(int16_t) (8 - n) : -16;
        c->up[ch]        = on ? (int16_t) p : 0;
        c->rgb[ch]       = ch == 3 ? 0 : 0xffff;
        c->alpha[ch]     = ch == 3 ? 0xffff : 0;
        c->alpha255[ch]  = ch == 3 ? 255 : 0;
        for (unsigned i = 0; i < 16; i++)
            c->dither[i][ch] = on ? (dither_table[i] << 4) >> n : 0;
    }
}

static void
emit_dirty_pages(as *a, unsigned lo_reg, unsigned hi_reg)
{
    a64_shift_imm(a, A64_LSR, false, AX, lo_reg, 12);
    a64_shift_imm(a, A64_LSR, false, CX, hi_reg, 12);
    a64_rrr(a, A64_SUB, false, DX, CX, AX);
    a64_cmp_imm(a, false, DX, 16);
    uint8_t *few_pages = a64_bcond(a, A64_LS);
    a64_ldst(a, A64_LDR_W, AX, R8, offsetof(verite_core, global_gen));
    a64_alu_imm(a, A64_ADD, false, AX, AX, 1);
    a64_ldst(a, A64_STR_W, AX, R8, offsetof(verite_core, global_gen));
    uint8_t *dirty_done = a64_b(a);
    a64_land(few_pages, a->p);
    a64_alu_imm(a, A64_ADD, true, A64_IP1, R8, offsetof(verite_core, page_gen));
    uint8_t *page_loop = a->p;
    a64_alu_imm(a, A64_AND, false, DX, AX, 1023);
    a64_ldst_reg(a, A64_LDR_W, R9, A64_IP1, DX, A64_UXTW, true);
    a64_alu_imm(a, A64_ADD, false, R9, R9, 1);
    a64_ldst_reg(a, A64_STR_W, R9, A64_IP1, DX, A64_UXTW, true);
    a64_alu_imm(a, A64_ADD, false, AX, AX, 1);
    a64_cmp(a, false, AX, CX);
    a64_bcond_to(a, A64_LS, page_loop);
    a64_land(dirty_done, a->p);
}

static uint8_t *
compile(as *a, const span_key *k)
{
    unsigned    count  = k->op & 0xff;
    bool        single = !!(k->op & 0x100);
    unsigned    ctl    = k->drawctl;
    unsigned    i2     = (ctl >> 4) & 31;
    jit_pair    p2     = i2 <= 7 ? jit_pair3[i2] : jit_pair2[i2 - 8];
    jit_pair    p3 = jit_pair3[(ctl >> 9) & 15], p4 = jit_pair3[(ctl >> 13) & 7];
    pixel_param params[] = { jit_pair1[ctl & 15].pair.even, jit_pair1[ctl & 15].pair.odd, p2.even, p2.odd,
                             p3.even, p3.odd, p4.even, p4.odd, P_G, P_B };
    bool        shadow   = jit_pair1[ctl & 15].shadow;
    int         q_index  = -1;
    int         src_x = -1, src_y = -1, src_z = -1, src_u = -1, src_v = -1, src_f = -1;
    int         src_ch[4] = { -1, -1, -1, -1 };
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
    bool     dither     = !!(k->dst_mode & DST_RAW_DITHER) && k->dst_fmt != 12;
    bool     blend      = !!(k->dst_mode & DST_RAW_BLEND);
    bool     fog        = !!(k->dst_mode & DST_RAW_FOG);
    bool     raw_copy   = func == 1 && fmt == 4 && k->dst_fmt == 4 && (k->src_mode & SRC_NO_PAD)
        && !(k->src_mode & (SRC_FILTER | SRC_BGR)) && !blend && !fog;
    bool     preclip       = src_y < 0 && shadow;
    bool     unconditional = preclip && !depth_mode && !(k->dst_mode & DST_RAW_TREJECT)
        && !((k->src_mode & SRC_Z_SCISSOR) && src_z >= 0);
    bool     src_16        = fmt == 4 || fmt == 5 || fmt == 6;
    bool     dst_16        = k->dst_fmt == 4 || k->dst_fmt == 5 || k->dst_fmt == 6;
    bool     same_table    = func && !raw_copy && src_16 && dst_16 && fmt == k->dst_fmt
        && !!(k->src_mode & SRC_NO_PAD) == !!(k->dst_mode & DST_RAW_NO_PAD);
    bool     dst_computed  = blend && !(k->dst_mode & DST_RAW_READ_DISABLE);

    while ((uintptr_t) a->p & 15)
        *a->p++ = 0;
    span_consts *kc = (span_consts *) a->p;
    build_consts(kc, k);
    a->p += (sizeof(span_consts) + 15) & ~(size_t) 15;
    uint8_t *start = a->p;

    uint32_t frame = (sizeof(span_ctx) + 15) & ~15U;
    a64_pair(a, false, 3, LO, HI, A64_SP, -16);
    a64_alu_imm(a, A64_SUB, true, A64_SP, A64_SP, frame);
    a64_ldst(a, A64_STR_X, 0, CTX, CTXF(core));
    a64_ldst(a, A64_STR_X, 1, CTX, CTXF(site));
    a64_alu_imm(a, A64_ADD, true, PE, 0, offsetof(verite_core, pe));
    a64_ldst(a, A64_LDR_X, A64_IP1, 0, offsetof(verite_core, jit_mem));
    a64_ldst(a, A64_LDR_X, VRAM, A64_IP1, offsetof(verite_jit_memory, vram));
    a64_ldst(a, A64_LDR_W, MASK, A64_IP1, offsetof(verite_jit_memory, mask));
    ST(MASK, CTX, CTXF(byte_mask));
    a64_alu_imm(a, A64_AND, false, MASK, MASK, ~1U);
    a64_mov_imm(a, false, DREW, 0);
    a64_mov_imm(a, false, LO, UINT32_MAX);
    a64_mov_imm(a, false, HI, 0);
    if (zwrite) {
        ST(LO, CTX, CTXF(zlo));
        ST(HI, CTX, CTXF(zhi));
    }
    a64_adr(a, KC, kc);

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

    uint8_t *ff[8];
    unsigned nff = 0;
    LD(AX, CTX, CTXF(scissor[0]));
    LD(CX, CTX, CTXF(scissor[1]));
    a64_cmp(a, false, AX, CX);
    ff[nff++] = a64_bcond(a, A64_GE);
    LD(AX, CTX, CTXF(scissor[2]));
    LD(CX, CTX, CTXF(scissor[3]));
    a64_cmp(a, false, AX, CX);
    ff[nff++] = a64_bcond(a, A64_GE);
    if (src_y < 0) {
        LD(DX, PE, PEF(y));
        a64_shift_imm(a, A64_ASR, false, DX, DX, 16);
        LD(A64_IP0, CTX, CTXF(scissor[2]));
        a64_cmp(a, false, DX, A64_IP0);
        ff[nff++] = a64_bcond(a, A64_LT);
        LD(A64_IP0, CTX, CTXF(scissor[3]));
        a64_cmp(a, false, DX, A64_IP0);
        ff[nff++] = a64_bcond(a, A64_GE);
    }
    if ((k->src_mode & SRC_Z_SCISSOR) && src_z < 0) {
        a64_ldst(a, A64_LDRB, A64_IP0, PE, PEF(z_ovf));
        ff[nff++] = a64_cbz(a, true, false, A64_IP0);
    }
    if (depth_mode == 7)
        ff[nff++] = a64_b(a);
    if (raw_copy && (k->dst_mode & DST_RAW_TREJECT)) {
        LD(CX, PE, PEF(alpha_thres));
        a64_alu_imm(a, A64_AND, false, CX, CX, 255);
        a64_cmp_imm(a, false, CX, 255);
        ff[nff++] = a64_bcond(a, A64_EQ);
    }

    uint8_t *preclip_ff = NULL;
    if (preclip) {
        LD(R8, PE, PEF(x_shadow));
        a64_shift_imm(a, A64_ASR, false, R8, R8, 16);
        a64_ldst(a, A64_LDRB, AX, PE, PEF(x_dir));
        a64_mov_imm(a, false, R9, 0x10000);
        a64_cmp_imm(a, false, AX, 0);
        a64_cneg(a, false, R9, R9, A64_NE);
        LD(AX, PE, PEF(x_shadow));
        a64_madd(a, false, AX, R9, COUNT, AX);
        ST(AX, PE, PEF(x_shadow));

        LD(AX, CTX, CTXF(scissor[0]));
        LD(CX, CTX, CTXF(scissor[1]));
        a64_cmp_imm(a, false, R9, 0);
        uint8_t *backward = a64_bcond(a, A64_LT);

        /* Forward */
        a64_cmp(a, false, R8, AX);
        a64_csel(a, false, XR, AX, R8, A64_LT); /* x_start = max(x0, x_min) */
        a64_rrr(a, A64_ADD, false, DX, R8, COUNT);
        a64_cmp(a, false, DX, CX);
        a64_csel(a, false, DX, CX, DX, A64_GT); /* x_stop = min(x0 + n, x_end) */
        a64_rrr(a, A64_SUBS, false, AX, DX, XR); /* vis = stop - start */
        uint8_t *fwd_empty = a64_bcond(a, A64_LE);
        a64_rrr(a, A64_SUB, false, CX, XR, R8);  /* skip = start - x0 */
        a64_mov_imm(a, false, R8, 1);
        uint8_t *clip_done = a64_b(a);

        a64_land(backward, a->p);
        a64_alu_imm(a, A64_SUB, false, CX, CX, 1);
        a64_cmp(a, false, R8, CX);
        a64_csel(a, false, XR, R8, CX, A64_LT); /* hi_x = min(x0, x_end - 1) */
        a64_rrr(a, A64_SUB, false, DX, R8, COUNT);
        a64_alu_imm(a, A64_ADD, false, DX, DX, 1);
        a64_cmp(a, false, DX, AX);
        a64_csel(a, false, DX, AX, DX, A64_LT); /* lo_x = max(x0 - n + 1, x_min) */
        a64_rrr(a, A64_SUB, false, AX, XR, DX);
        a64_alu_imm(a, A64_ADD, false, AX, AX, 1);
        a64_cmp_imm(a, false, AX, 0);
        uint8_t *bwd_empty = a64_bcond(a, A64_LE);
        a64_rrr(a, A64_SUB, false, CX, R8, XR);  /* skip = x0 - hi_x */
        a64_mov_imm(a, false, R8, UINT32_MAX);   /* -1 */

        a64_land(clip_done, a->p);
        ST(R8, CTX, CTXF(x_dir_step));
        if (unconditional)
            ST(XR, CTX, CTXF(x_first));
        a64_rrr(a, A64_SUB, false, DX, COUNT, AX);
        a64_rrr(a, A64_SUB, false, DX, DX, CX);
        ST(DX, CTX, CTXF(tail));
        MOV(COUNT, AX);
        uint8_t *no_skip = a64_cbz(a, false, false, CX);
        if (q_index >= 0) {
            a64_alu_imm(a, A64_SUB, false, AX, CX, 1);
            LD(DX, CTX, CTXF(deltas[0]) + 4 * q_index);
            LD(R9, CTX, CTXF(values[0]) + 4 * q_index);
            a64_madd(a, false, AX, AX, DX, R9);
            ST(AX, PE, PEF(q));
        }
        for (unsigned i = 0; i < count; i++) {
            if (params[i] == P_NONE && !single)
                continue;
            LD(AX, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
            LD(DX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
            a64_madd(a, false, AX, AX, CX, DX);
            ST(AX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
        }
        a64_land(no_skip, a->p);
        uint8_t *clip_ok = a64_b(a);

        a64_land(fwd_empty, a->p);
        a64_land(bwd_empty, a->p);
        preclip_ff = a64_b(a);
        a64_land(clip_ok, a->p);
    } else if (shadow) {
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
        if (!single) {
            ST(A64_ZR, CTX, CTXF(prev_u));
            ST(A64_ZR, CTX, CTXF(prev_v));
        }
    }

    if (src_y < 0) {
        LD(DX, PE, PEF(y));
        a64_shift_imm(a, A64_ASR, false, DX, DX, 16);
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
            LD(AX, PE, PEF(y));
            a64_shift_imm(a, A64_ASR, false, AX, AX, 16);
            a64_rrr(a, A64_ADD, false, AX, AX, DX);
            a64_ubfx(a, false, AX, AX, 0, 2);
            a64_alu_imm(a, A64_ADD, true, CX, KC, KF(dither));
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
    if (fog && !raw_copy) {
        static const unsigned byte[] = { 16, 8, 0 };
        LD(AX, PE, PEF(fog_color));
        a64_mov_imm(a, true, CX, 0);
        for (unsigned lane = 0; lane < 3; lane++) {
            a64_ubfx(a, true, DX, AX, byte[lane], 8);
            a64_alu(a, A64_ORR, true, CX, CX, DX, A64_LSL, 16 * lane);
        }
        a64_ldst(a, A64_STR_X, CX, CTX, CTXF(fog));
        if (src_f < 0) {
            LD(AX, PE, PEF(f));
            emit_sat(a);
            a64_dup(a, true, 0, AX);
            a64_ldst(a, A64_STR_D, 0, CTX, CTXF(fog_f));
        }
    }
    if (blend && (k->dst_mode & DST_RAW_READ_DISABLE) && !raw_copy) {
        static const unsigned byte[] = { 16, 8, 0, 24 };
        LD(AX, PE, PEF(dst_color));
        a64_mov_imm(a, true, CX, 0);
        for (unsigned lane = 0; lane < 4; lane++) {
            a64_ubfx(a, true, DX, AX, byte[lane], 8);
            a64_alu(a, A64_ORR, true, CX, CX, DX, A64_LSL, 16 * lane);
        }
        a64_ldst(a, A64_STR_X, CX, CTX, CTXF(dst_c));
    }
    if (dst_computed && !same_table && dst_16 && !raw_copy) {
        a64_mov_imm(a, true, AX, (uint64_t) (uintptr_t) texel_table(k->dst_fmt, !!(k->dst_mode & DST_RAW_NO_PAD)));
        a64_ldst(a, A64_STR_X, AX, CTX, CTXF(dst_table));
    }
    if (func && !raw_copy) {
        if (src_16)
            a64_mov_imm(a, true, TABLE, (uint64_t) (uintptr_t) texel_table(fmt, !!(k->src_mode & SRC_NO_PAD)));
        else if (fmt >= 8 && fmt <= 10) {
            const uint32_t *ptab = texel_table(fmt - 4, !!(k->src_mode & SRC_NO_PAD));
            a64_mov_imm(a, true, DX, (uint64_t) (uintptr_t) ptab);
            for (unsigned i = 0; i < 8; i++) {
                LD(AX, PE, PEF(tex4pal[0]) + 4 * (int32_t) i);
                a64_shift_imm(a, A64_LSR, false, CX, AX, 16);
                a64_ldst_reg(a, A64_LDR_W, CX, DX, CX, A64_UXTW, true);
                ST(CX, CTX, CTXF(pal[0]) + 4 * (int32_t) (2 * i));
                a64_alu_imm(a, A64_AND, false, AX, AX, 0xffff);
                a64_ldst_reg(a, A64_LDR_W, AX, DX, AX, A64_UXTW, true);
                ST(AX, CTX, CTXF(pal[0]) + 4 * (int32_t) (2 * i + 1));
            }
        }
        if (!single) {
            LD(AX, PE, PEF(u_mask));
            a64_alu_imm(a, A64_ADD, false, AX, AX, 1);
            if (fmt >= 8 && fmt <= 10)
                a64_shift_imm(a, A64_LSR, false, AX, AX, 1);
            else if (fmt == 12)
                a64_shift_imm(a, A64_LSL, false, AX, AX, 2);
            else if (fmt != 2 && fmt != 3)
                a64_rrr(a, A64_ADD, false, AX, AX, AX);
            a64_alu_imm(a, A64_ADD, false, AX, AX, 63);
            a64_alu_imm(a, A64_AND, false, AX, AX, ~63U);
            LD(CX, PE, PEF(v_mask));
            a64_alu_imm(a, A64_ADD, false, CX, CX, 1);
            a64_mul(a, false, AX, AX, CX);
            a64_cmp_imm(a, false, AX, 16384);
            a64_cset(a, false, CX, A64_HI);
            ST(CX, CTX, CTXF(prefetch));
        }
    }
    {
        static const int32_t konst[] = { KF(k127), KF(k8081), KF(k255), KF(down), KF(up), KF(rgb), KF(alpha) };
        for (unsigned i = 0; i < 7; i++)
            a64_ldst(a, A64_LDR_D, K127 + i, KC, konst[i]);
    }

    uint8_t *loop = a->p;

    uint8_t *skips[16];
    unsigned nskip = 0;
    if (!preclip) {
        emit_source(a, XR, src_x, shadow ? PEF(x_shadow) : PEF(x));
        a64_shift_imm(a, A64_ASR, false, XR, XR, 16);
    }
    if (src_y >= 0) {
        emit_source(a, YR, src_y, PEF(y));
        a64_shift_imm(a, A64_ASR, false, YR, YR, 16);
    }
    if (!preclip) {
        for (unsigned axis = 0; axis < (src_y >= 0 ? 2U : 1U); axis++) {
            unsigned r = axis ? YR : XR;
            LD(A64_IP0, CTX, CTXF(scissor[0]) + 8 * (int32_t) axis);
            a64_cmp(a, false, r, A64_IP0);
            skips[nskip++] = a64_bcond(a, A64_LT);
            LD(A64_IP0, CTX, CTXF(scissor[1]) + 8 * (int32_t) axis);
            a64_cmp(a, false, r, A64_IP0);
            skips[nskip++] = a64_bcond(a, A64_GE);
        }
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
        if (depth_mode) {
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
        if (k->src_mode & SRC_CHROMA) {
            a64_mov_imm(a, false, A64_IP1, 0x10001);
            a64_mul(a, false, CX, R8, A64_IP1);
            LD(A64_IP1, PE, PEF(chroma_color));
            a64_rrr(a, A64_EOR, false, CX, CX, A64_IP1);
            LD(A64_IP1, PE, PEF(chroma_mask));
            a64_rrr(a, A64_ANDS, false, CX, CX, A64_IP1);
            uint8_t *opaque = a64_bcond(a, A64_NE);
            if (k->dst_mode & DST_RAW_TREJECT)
                skips[nskip++] = a64_b(a);
            else
                a64_mov_imm(a, false, R8, k->src_mode & SRC_CHROMA_BLACK_YUV ? 0x8090 : 0);
            a64_land(opaque, a->p);
        }
    } else {
        if (func)
            emit_sample(a, fmt, k->src_mode, single);
        if (func != 1) {
            a64_ldst(a, A64_LDR_D, 2, CTX, CTXF(fg));
            for (unsigned ch = 0; ch < 4; ch++) {
                int  idx   = src_ch[ch];
                bool first = idx >= 0;
                for (unsigned prev = 0; prev < ch; prev++)
                    first = first && src_ch[prev] != idx;
                if (!first)
                    continue;
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
            A64_VMUL(a, V4H, 2, 2, 3);
            A64_VADD(a, V4H, 2, 2, K127);
            a64_umull(a, 0, 2, K8081);
            a64_shrn(a, false, 2, 0, 16);
            a64_ushr4h(a, 2, 2, 7);
        } else if (func == 2) {
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

        if (fog)
            emit_fog(a, src_f);
        if (blend)
            emit_blend(a, k, src_y, same_table);

        if (k->dst_fmt == 12) {
            a64_umov(a, AX, 2, 0);
            a64_umov(a, CX, 2, 2);
            a64_ins(a, 2, 0, CX);
            a64_ins(a, 2, 2, AX);
            /* Pack 4 bytes into W5 (R8). */
            a64_umov(a, R8, 2, 0);
            a64_umov(a, AX, 2, 1);
            a64_alu(a, A64_ORR, false, R8, R8, AX, A64_LSL, 8);
            a64_umov(a, AX, 2, 2);
            a64_alu(a, A64_ORR, false, R8, R8, AX, A64_LSL, 16);
            a64_umov(a, AX, 2, 3);
            a64_alu(a, A64_ORR, false, R8, R8, AX, A64_LSL, 24);
        } else {
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
    }

    if (dst_computed) {
        LD(AX, CTX, CTXF(dst_addr));
    } else {
        if (src_y >= 0) {
            LD(A64_IP0, CTX, CTXF(dst_stride));
            LD(A64_IP1, PE, PEF(dst_base));
            a64_madd(a, false, AX, YR, A64_IP0, A64_IP1);
        } else
            LD(AX, CTX, CTXF(dst_row));
        a64_alu(a, A64_ADD, false, AX, AX, XR, A64_LSL, k->dst_fmt == 12 ? 2 : 1);
    }
    if (k->dst_fmt == 12)
        emit_store32(a, R8, AX, !unconditional);
    else
        emit_store16(a, R8, AX, !unconditional, false);
    if (zwrite) {
        LD(AX, CTX, CTXF(z_addr));
        LD(CX, CTX, CTXF(z));
        emit_store16(a, CX, AX, !unconditional, true);
    }

    for (unsigned i = 0; i < nskip; i++)
        a64_land(skips[i], a->p);
    if (q_index >= 0) {
        LD(AX, CTX, CTXF(values[0]) + 4 * q_index);
        ST(AX, PE, PEF(q));
    }
    if (!preclip && shadow) {
        LD(AX, CTX, CTXF(x_step));
        LD(CX, PE, PEF(x_shadow));
        a64_rrr(a, A64_ADD, false, CX, CX, AX);
        ST(CX, PE, PEF(x_shadow));
    }
    for (unsigned i = 0; i < count; i++) {
        if (params[i] == P_NONE && !single)
            continue;
        LD(AX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
        LD(CX, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
        a64_rrr(a, A64_ADD, false, AX, AX, CX);
        ST(AX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
    }
    if (!single) {
        if (preclip) {
            LD(AX, CTX, CTXF(x_dir_step));
            a64_rrr(a, A64_ADD, false, XR, XR, AX);
        }
        a64_alu_imm(a, A64_SUBS, false, COUNT, COUNT, 1);
        a64_bcond_to(a, A64_NE, loop);
    }

    if (unconditional) {
        if (!single) {
            LD(AX, CTX, CTXF(x_dir_step));
            a64_rrr(a, A64_SUB, false, XR, XR, AX);
        }
        LD(AX, CTX, CTXF(x_first));
        a64_cmp(a, false, AX, XR);
        a64_csel(a, false, CX, AX, XR, A64_LT); /* x_lo */
        a64_csel(a, false, DX, XR, AX, A64_LT); /* x_hi */
        LD(A64_IP0, CTX, CTXF(dst_row));
        unsigned shift_b = k->dst_fmt == 12 ? 2 : 1;
        a64_alu(a, A64_ADD, false, LO, A64_IP0, CX, A64_LSL, shift_b);
        a64_alu(a, A64_ADD, false, HI, A64_IP0, DX, A64_LSL, shift_b);
        a64_rrr(a, A64_AND, false, LO, LO, MASK);
        a64_rrr(a, A64_AND, false, HI, HI, MASK);
        if (k->dst_fmt == 12) {
            a64_alu_imm(a, A64_AND, false, LO, LO, ~3U);
            a64_alu_imm(a, A64_AND, false, HI, HI, ~3U);
            a64_alu_imm(a, A64_ADD, false, HI, HI, 2);
        }
        a64_cmp(a, false, LO, HI);
        uint8_t *no_wrap = a64_bcond(a, A64_LS);
        a64_mov_imm(a, false, LO, 0);
        a64_alu_imm(a, A64_ORR, false, HI, MASK, 1);
        a64_land(no_wrap, a->p);
        if (zwrite) {
            LD(A64_IP0, CTX, CTXF(z_row));
            a64_alu(a, A64_ADD, false, AX, A64_IP0, CX, A64_LSL, 1);
            a64_alu(a, A64_ADD, false, DX, A64_IP0, DX, A64_LSL, 1);
            a64_rrr(a, A64_AND, false, AX, AX, MASK);
            a64_rrr(a, A64_AND, false, DX, DX, MASK);
            a64_cmp(a, false, AX, DX);
            uint8_t *z_ok = a64_bcond(a, A64_LS);
            a64_mov_imm(a, false, AX, 0);
            a64_alu_imm(a, A64_ORR, false, DX, MASK, 1);
            a64_land(z_ok, a->p);
            ST(AX, CTX, CTXF(zlo));
            ST(DX, CTX, CTXF(zhi));
        }
        a64_mov_imm(a, false, DREW, 1);
    }

    if (preclip) {
        LD(CX, CTX, CTXF(tail));
        uint8_t *no_tail = a64_cbz(a, false, false, CX);
        for (unsigned i = 0; i < count; i++) {
            if (params[i] == P_NONE && !single)
                continue;
            LD(AX, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
            LD(DX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
            a64_madd(a, false, AX, AX, CX, DX);
            ST(AX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
        }
        if (q_index >= 0) {
            LD(AX, CTX, CTXF(values[0]) + 4 * q_index);
            LD(CX, CTX, CTXF(deltas[0]) + 4 * q_index);
            a64_rrr(a, A64_SUB, false, AX, AX, CX);
            ST(AX, PE, PEF(q));
        }
        a64_land(no_tail, a->p);
    }

    uint8_t *set_state = a64_b(a);

    for (unsigned i = 0; i < nff; i++)
        a64_land(ff[i], a->p);
    if (shadow) {
        a64_ldst(a, A64_LDRB, AX, PE, PEF(x_dir));
        a64_mov_imm(a, false, R9, 0x10000);
        a64_cmp_imm(a, false, AX, 0);
        a64_cneg(a, false, R9, R9, A64_NE);
        LD(AX, PE, PEF(x_shadow));
        a64_madd(a, false, AX, R9, COUNT, AX);
        ST(AX, PE, PEF(x_shadow));
    }
    if (preclip_ff)
        a64_land(preclip_ff, a->p);
    for (unsigned i = 0; i < count; i++) {
        if (params[i] == P_NONE && !single)
            continue;
        LD(AX, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
        LD(DX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
        a64_madd(a, false, AX, AX, COUNT, DX);
        ST(AX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
    }
    if (q_index >= 0) {
        LD(AX, CTX, CTXF(values[0]) + 4 * q_index);
        LD(CX, CTX, CTXF(deltas[0]) + 4 * q_index);
        a64_rrr(a, A64_SUB, false, AX, AX, CX);
        ST(AX, PE, PEF(q));
    }

    a64_land(set_state, a->p);
    for (unsigned i = 0; i < count; i++) {
        if (params[i] == P_NONE || params[i] == P_Q)
            continue;
        LD(AX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
        LD(CX, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
        a64_rrr(a, A64_SUB, false, AX, AX, CX);
        emit_set(a, params[i], k->src_mode);
    }
    if (!unconditional) {
        a64_cmp_imm(a, false, LO, UINT32_MAX);
        a64_cset(a, false, DREW, A64_NE);
    }

    if (single) {
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
    a64_ldst(a, A64_LDR_X, R8, CTX, CTXF(core));
    a64_ldst(a, A64_STRB, AX, R8, offsetof(verite_core, drew));
    /* Mark dirty VRAM pages in core->page_gen / global_gen separately for color and Z. */
    emit_dirty_pages(a, LO, HI);
    if (zwrite) {
        LD(AX, CTX, CTXF(zlo));
        LD(CX, CTX, CTXF(zhi));
        a64_cmp(a, false, AX, CX);
        uint8_t *no_z = a64_bcond(a, A64_HI);
        a64_cmp(a, false, AX, LO);
        a64_csel(a, false, LO, AX, LO, A64_LO);
        a64_cmp(a, false, CX, HI);
        a64_csel(a, false, HI, CX, HI, A64_HI);
        emit_dirty_pages(a, AX, CX);
        a64_land(no_z, a->p);
    }
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

static uint32_t
span_hash(const span_key *k)
{
    uint32_t h = k->op * 0x9e3779b1U ^ k->drawctl * 0x85ebca6bU ^ k->src_mode * 0xc2b2ae35U
               ^ k->dst_mode * 0x27d4eb2fU ^ k->dst_fmt * 0x165667b1U;
    h ^= h >> 16;
    h *= 0x85ebca6bU;
    h ^= h >> 13;
    return h;
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
    uint32_t idx = span_hash(key) & (SPAN_CACHE - 1);
    for (unsigned probe = 0; probe < SPAN_CACHE; probe++) {
        span_entry *e = &jit->entries[(idx + probe) & (SPAN_CACHE - 1)];
        if (!e->used)
            break;
        if (!memcmp(&e->key, key, sizeof(*key)))
            return e->fn;
    }
    if (jit->used + SPAN_MAX_CODE > SPAN_ARENA || jit->count >= (SPAN_CACHE * 3) / 4) {
        memset(jit->entries, 0, sizeof(jit->entries));
        jit->used  = 0;
        jit->count = 0;
        risc_jit_reset_sites(c);
    }
    unsigned fmt = key->src_mode & 15 ? key->src_mode & 15 : key->dst_fmt;
    if ((key->src_mode >> 4) & 7) {
        if (fmt == 4 || fmt == 5 || fmt == 6)
            texel_table(fmt, !!(key->src_mode & SRC_NO_PAD));
        else if (fmt >= 8 && fmt <= 10)
            texel_table(fmt - 4, !!(key->src_mode & SRC_NO_PAD));
    }
    if ((key->dst_mode & DST_RAW_BLEND) && !(key->dst_mode & DST_RAW_READ_DISABLE)
        && (key->dst_fmt == 4 || key->dst_fmt == 5 || key->dst_fmt == 6))
        texel_table(key->dst_fmt, !!(key->dst_mode & DST_RAW_NO_PAD));

    uint8_t *base = jit->arena + jit->used;
    as       a    = { base };
    a64_write_begin();
    uint8_t *code = compile(&a, key);
    size_t   size = (size_t) (a.p - base);
    a64_write_end(base, size);
    if (size > SPAN_MAX_CODE)
        abort();
    jit->used += (size + 15) & ~(size_t) 15;
    idx = span_hash(key) & (SPAN_CACHE - 1);
    while (jit->entries[idx].used)
        idx = (idx + 1) & (SPAN_CACHE - 1);
    jit->entries[idx] = (span_entry) { *key, (span_fn) (void *) code, true };
    jit->count++;
    return jit->entries[idx].fn;
}

static span_fn
state_span(verite_core *c, uint8_t op)
{
    pixel_engine *pe   = &c->pe;
    uint32_t      fmt  = pe->dst_fmt & 15;
    uint32_t      mask = fmt == 12 ? UINT32_MAX : 0xffffU;
    if (c->jit_disabled || c->jit_unavailable || (pe->pat_mode & 0x10000) || (pe->pmask & mask) != mask)
        return NULL;
    span_key key = {
        .op       = draw_count(op) | (op & 1 ? 0 : 0x100),
        .drawctl  = pe->drawctl & 0xffff,
        .src_mode = pe->src_mode & 0x1ffff,
        .dst_mode = pe->dst_mode & 0x03ff7fff,
        .dst_fmt  = fmt,
    };
    return key_supported(&key) ? lookup(c, &key) : NULL;
}

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
