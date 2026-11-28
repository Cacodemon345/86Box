/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Native parameter-draw spans for the Vérité pixel engine.

   A repeated parameter draw (rdrawp) runs one pixel_full() per pixel with a
   render state that stays fixed for the whole span. Like the Voodoo
   recompiler, this compiles that state into a native loop and caches it. The
   key holds everything the code depends on; values that may change between
   spans (bases, scissors, colours, masks) are read from the engine at run time.
   States outside the supported subset return to the C path. */

#include "internal.h"

#ifdef VERITE_JIT_X64

#    include <stdlib.h>
#    ifdef _WIN32
#        include <windows.h>
#    else
#        include <sys/mman.h>
#    endif

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

/* The span's stack frame, addressed through R15 and 16-byte aligned. */
typedef struct {
    uint32_t          values[12], deltas[12];
    uint32_t          pal[16];
    uint16_t          fg[4], fog[4], fog_f[4], dst_c[4];
    verite_core      *core;
    verite_draw_site *site;
    int32_t           cnt;
    uint32_t          x_step, dst_stride, z_stride, tex_stride;
    uint32_t          z_addr, z, dst_addr, byte_mask;
    uint32_t          lo, hi, zlo, zhi, tail, x_dir_step;
    int32_t           x_first;
    uint32_t          prev_u, prev_v; /* the last pixel's texel coordinates */
    int32_t           scissor[4];     /* x lo, x hi, y lo, y hi */
    uint32_t          dst_row, z_row; /* row addresses when Y is no parameter */
    uint32_t          dither_x, dither_y, thres, zero;
    uint32_t          prefetch;   /* whether the texture outgrows the L1 cache */
    const uint16_t   *dither_row; /* the dither vectors for a fixed Y */
    const uint32_t   *dst_table;
} span_ctx;
typedef char span_ctx_aligned[offsetof(span_ctx, deltas) % 16 == 0 && offsetof(span_ctx, fg) % 8 == 0 ? 1 : -1];

/* Per-state SSE constants, placed before the span's code and addressed
   RIP-relative. Vectors hold words R, G, B, A twice. */
typedef struct {
    uint16_t k127[8], k128[8], k8081[8], k255[8];
    uint16_t shift[8]; /* x >> (8 - n) as a pmulhuw multiplier, n the field width */
    uint16_t pos[8];   /* the field's position as a pmullw multiplier */
    uint16_t ones[8], rgb[8], alpha[8], alpha255[8];
    uint32_t bias[4], round[4];
    uint16_t dither[16][8]; /* (dither_table[i] << 4) >> n */
    uint32_t black, pad;
    double   two24;
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

/* ---- Assembler ------------------------------------------------------------ */

enum { RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15 };
/* MASK is the VRAM mask with bit 0 clear; TABLE is the texel table in the
   loop and becomes the drew flag after it. */
enum { CTX = R15, VRAM = R14, MASK = R13, PE = R12, COUNT = RBX, TABLE = RBP, DREW = RBP };
enum { NO_INDEX = -1 };
enum { CC_B = 2, CC_AE, CC_E, CC_NE, CC_BE, CC_A, CC_S, CC_NS, CC_L = 12, CC_GE, CC_LE, CC_G };

typedef struct {
    uint8_t *p;
} as;

static void
b8(as *a, unsigned v)
{
    *a->p++ = (uint8_t) v;
}

static void
d32(as *a, uint32_t v)
{
    memcpy(a->p, &v, 4);
    a->p += 4;
}

static void
opcode(as *a, unsigned op)
{
    if (op > 0xff)
        b8(a, op >> 8);
    b8(a, op);
}

static void
rex(as *a, bool w, unsigned reg, unsigned index, unsigned base, bool force)
{
    unsigned p = 0x40 | (w << 3) | ((reg >> 3) << 2) | ((index >> 3) << 1) | (base >> 3);
    if (p != 0x40 || force)
        b8(a, p);
}

/* op reg, rm (both registers). */
static void
rr(as *a, unsigned op, unsigned reg, unsigned rm, bool w)
{
    rex(a, w, reg, 0, rm, false);
    opcode(a, op);
    b8(a, 0xc0 | ((reg & 7) << 3) | (rm & 7));
}

/* op reg, [base + (index << scale) + disp32]. byte forces a REX so SIL/DIL
   can be used. */
static void
rmx(as *a, unsigned op, unsigned reg, unsigned base, int index, unsigned scale, int32_t disp, bool w, bool byte)
{
    unsigned ix = index < 0 ? 4 : (unsigned) index;
    rex(a, w, reg, index < 0 ? 0 : ix, base, byte && reg >= 4);
    opcode(a, op);
    if (index < 0 && (base & 7) != 4) {
        b8(a, 0x80 | ((reg & 7) << 3) | (base & 7));
    } else {
        b8(a, 0x80 | ((reg & 7) << 3) | 4);
        b8(a, (scale << 6) | ((ix & 7) << 3) | (base & 7));
    }
    d32(a, (uint32_t) disp);
}

static void
rm(as *a, unsigned op, unsigned reg, unsigned base, int index, int32_t disp, bool w, bool byte)
{
    rmx(a, op, reg, base, index, 0, disp, w, byte);
}

/* SSE2, XMM0-XMM5 only: pfx op xmm, xmm/r32. */
static void
sse(as *a, unsigned pfx, unsigned op, unsigned reg, unsigned rm_)
{
    b8(a, pfx);
    rr(a, op, reg, rm_, false);
}

/* pfx op xmm, [base + (index << scale) + disp32] */
static void
sse_m(as *a, unsigned pfx, unsigned op, unsigned reg, unsigned base, int index, unsigned scale, int32_t disp)
{
    b8(a, pfx);
    rmx(a, op, reg, base, index, scale, disp, false, false);
}

/* pfx op xmm, [rip + target], for an instruction without an immediate. */
static void
sse_rip(as *a, unsigned pfx, unsigned op, unsigned reg, const void *target)
{
    b8(a, pfx);
    opcode(a, op);
    b8(a, ((reg & 7) << 3) | 5);
    d32(a, (uint32_t) ((const uint8_t *) target - (a->p + 4)));
}

/* pfx op xmm, xmm/r32, imm8 (shifts take the extension as reg). */
static void
sse_i(as *a, unsigned pfx, unsigned op, unsigned reg, unsigned rm_, unsigned imm)
{
    sse(a, pfx, op, reg, rm_);
    b8(a, imm);
}

enum {
    P66 = 0x66, PF2 = 0xf2, PF3 = 0xf3,
    PUNPCKLBW = 0x0f60, PUNPCKLWD = 0x0f61, PCMPGTW = 0x0f65, PACKUSWB = 0x0f67, PACKSSDW = 0x0f6b,
    MOVD_TO = 0x0f6e, MOVDQA = 0x0f6f, PSHUFD = 0x0f70, PSHIFTW = 0x0f71, PSHIFTD = 0x0f72,
    MOVD_FROM = 0x0f7e, MOVDQA_ST = 0x0f7f, PINSRW = 0x0fc4, PEXTRW = 0x0fc5, MOVQ_ST = 0x0fd6,
    PMULLW = 0x0fd5, PSUBUSW = 0x0fd9, PAND = 0x0fdb, PMULHUW = 0x0fe4, PMINSW = 0x0fea, POR = 0x0feb,
    PMAXSW = 0x0fee, PXOR = 0x0fef, PMADDWD = 0x0ff5, PSUBW = 0x0ff9, PADDW = 0x0ffd, PADDD = 0x0ffe,
    SH_SRL = 2, SH_SRA = 4, SH_SLL = 6,
};

static void
mov_ri(as *a, unsigned reg, uint32_t imm)
{
    rex(a, false, 0, 0, reg, false);
    b8(a, 0xb8 | (reg & 7));
    d32(a, imm);
}

static void
mov_ri64(as *a, unsigned reg, uint64_t imm)
{
    rex(a, true, 0, 0, reg, false);
    b8(a, 0xb8 | (reg & 7));
    memcpy(a->p, &imm, 8);
    a->p += 8;
}

/* Group-1 ALU with an immediate: 0 add, 1 or, 4 and, 5 sub, 6 xor, 7 cmp. */
static void
alu_ri(as *a, unsigned ext, unsigned reg, uint32_t imm)
{
    int32_t s = (int32_t) imm;
    if (s >= -128 && s <= 127) {
        rr(a, 0x83, ext, reg, false);
        b8(a, imm);
    } else {
        rr(a, 0x81, ext, reg, false);
        d32(a, imm);
    }
}

/* Shifts: 4 shl, 5 shr, 7 sar. */
static void
shift_ri(as *a, unsigned ext, unsigned reg, unsigned n, bool w)
{
    if (n) {
        rr(a, 0xc1, ext, reg, w);
        b8(a, n);
    }
}

static uint8_t *
jcc(as *a, unsigned cc)
{
    b8(a, 0x0f);
    b8(a, 0x80 | cc);
    uint8_t *at = a->p;
    d32(a, 0);
    return at;
}

static uint8_t *
jmp(as *a)
{
    b8(a, 0xe9);
    uint8_t *at = a->p;
    d32(a, 0);
    return at;
}

static void
land(uint8_t *at, const uint8_t *target)
{
    uint32_t rel = (uint32_t) (target - (at + 4));
    memcpy(at, &rel, 4);
}

#    define CTXF(f)    ((int32_t) offsetof(span_ctx, f))
#    define PEF(f)     ((int32_t) offsetof(pixel_engine, f))
#    define LD(r, b, o)  rm(a, 0x8b, r, b, NO_INDEX, o, false, false)
#    define ST(r, b, o)  rm(a, 0x89, r, b, NO_INDEX, o, false, false)
#    define MOV(d, s)    rr(a, 0x89, s, d, false)

/* dst = 8-bit VRAM byte at the address in addr (clobbers addr). */
static void
emit_load8(as *a, unsigned dst, unsigned addr)
{
    rm(a, 0x23, addr, CTX, NO_INDEX, CTXF(byte_mask), false, false);
    alu_ri(a, 6, addr, 3);
    rm(a, 0x0fb6, dst, VRAM, (int) addr, 0, false, false);
}

/* dst = 16-bit VRAM word at the address in addr (clobbers addr). */
static void
emit_load16(as *a, unsigned dst, unsigned addr)
{
    rr(a, 0x21, MASK, addr, false); /* and addr, mask */
    alu_ri(a, 6, addr, 2);
    rm(a, 0x0fb7, dst, VRAM, (int) addr, 0, false, false);
}

/* dst = 32-bit VRAM dword at the address in addr (clobbers addr). */
static void
emit_load32(as *a, unsigned dst, unsigned addr)
{
    rr(a, 0x21, MASK, addr, false);
    alu_ri(a, 4, addr, ~3U);
    rm(a, 0x8b, dst, VRAM, (int) addr, 0, false, false);
}

/* Store the low 16 bits of val at the address in addr (clobbers addr, RDX),
   branchlessly widening [ctx.lo, ctx.hi] (or [ctx.zlo, ctx.zhi]) when track_bounds is true. */
static void
emit_store16(as *a, unsigned val, unsigned addr, bool track_bounds, bool is_z)
{
    rr(a, 0x21, MASK, addr, false);
    if (track_bounds) {
        int32_t lo_off = is_z ? CTXF(zlo) : CTXF(lo);
        int32_t hi_off = is_z ? CTXF(zhi) : CTXF(hi);
        LD(RDX, CTX, lo_off);
        rr(a, 0x39, RDX, addr, false);   /* cmp addr, edx */
        rr(a, 0x0f42, RDX, addr, false); /* cmovb edx, addr */
        ST(RDX, CTX, lo_off);
        LD(RDX, CTX, hi_off);
        rr(a, 0x39, RDX, addr, false);   /* cmp addr, edx */
        rr(a, 0x0f47, RDX, addr, false); /* cmova edx, addr */
        ST(RDX, CTX, hi_off);
    }
    alu_ri(a, 6, addr, 2);
    b8(a, 0x66);
    rm(a, 0x89, val, VRAM, (int) addr, 0, false, false);
}

/* Store 32-bit val at the address in addr (clobbers addr, RCX, RDX). */
static void
emit_store32(as *a, unsigned val, unsigned addr, bool track_bounds)
{
    rr(a, 0x21, MASK, addr, false);
    alu_ri(a, 4, addr, ~3U);
    if (track_bounds) {
        LD(RDX, CTX, CTXF(lo));
        rr(a, 0x39, RDX, addr, false);
        rr(a, 0x0f42, RDX, addr, false);
        ST(RDX, CTX, CTXF(lo));
        rm(a, 0x8d, RDX, addr, NO_INDEX, 2, false, false);
        LD(RCX, CTX, CTXF(hi));
        rr(a, 0x39, RCX, RDX, false);
        rr(a, 0x0f47, RCX, RDX, false);
        ST(RCX, CTX, CTXF(hi));
    }
    rm(a, 0x89, val, VRAM, (int) addr, 0, false, false);
}

/* EAX = clamp(EAX >> 16, 0, 255). Clobbers ECX. */
static void
emit_sat(as *a)
{
    shift_ri(a, 7, RAX, 16, false);
    mov_ri(a, RCX, 255);
    rr(a, 0x39, RCX, RAX, false);   /* cmp eax, ecx */
    rr(a, 0x0f4f, RAX, RCX, false); /* cmovg eax, ecx */
    rr(a, 0x31, RCX, RCX, false);
    rr(a, 0x85, RAX, RAX, false);
    rr(a, 0x0f4c, RAX, RCX, false); /* cmovl eax, ecx */
}

/* fg = (fg & ~mask) | (EAX & mask). Clobbers ECX. */
static void
emit_fg_merge(as *a, uint32_t mask)
{
    alu_ri(a, 4, RAX, mask);
    LD(RCX, PE, PEF(fg));
    alu_ri(a, 4, RCX, ~mask);
    rr(a, 0x09, RCX, RAX, false);
    ST(RAX, PE, PEF(fg));
}

/* pixel_set() for one parameter, the value in EAX. */
static void
emit_set(as *a, pixel_param param, uint32_t src_mode)
{
    static const int8_t shift[] = { [P_R] = 16, [P_G] = 8, [P_B] = 0, [P_A] = 24 };
    switch (param) {
        case P_X:
            ST(RAX, PE, PEF(x));
            break;
        case P_Y:
            ST(RAX, PE, PEF(y));
            break;
        case P_F:
            ST(RAX, PE, PEF(f));
            break;
        case P_Z:
            ST(RAX, PE, PEF(z));
            if (src_mode & SRC_Z_SCISSOR) {
                b8(a, 0xa9); /* test eax, imm32 */
                d32(a, 0xf8000000);
                rr(a, 0x0f95, 0, RCX, false); /* setnz cl */
                rm(a, 0x88, RCX, PE, NO_INDEX, PEF(z_ovf), false, true);
            } else {
                rm(a, 0xc6, 0, PE, NO_INDEX, PEF(z_ovf), false, false);
                b8(a, 0);
            }
            break;
        case P_U:
        case P_V:
            {
                bool u = (param == P_U) != !!(src_mode & SRC_SWAP_UV);
                /* emit_uv_mask() sets the mask once, before the loop. */
                ST(RAX, PE, u ? PEF(u) : PEF(v));
                break;
            }
        case P_R:
        case P_G:
        case P_B:
        case P_A:
            ST(RAX, PE, param == P_R ? PEF(r) : param == P_G ? PEF(g) : param == P_B ? PEF(b) : PEF(a));
            emit_sat(a);
            shift_ri(a, 4, RAX, shift[param], false);
            emit_fg_merge(a, 0xffU << shift[param]);
            break;
        case P_I:
            ST(RAX, PE, PEF(r));
            ST(RAX, PE, PEF(g));
            ST(RAX, PE, PEF(b));
            emit_sat(a);
            rr(a, 0x69, RAX, RAX, false); /* imul eax, eax, 0x010101 */
            d32(a, 0x010101);
            emit_fg_merge(a, 0xffffff);
            break;
        default:
            break;
    }
}

/* The mask pixel_set() takes from SrcMask with U or V. SrcMask is fixed for
   the span, so this runs once before its first pixel. Clobbers ECX. */
static void
emit_uv_mask(as *a, pixel_param param, uint32_t src_mode)
{
    bool u = (param == P_U) != !!(src_mode & SRC_SWAP_UV);
    LD(RCX, PE, PEF(src_mask));
    if (!u)
        shift_ri(a, 5, RCX, 16, false);
    alu_ri(a, 4, RCX, 0x7ff);
    ST(RCX, PE, u ? PEF(u_mask) : PEF(v_mask));
}

/* EAX = texture_coord(EAX, mask from the engine). Clobbers ECX. */
static void
emit_coord(as *a, int32_t mask_field, bool clamped)
{
    LD(RCX, PE, mask_field);
    if (clamped) {
        rr(a, 0x39, RCX, RAX, false);   /* cmp eax, ecx */
        rr(a, 0x0f4f, RAX, RCX, false); /* cmovg eax, ecx (signed) */
        rr(a, 0x31, RCX, RCX, false);
        rr(a, 0x85, RAX, RAX, false);
        rr(a, 0x0f4c, RAX, RCX, false); /* cmovl eax, ecx */
    } else
        rr(a, 0x21, RCX, RAX, false);
}

/* XMM xmm = the decoded texel at (row, x) as bytes R, G, B, A (or B, G, R, A
   for fmt == 12, swapped after sampling). Clobbers R9, R10 (for fmt 8..10). */
static void
emit_tap(as *a, unsigned fmt, uint32_t src_mode, const span_consts *k, unsigned xmm, unsigned row, unsigned x)
{
    if (fmt >= 8 && fmt <= 10) {
        MOV(R9, x);
        shift_ri(a, 5, R9, 1, false);
        rr(a, 0x01, row, R9, false);
        emit_load8(a, R9, R9);
        MOV(R10, R9);
        alu_ri(a, 4, R10, 15);
        shift_ri(a, 5, R9, 4, false);
        rr(a, 0xf7, 0, x, false); /* test x, 1 */
        d32(a, 1);
        rr(a, 0x0f45, R9, R10, false); /* cmovnz r9d, r10d */
        sse_m(a, P66, MOVD_TO, xmm, CTX, (int) R9, 2, CTXF(pal[0]));
        return;
    }
    if (fmt == 2 || fmt == 3) {
        rmx(a, 0x8d, R9, row, (int) x, 0, 0, true, false);
        emit_load8(a, R9, R9);
        uint8_t *opaque = NULL;
        if (src_mode & SRC_CHROMA) {
            rr(a, 0x69, R10, R9, false); /* imul r10d, r9d, 0x01010101 */
            d32(a, 0x01010101);
            rm(a, 0x33, R10, PE, NO_INDEX, PEF(chroma_color), false, false);
            rm(a, 0x23, R10, PE, NO_INDEX, PEF(chroma_mask), false, false);
            opaque = jcc(a, CC_NE);
            sse_rip(a, P66, MOVD_TO, xmm, &k->black);
            uint8_t *keyed = jmp(a);
            land(opaque, a->p);
            opaque = keyed;
        }
        if (fmt == 2) {
            rr(a, 0x69, R9, R9, false);
            d32(a, 0x010101);
            alu_ri(a, 1, R9, 0xff000000U);
        } else {
            shift_ri(a, 4, R9, 24, false);
            alu_ri(a, 1, R9, 0x00ffffffU);
        }
        sse(a, P66, MOVD_TO, xmm, R9);
        if (opaque)
            land(opaque, a->p);
        return;
    }
    if (fmt == 12) {
        rmx(a, 0x8d, R9, row, (int) x, 2, 0, true, false);
        emit_load32(a, R9, R9);
        sse(a, P66, MOVD_TO, xmm, R9);
        if (src_mode & SRC_CHROMA) {
            rm(a, 0x33, R9, PE, NO_INDEX, PEF(chroma_color), false, false);
            rm(a, 0x23, R9, PE, NO_INDEX, PEF(chroma_mask), false, false);
            uint8_t *opaque = jcc(a, CC_NE);
            sse_rip(a, P66, MOVD_TO, xmm, &k->black);
            land(opaque, a->p);
        }
        return;
    }
    rmx(a, 0x8d, R9, row, (int) x, 1, 0, true, false); /* lea r9, [row + x * 2] */
    rr(a, 0x21, MASK, R9, false);
    alu_ri(a, 6, R9, 2);
    rm(a, 0x0fb7, R9, VRAM, R9, 0, false, false);
    sse_m(a, P66, MOVD_TO, xmm, TABLE, R9, 2, 0); /* movd xmm, [table + r9 * 4] */
    if (src_mode & SRC_CHROMA) {
        rr(a, 0x69, R9, R9, false); /* imul r9d, r9d, 0x10001 */
        d32(a, 0x10001);
        rm(a, 0x33, R9, PE, NO_INDEX, PEF(chroma_color), false, false);
        rm(a, 0x23, R9, PE, NO_INDEX, PEF(chroma_mask), false, false);
        uint8_t *opaque = jcc(a, CC_NE);
        sse_rip(a, P66, MOVD_TO, xmm, &k->black);
        land(opaque, a->p);
    }
}

/* reg = texture_coord(reg, mask from the engine). */
static void
emit_coord_reg(as *a, unsigned reg, int32_t mask_field, bool clamped)
{
    if (clamped) {
        rm(a, 0x3b, reg, PE, NO_INDEX, mask_field, false, false);   /* cmp reg, mask */
        rm(a, 0x0f4f, reg, PE, NO_INDEX, mask_field, false, false); /* cmovg reg, mask */
        rr(a, 0x85, reg, reg, false);
        rm(a, 0x0f4c, reg, CTX, NO_INDEX, CTXF(zero), false, false); /* cmovl reg, 0 */
    } else
        rm(a, 0x23, reg, PE, NO_INDEX, mask_field, false, false);
}

/* Prefetch the texel 1 << PREFETCH_SHIFT pixels on, extrapolated from how far u
   and v moved since the last pixel. R10D = u, R11D = v. Clobbers EAX, ECX. */
static void
emit_prefetch(as *a, unsigned fmt)
{
    for (unsigned axis = 0; axis < 2; axis++) {
        unsigned reg  = axis ? RCX : RAX, coord = axis ? R11 : R10;
        int32_t  prev = axis ? CTXF(prev_v) : CTXF(prev_u);
        MOV(reg, coord);
        rm(a, 0x2b, reg, CTX, NO_INDEX, prev, false, false); /* sub reg, [prev] */
        shift_ri(a, 4, reg, PREFETCH_SHIFT, false);
        rr(a, 0x01, coord, reg, false);
        ST(coord, CTX, prev);
        shift_ri(a, 7, reg, 16, false);
        rm(a, 0x23, reg, PE, NO_INDEX, axis ? PEF(v_mask) : PEF(u_mask), false, false);
    }
    rm(a, 0x0faf, RCX, CTX, NO_INDEX, CTXF(tex_stride), false, false);
    rm(a, 0x03, RCX, PE, NO_INDEX, PEF(src_base), false, false);
    if (fmt >= 8 && fmt <= 10)
        shift_ri(a, 5, RAX, 1, false);
    else if (fmt == 12)
        shift_ri(a, 4, RAX, 2, false);
    else if (fmt != 2 && fmt != 3)
        rr(a, 0x01, RAX, RAX, false);
    rr(a, 0x01, RAX, RCX, false);
    rr(a, 0x21, MASK, RCX, false);
    rm(a, 0x0f18, 1, VRAM, RCX, 0, false, false); /* prefetcht0 [vram + rcx] */
}

/* sample() into XMM3 as words R, G, B, A. R10D = u, R11D = v after any
   perspective division. Clobbers RAX, RCX, RDX, R8-R11, XMM0, XMM1, XMM4,
   XMM5. */
static void
emit_sample(as *a, unsigned fmt, uint32_t src_mode, const span_consts *k, bool single)
{
    bool uc = !!(src_mode & SRC_U_CLAMP), vc = !!(src_mode & SRC_V_CLAMP);
    bool filter = !!(src_mode & SRC_FILTER);
    if (!single) {
        rm(a, 0x83, 7, CTX, NO_INDEX, CTXF(prefetch), false, false); /* cmp dword [prefetch], 0 */
        b8(a, 0);
        uint8_t *cached = jcc(a, CC_E);
        emit_prefetch(a, fmt);
        land(cached, a->p);
    }

    /* EAX = x0, ECX = x1, EDX = row0, R8D = row1 */
    MOV(RAX, R10);
    shift_ri(a, 7, RAX, 16, false);
    MOV(RDX, R11);
    shift_ri(a, 7, RDX, 16, false);
    if (filter) {
        rm(a, 0x8d, RCX, RAX, NO_INDEX, 1, false, false); /* lea ecx, [rax + 1] */
        rm(a, 0x8d, R8, RDX, NO_INDEX, 1, false, false);
        emit_coord_reg(a, RCX, PEF(u_mask), uc);
        emit_coord_reg(a, R8, PEF(v_mask), vc);
        rm(a, 0x0faf, R8, CTX, NO_INDEX, CTXF(tex_stride), false, false);
        rm(a, 0x03, R8, PE, NO_INDEX, PEF(src_base), false, false);
    }
    emit_coord_reg(a, RAX, PEF(u_mask), uc);
    emit_coord_reg(a, RDX, PEF(v_mask), vc);
    rm(a, 0x0faf, RDX, CTX, NO_INDEX, CTXF(tex_stride), false, false);
    rm(a, 0x03, RDX, PE, NO_INDEX, PEF(src_base), false, false);

    if (!filter) {
        emit_tap(a, fmt, src_mode, k, 3, RDX, RAX);
        sse(a, P66, PXOR, 1, 1);
        sse(a, P66, PUNPCKLBW, 3, 1);
        if (fmt == 12)
            sse_i(a, PF2, PSHUFD, 3, 3, 0xc6);
        return;
    }

    /* (256 - f) | f << 16 = 256 + f * 0xffff, from u in R10D and v in R11D. */
    for (unsigned axis = 0; axis < 2; axis++) {
        unsigned reg = axis ? R11 : R10;
        shift_ri(a, 5, reg, 8, false);
        alu_ri(a, 4, reg, 255);
        rr(a, 0x69, reg, reg, false);
        d32(a, 0xffff);
        alu_ri(a, 0, reg, 256);
    }
    sse(a, P66, MOVD_TO, 3, R10);
    sse_i(a, P66, PSHUFD, 3, 3, 0);

    emit_tap(a, fmt, src_mode, k, 0, RDX, RAX);
    emit_tap(a, fmt, src_mode, k, 1, RDX, RCX);
    emit_tap(a, fmt, src_mode, k, 4, R8, RAX);
    emit_tap(a, fmt, src_mode, k, 5, R8, RCX);
    sse(a, P66, PUNPCKLBW, 0, 1); /* t00, t01 interleaved */
    sse(a, P66, PUNPCKLBW, 4, 5); /* t10, t11 */
    sse(a, P66, PXOR, 1, 1);
    sse(a, P66, PUNPCKLBW, 0, 1);
    sse(a, P66, PUNPCKLBW, 4, 1);
    sse(a, P66, PMADDWD, 0, 3);
    sse(a, P66, PMADDWD, 4, 3);
    sse_i(a, P66, PSHIFTD, SH_SLL, 4, 16);
    sse(a, P66, POR, 0, 4); /* h0 | h1 << 16 */
    sse_rip(a, P66, PXOR, 0, k->bias);
    sse(a, P66, MOVD_TO, 3, R11);
    sse_i(a, P66, PSHUFD, 3, 3, 0);
    sse(a, P66, PMADDWD, 3, 0);
    sse_rip(a, P66, PADDD, 3, k->round);
    sse_i(a, P66, PSHIFTD, SH_SRL, 3, 16);
    sse(a, P66, PACKSSDW, 3, 3);
    if (fmt == 12)
        sse_i(a, PF2, PSHUFD, 3, 3, 0xc6);
}

/* R10D, R11D = u, v * 2^24 / q unless Q is 0 (exact truncating division).
   Branchless remainder fixup for Q >= 64. Clobbers RAX, RCX, RDX, R8,
   XMM0-XMM1. Preserves R9 for vid_verite_test.c. */
static void
emit_divide_uv(as *a, const span_consts *k)
{
    LD(R8, PE, PEF(q));
    rr(a, 0x85, R8, R8, false);
    uint8_t *flat = jcc(a, CC_E);
    alu_ri(a, 7, R8, 64);
    uint8_t *fast = jcc(a, CC_AE);
    for (unsigned axis = 0; axis < 2; axis++) {
        rr(a, 0x63, RAX, axis ? R11 : R10, true); /* movsxd rax, r */
        shift_ri(a, 4, RAX, 24, true);
        b8(a, 0x48);
        b8(a, 0x99);              /* cqo */
        rr(a, 0xf7, 7, R8, true); /* idiv r8 */
        MOV(axis ? R11 : R10, RAX);
    }
    uint8_t *done = jmp(a);

    land(fast, a->p);
    /* XMM0 = 2^24 / q */
    b8(a, 0x66);
    rr(a, 0x0fef, 1, 1, false);             /* pxor xmm1, xmm1 */
    b8(a, 0xf2);
    rr(a, 0x0f2a, 1, R8, true);             /* cvtsi2sd xmm1, r8 */
    sse_rip(a, PF2, 0x0f10, 0, &k->two24);  /* movsd xmm0, 2^24 */
    b8(a, 0xf2);
    rr(a, 0x0f5e, 0, 1, false);             /* divsd xmm0, xmm1 */
    for (unsigned axis = 0; axis < 2; axis++) {
        unsigned coord = axis ? R11 : R10;
        /* RCX = estimate = (int64_t) (coord * scale) */
        b8(a, 0x66);
        rr(a, 0x0fef, 1, 1, false);
        b8(a, 0xf2);
        rr(a, 0x0f2a, 1, coord, false); /* cvtsi2sd xmm1, coord */
        b8(a, 0xf2);
        rr(a, 0x0f59, 1, 0, false);     /* mulsd xmm1, xmm0 */
        b8(a, 0xf2);
        rr(a, 0x0f2c, RCX, 1, true);    /* cvttsd2si rcx, xmm1 */
        /* RAX = remainder = coord * 2^24 - estimate * q */
        rr(a, 0x63, RAX, coord, true);
        shift_ri(a, 4, RAX, 24, true);
        rr(a, 0x89, RCX, RDX, true);
        rr(a, 0x0faf, RDX, R8, true);   /* imul rdx, r8 */
        rr(a, 0x29, RDX, RAX, true);    /* sub rax, rdx */
        /* Branchless:
           coord >= 0: estimate += (rem >= q) - (rem < 0)
           coord <  0: estimate += (rem >= 1) - (rem + q - 1 < 0) */
        mov_ri(a, RDX, 1);
        rr(a, 0x85, coord, coord, false);
        rr(a, 0x0f49, RDX, R8, true);   /* cmovns rdx, r8 */
        rr(a, 0x39, RDX, RAX, true);    /* cmp rax, rdx */
        rr(a, 0x0f9d, 0, RDX, false);   /* setge dl */
        rr(a, 0x0fb6, RDX, RDX, false); /* movzx edx, dl */
        rr(a, 0x01, RDX, RCX, true);    /* add rcx, rdx */
        rm(a, 0x8d, RDX, RAX, (int) R8, -1, true, false); /* lea rdx, [rax + r8 - 1] */
        rr(a, 0x85, coord, coord, false);
        rr(a, 0x0f48, RAX, RDX, true);  /* cmovs rax, rdx */
        shift_ri(a, 5, RAX, 63, true);  /* shr rax, 63 */
        rr(a, 0x29, RAX, RCX, true);    /* sub rcx, rax */
        MOV(coord, RCX);
    }
    land(done, a->p);
    land(flat, a->p);
}

/* Vectorized fog in XMM2 (4 x uint16 R, G, B, A). Clobbers RAX, RCX, XMM0,
   XMM1, XMM3, XMM4. */
static void
emit_fog(as *a, const span_consts *kc, int src_f)
{
    if (src_f >= 0) {
        LD(RAX, CTX, CTXF(values[0]) + 4 * src_f);
        emit_sat(a);
        sse(a, P66, MOVD_TO, 0, RAX);
        sse_i(a, PF2, PSHUFD, 0, 0, 0); /* pshuflw xmm0, xmm0, 0 */
    } else {
        sse_m(a, PF3, MOVD_FROM, 0, CTX, NO_INDEX, 0, CTXF(fog_f));
    }
    sse_m(a, PF3, MOVD_FROM, 1, CTX, NO_INDEX, 0, CTXF(fog));
    sse(a, P66, MOVDQA, 3, 2);
    sse(a, P66, PSUBUSW, 3, 1); /* max(c - fog, 0) */
    sse(a, P66, MOVDQA, 4, 1);
    sse(a, P66, PSUBUSW, 4, 2); /* max(fog - c, 0) */
    sse(a, P66, POR, 3, 4);     /* abs(c - fog) */
    sse(a, P66, MOVDQA, 4, 1);
    sse(a, P66, PCMPGTW, 4, 2); /* (fog > c) ? 0xffff : 0 */
    sse(a, P66, PMULLW, 3, 0);
    sse_rip(a, P66, PMULHUW, 3, kc->k8081);
    sse_i(a, P66, PSHIFTW, SH_SRL, 3, 7);
    sse(a, P66, PXOR, 3, 4);
    sse(a, P66, PSUBW, 3, 4);
    sse(a, P66, PADDW, 3, 1);
    sse_rip(a, P66, PAND, 3, kc->rgb);
    sse_rip(a, P66, PAND, 2, kc->alpha);
    sse(a, P66, POR, 2, 3);
}

static void
emit_blend_side(as *a, const span_consts *kc, unsigned func, bool source, unsigned dst_xmm, unsigned tmp_xmm)
{
    unsigned own   = source ? 2 : 3;
    unsigned other = source ? 3 : 2;
    if (func == 6) {
        sse(a, P66, PXOR, dst_xmm, dst_xmm);
        return;
    }
    if (func == 7 || func == 12 || func == 13 || func == 14) {
        sse(a, P66, MOVDQA, dst_xmm, own);
        return;
    }
    if (func == 15) {
        sse(a, P66, MOVDQA, dst_xmm, own);
        sse_rip(a, P66, PSUBW, dst_xmm, kc->k128);
        return;
    }
    switch (func) {
        case 0:
            sse(a, P66, MOVDQA, dst_xmm, other);
            break;
        case 1:
            sse_rip(a, P66, MOVDQA, dst_xmm, kc->k255);
            sse(a, P66, PSUBW, dst_xmm, other);
            break;
        case 2:
            sse_i(a, PF2, PSHUFD, dst_xmm, 2, 0xff);
            break;
        case 3:
            sse_i(a, PF2, PSHUFD, tmp_xmm, 2, 0xff);
            sse_rip(a, P66, MOVDQA, dst_xmm, kc->k255);
            sse(a, P66, PSUBW, dst_xmm, tmp_xmm);
            break;
        case 4:
            sse_i(a, PF2, PSHUFD, dst_xmm, 3, 0xff);
            break;
        case 5:
            sse_i(a, PF2, PSHUFD, tmp_xmm, 3, 0xff);
            sse_rip(a, P66, MOVDQA, dst_xmm, kc->k255);
            sse(a, P66, PSUBW, dst_xmm, tmp_xmm);
            break;
        case 8:
        case 9:
            sse_i(a, PF2, PSHUFD, tmp_xmm, 3, 0xff);
            sse_rip(a, P66, MOVDQA, dst_xmm, kc->k255);
            sse(a, P66, PSUBW, dst_xmm, tmp_xmm);
            sse_i(a, PF2, PSHUFD, tmp_xmm, 2, 0xff);
            sse(a, P66, PMINSW, dst_xmm, tmp_xmm);
            sse_rip(a, P66, PAND, dst_xmm, kc->rgb);
            sse_rip(a, P66, POR, dst_xmm, kc->alpha255);
            if (func == 9) {
                sse_rip(a, P66, MOVDQA, tmp_xmm, kc->k255);
                sse(a, P66, PSUBW, tmp_xmm, dst_xmm);
                sse(a, P66, MOVDQA, dst_xmm, tmp_xmm);
            }
            break;
        case 10:
            sse(a, P66, MOVDQA, dst_xmm, own);
            break;
        case 11:
            sse_rip(a, P66, MOVDQA, dst_xmm, kc->k255);
            sse(a, P66, PSUBW, dst_xmm, own);
            break;
    }
    sse(a, P66, PMULLW, dst_xmm, own);
    sse_rip(a, P66, PADDW, dst_xmm, kc->k127);
    sse_rip(a, P66, PMULHUW, dst_xmm, kc->k8081);
    sse_i(a, P66, PSHIFTW, SH_SRL, dst_xmm, 7);
}

/* Vectorized blend in XMM2. Computes and saves ctx.dst_addr when reading dst.
   Clobbers RAX, RCX, XMM0, XMM1, XMM3, XMM4. */
static void
emit_blend(as *a, const span_key *k, const span_consts *kc, int src_y, bool same_table)
{
    if (k->dst_mode & DST_RAW_READ_DISABLE) {
        sse_m(a, PF3, MOVD_FROM, 3, CTX, NO_INDEX, 0, CTXF(dst_c));
    } else {
        if (src_y >= 0) {
            MOV(RAX, RDI);
            rm(a, 0x0faf, RAX, CTX, NO_INDEX, CTXF(dst_stride), false, false);
            rm(a, 0x03, RAX, PE, NO_INDEX, PEF(dst_base), false, false);
        } else
            LD(RAX, CTX, CTXF(dst_row));
        if (k->dst_fmt == 12)
            rmx(a, 0x8d, RAX, RAX, RSI, 2, 0, false, false);
        else {
            rr(a, 0x01, RSI, RAX, false);
            rr(a, 0x01, RSI, RAX, false);
        }
        ST(RAX, CTX, CTXF(dst_addr));
        if (k->dst_fmt == 12) {
            emit_load32(a, RAX, RAX);
            sse(a, P66, MOVD_TO, 3, RAX);
            sse(a, P66, PXOR, 1, 1);
            sse(a, P66, PUNPCKLBW, 3, 1);
            sse_i(a, PF2, PSHUFD, 3, 3, 0xc6);
        } else {
            emit_load16(a, RAX, RAX);
            if (same_table) {
                sse_m(a, P66, MOVD_TO, 3, TABLE, RAX, 2, 0);
            } else {
                rm(a, 0x8b, RCX, CTX, NO_INDEX, CTXF(dst_table), true, false);
                sse_m(a, P66, MOVD_TO, 3, RCX, RAX, 2, 0);
            }
            sse(a, P66, PXOR, 1, 1);
            sse(a, P66, PUNPCKLBW, 3, 1);
        }
    }
    unsigned sf = (k->dst_mode >> 4) & 15, df = (k->dst_mode >> 8) & 15;
    emit_blend_side(a, kc, sf, true, 0, 4);
    emit_blend_side(a, kc, df, false, 1, 4);
    sse(a, P66, MOVDQA, 2, 0);
    sse(a, P66, PADDW, 2, 1);
    if (sf == 15 || df == 15) {
        sse(a, P66, PXOR, 0, 0);
        sse(a, P66, PMAXSW, 2, 0);
    }
    sse_rip(a, P66, PMINSW, 2, kc->k255);
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

/* ctx.field = stride_table[(pe->stride >> shift) & 0x7f]. */
static void
emit_stride(as *a, unsigned shift, int32_t field)
{
    LD(RAX, PE, PEF(stride));
    if (shift)
        shift_ri(a, 5, RAX, shift, false);
    alu_ri(a, 4, RAX, 0x7f);
    mov_ri64(a, RDX, (uint64_t) (uintptr_t) stride_table);
    rmx(a, 0x0fb7, RAX, RDX, (int) RAX, 1, 0, false, false);
    ST(RAX, CTX, field);
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

/* The constants for this state; see span_consts. */
static void
build_consts(span_consts *c, const span_key *k)
{
    /* Field width and position for R, G, B, A. */
    static const uint8_t n565[]  = { 5, 6, 5, 0 }, pos565[] = { 11, 5, 0, 0 };
    static const uint8_t n4444[] = { 4, 4, 4, 4 }, pos4444[] = { 8, 4, 0, 12 };
    static const uint8_t n1555[] = { 5, 5, 5, 1 }, pos1555[] = { 10, 5, 0, 15 };
    const uint8_t       *ns  = k->dst_fmt == 4 ? n565 : k->dst_fmt == 5 ? n4444 : n1555;
    const uint8_t       *pos = k->dst_fmt == 4 ? pos565 : k->dst_fmt == 5 ? pos4444 : pos1555;
    memset(c, 0, sizeof(*c));
    for (unsigned l = 0; l < 8; l++) {
        unsigned ch      = l & 3;
        unsigned n       = k->dst_fmt == 12 ? 8 : ns[ch];
        unsigned p       = k->dst_fmt == 12 ? 0 : pos[ch];
        bool     on      = n != 0;
        c->k127[l]       = 127;
        c->k128[l]       = 128;
        c->k8081[l]      = 0x8081;
        c->k255[l]       = 255;
        c->shift[l]      = on ? 1U << (8 + n) : 0;
        c->pos[l]        = on ? 1U << p : 0;
        c->ones[l]       = 1;
        c->rgb[l]        = ch == 3 ? 0 : 0xffff;
        c->alpha[l]      = ch == 3 ? 0xffff : 0;
        c->alpha255[l]   = ch == 3 ? 255 : 0;
        for (unsigned i = 0; i < 16; i++)
            c->dither[i][l] = on ? (dither_table[i] << 4) >> n : 0;
    }
    for (unsigned l = 0; l < 4; l++) {
        c->bias[l]  = 0x80008000;
        c->round[l] = 0x00808000;
    }
    c->black = k->src_mode & SRC_CHROMA_BLACK_YUV ? 128 | 16 << 8 | 128 << 16 : 0;
    c->two24 = 16777216.0;
}

/* Mark dirty VRAM pages for [lo_off, hi_off] in core->page_gen / global_gen.
   Assumes R8 = ctx.core. Clobbers RAX, RCX, RDX. */
static void
emit_dirty_pages(as *a, int32_t lo_off, int32_t hi_off)
{
    LD(RAX, CTX, lo_off);
    LD(RCX, CTX, hi_off);
    rr(a, 0x39, RCX, RAX, false); /* cmp eax, ecx */
    uint8_t *none = jcc(a, CC_A);
    shift_ri(a, 5, RAX, 12, false);
    shift_ri(a, 5, RCX, 12, false);
    MOV(RDX, RCX);
    rr(a, 0x29, RAX, RDX, false);
    alu_ri(a, 7, RDX, 16);
    uint8_t *few_pages = jcc(a, CC_BE);
    rm(a, 0xff, 0, R8, NO_INDEX, (int32_t) offsetof(verite_core, global_gen), false, false);
    uint8_t *dirty_done = jmp(a);
    land(few_pages, a->p);
    uint8_t *page_loop = a->p;
    MOV(RDX, RAX);
    alu_ri(a, 4, RDX, 1023);
    rmx(a, 0xff, 0, R8, (int) RDX, 2, (int32_t) offsetof(verite_core, page_gen), false, false);
    rr(a, 0xff, 0, RAX, false);
    rr(a, 0x39, RCX, RAX, false);
    land(jcc(a, CC_BE), page_loop);
    land(dirty_done, a->p);
    land(none, a->p);
}

/* Generates a span_fn. op holds the register count, plus 0x100 for a single
   draw that writes back its stepped values and decrements Cnt. */
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

    /* Constants first, then the code. */
    while ((uintptr_t) a->p & 15)
        b8(a, 0xcc);
    span_consts *kc = (span_consts *) a->p;
    build_consts(kc, k);
    a->p += sizeof(span_consts);
    uint8_t *start = a->p;

    static const unsigned saved[] = { RBX, RBP, RSI, RDI, R12, R13, R14, R15 };
    uint32_t              frame   = ((sizeof(span_ctx) + 15) & ~15U) + 16;
    for (unsigned i = 0; i < 8; i++)
        rex(a, false, 0, 0, saved[i], false), b8(a, 0x50 | (saved[i] & 7));
    rr(a, 0x81, 5, RSP, true); /* sub rsp, frame */
    d32(a, frame);
    rm(a, 0x8d, CTX, RSP, NO_INDEX, 15, true, false); /* lea r15, [rsp + 15] */
    rr(a, 0x83, 4, CTX, true);                          /* and r15, -16 */
    b8(a, 0xf0);
#    ifdef _WIN32
    unsigned arg_core = RCX, arg_site = RDX;
#    else
    unsigned arg_core = RDI, arg_site = RSI;
#    endif
    rm(a, 0x89, arg_core, CTX, NO_INDEX, CTXF(core), true, false);
    rm(a, 0x89, arg_site, CTX, NO_INDEX, CTXF(site), true, false);
    rm(a, 0x8d, PE, arg_core, NO_INDEX, (int32_t) offsetof(verite_core, pe), true, false);
    rm(a, 0x8b, RAX, arg_core, NO_INDEX, (int32_t) offsetof(verite_core, jit_mem), true, false);
    rm(a, 0x8b, VRAM, RAX, NO_INDEX, (int32_t) offsetof(verite_jit_memory, vram), true, false);
    LD(MASK, RAX, (int32_t) offsetof(verite_jit_memory, mask));
    ST(MASK, CTX, CTXF(byte_mask));
    alu_ri(a, 4, MASK, ~1U);
    rr(a, 0x31, DREW, DREW, false);
    rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(lo), false, false);
    d32(a, UINT32_MAX);
    rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(hi), false, false);
    d32(a, 0);
    if (zwrite) {
        rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(zlo), false, false);
        d32(a, UINT32_MAX);
        rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(zhi), false, false);
        d32(a, 0);
    }

    /* cnt = (int16_t) Cnt; nothing happens unless it is positive. */
    rm(a, 0x0fbf, RAX, PE, NO_INDEX, PEF(cnt), false, false);
    ST(RAX, CTX, CTXF(cnt));
    rr(a, 0x85, RAX, RAX, false);
    uint8_t *empty = jcc(a, CC_LE);
    if (single)
        mov_ri(a, COUNT, 1);
    else
        MOV(COUNT, RAX);

    rm(a, 0x8b, RCX, CTX, NO_INDEX, CTXF(site), true, false);
    rm(a, 0x8b, RDX, RCX, NO_INDEX, (int32_t) offsetof(verite_draw_site, values), true, false);
    rm(a, 0x8b, RCX, RCX, NO_INDEX, (int32_t) offsetof(verite_draw_site, deltas), true, false);
    for (unsigned i = 0; i < count; i++) {
        LD(RAX, RDX, 4 * (int32_t) i);
        ST(RAX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
        LD(RAX, RCX, 4 * (int32_t) i);
        ST(RAX, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
    }

    for (unsigned i = 0; i < count; i++)
        if (params[i] == P_U || params[i] == P_V)
            emit_uv_mask(a, params[i], k->src_mode);

    /* Unpack scissor bounds [lo, min(hi, 2048)). */
    for (unsigned axis = 0; axis < 2; axis++) {
        LD(RAX, PE, axis ? PEF(scissor_y) : PEF(scissor_x));
        MOV(RCX, RAX);
        shift_ri(a, 5, RCX, 16, false);
        alu_ri(a, 4, RCX, 0xfff);
        ST(RCX, CTX, CTXF(scissor[0]) + 8 * (int32_t) axis);
        alu_ri(a, 4, RAX, 0xfff);
        mov_ri(a, RCX, 2048);
        rr(a, 0x39, RCX, RAX, false);
        rr(a, 0x0f4f, RAX, RCX, false);
        ST(RAX, CTX, CTXF(scissor[1]) + 8 * (int32_t) axis);
    }
    rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(zero), false, false);
    d32(a, 0);

    /* Reject whole spans early (O(1) fast-forward). */
    uint8_t *ff[8];
    unsigned nff = 0;
    LD(RAX, CTX, CTXF(scissor[0]));
    rm(a, 0x3b, RAX, CTX, NO_INDEX, CTXF(scissor[1]), false, false);
    ff[nff++] = jcc(a, CC_GE);
    LD(RAX, CTX, CTXF(scissor[2]));
    rm(a, 0x3b, RAX, CTX, NO_INDEX, CTXF(scissor[3]), false, false);
    ff[nff++] = jcc(a, CC_GE);
    if (src_y < 0) {
        LD(RDX, PE, PEF(y));
        shift_ri(a, 7, RDX, 16, false);
        rm(a, 0x3b, RDX, CTX, NO_INDEX, CTXF(scissor[2]), false, false);
        ff[nff++] = jcc(a, CC_L);
        rm(a, 0x3b, RDX, CTX, NO_INDEX, CTXF(scissor[3]), false, false);
        ff[nff++] = jcc(a, CC_GE);
    }
    if ((k->src_mode & SRC_Z_SCISSOR) && src_z < 0) {
        rm(a, 0x80, 7, PE, NO_INDEX, PEF(z_ovf), false, false);
        b8(a, 0);
        ff[nff++] = jcc(a, CC_NE);
    }
    if (depth_mode == 7)
        ff[nff++] = jmp(a);
    if (raw_copy && (k->dst_mode & DST_RAW_TREJECT)) {
        LD(RCX, PE, PEF(alpha_thres));
        alu_ri(a, 4, RCX, 255);
        alu_ri(a, 7, RCX, 255);
        ff[nff++] = jcc(a, CC_E);
    }

    uint8_t *preclip_ff = NULL;
    if (preclip) {
        /* x0 = pe->x_shadow >> 16 */
        LD(R8, PE, PEF(x_shadow));
        shift_ri(a, 7, R8, 16, false);
        /* x_step = x_dir ? -0x10000 : 0x10000; pe->x_shadow += n * x_step */
        rm(a, 0x0fb6, R9, PE, NO_INDEX, PEF(x_dir), false, false);
        rr(a, 0xf7, 3, R9, false);
        alu_ri(a, 4, R9, 0xfffe0000);
        alu_ri(a, 0, R9, 0x10000);
        MOV(RAX, R9);
        rr(a, 0x0faf, RAX, COUNT, false);
        rm(a, 0x01, RAX, PE, NO_INDEX, PEF(x_shadow), false, false);

        /* Clip [0, COUNT) against [x_min, x_end). */
        LD(R10, CTX, CTXF(scissor[0]));
        LD(R11, CTX, CTXF(scissor[1]));
        rr(a, 0x85, R9, R9, false);
        uint8_t *backward = jcc(a, CC_S);

        /* Forward: start = max(x0, x_min), stop = min(x0 + COUNT, x_end). */
        MOV(RSI, R8);
        rr(a, 0x39, R10, RSI, false);
        rr(a, 0x0f4c, RSI, R10, false); /* cmovl esi, r10d */
        rmx(a, 0x8d, RAX, R8, (int) COUNT, 0, 0, false, false);
        rr(a, 0x39, R11, RAX, false);
        rr(a, 0x0f4f, RAX, R11, false); /* cmovg eax, r11d */
        rr(a, 0x29, RSI, RAX, false);   /* vis = stop - start */
        uint8_t *fwd_empty = jcc(a, CC_LE);
        MOV(RCX, RSI);
        rr(a, 0x29, R8, RCX, false);    /* skip = start - x0 */
        mov_ri(a, R8, 1);
        uint8_t *clip_done = jmp(a);

        land(backward, a->p);
        /* Backward: hi_x = min(x0, x_end - 1), lo_x = max(x0 - COUNT + 1, x_min). */
        rm(a, 0x8d, RSI, R11, NO_INDEX, -1, false, false);
        rr(a, 0x39, R8, RSI, false);
        rr(a, 0x0f4f, RSI, R8, false); /* cmovg esi, r8d */
        MOV(RAX, R8);
        rr(a, 0x29, COUNT, RAX, false);
        alu_ri(a, 0, RAX, 1);
        rr(a, 0x39, R10, RAX, false);
        rr(a, 0x0f4c, RAX, R10, false); /* cmovl eax, r10d */
        MOV(R11, RSI);
        rr(a, 0x29, RAX, R11, false);
        alu_ri(a, 0, R11, 1);          /* vis = hi_x - lo_x + 1 */
        uint8_t *bwd_empty = jcc(a, CC_LE);
        MOV(RAX, R11);
        MOV(RCX, R8);
        rr(a, 0x29, RSI, RCX, false);  /* skip = x0 - hi_x */
        mov_ri(a, R8, UINT32_MAX);     /* -1 */

        land(clip_done, a->p);
        /* RAX = vis, RCX = skip, R8D = x_dir_step, RSI = x_first. */
        ST(R8, CTX, CTXF(x_dir_step));
        if (unconditional)
            ST(RSI, CTX, CTXF(x_first));
        MOV(R9, COUNT);
        rr(a, 0x29, RAX, R9, false);
        rr(a, 0x29, RCX, R9, false);   /* tail = n - vis - skip */
        ST(R9, CTX, CTXF(tail));
        MOV(COUNT, RAX);
        rr(a, 0x85, RCX, RCX, false);
        uint8_t *no_skip = jcc(a, CC_E);
        if (q_index >= 0) {
            rm(a, 0x8d, RAX, RCX, NO_INDEX, -1, false, false);
            LD(RDX, CTX, CTXF(deltas[0]) + 4 * q_index);
            rr(a, 0x0faf, RAX, RDX, false);
            rm(a, 0x03, RAX, CTX, NO_INDEX, CTXF(values[0]) + 4 * q_index, false, false);
            ST(RAX, PE, PEF(q));
        }
        for (unsigned i = 0; i < count; i++) {
            if (params[i] == P_NONE && !single)
                continue;
            LD(RAX, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
            rr(a, 0x0faf, RAX, RCX, false);
            rm(a, 0x01, RAX, CTX, NO_INDEX, CTXF(values[0]) + 4 * (int32_t) i, false, false);
        }
        land(no_skip, a->p);
        uint8_t *clip_ok = jmp(a);

        land(fwd_empty, a->p);
        land(bwd_empty, a->p);
        preclip_ff = jmp(a);
        land(clip_ok, a->p);
    } else if (shadow) {
        rm(a, 0x0fb6, RAX, PE, NO_INDEX, PEF(x_dir), false, false);
        rr(a, 0xf7, 3, RAX, false);
        alu_ri(a, 4, RAX, 0xfffe0000);
        alu_ri(a, 0, RAX, 0x10000);
        ST(RAX, CTX, CTXF(x_step));
    }

    emit_stride(a, 8, CTXF(dst_stride));
    if (depth_mode || zwrite)
        emit_stride(a, 16, CTXF(z_stride));
    if (func) {
        emit_stride(a, 0, CTXF(tex_stride));
        if (!single) {
            rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(prev_u), false, false);
            d32(a, 0);
            rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(prev_v), false, false);
            d32(a, 0);
        }
    }

    if (src_y < 0) {
        LD(RDX, PE, PEF(y));
        shift_ri(a, 7, RDX, 16, false);
        MOV(RAX, RDX);
        rm(a, 0x0faf, RAX, CTX, NO_INDEX, CTXF(dst_stride), false, false);
        rm(a, 0x03, RAX, PE, NO_INDEX, PEF(dst_base), false, false);
        ST(RAX, CTX, CTXF(dst_row));
        if (depth_mode || zwrite) {
            rm(a, 0x0faf, RDX, CTX, NO_INDEX, CTXF(z_stride), false, false);
            rm(a, 0x03, RDX, PE, NO_INDEX, PEF(z_base), false, false);
            ST(RDX, CTX, CTXF(z_row));
        }
    }
    if (dither && !raw_copy) {
        LD(RDX, PE, PEF(dither_offset));
        MOV(RAX, RDX);
        alu_ri(a, 4, RAX, 3);
        ST(RAX, CTX, CTXF(dither_x));
        shift_ri(a, 5, RDX, 4, false);
        alu_ri(a, 4, RDX, 3);
        ST(RDX, CTX, CTXF(dither_y));
        if (src_y < 0) {
            LD(RAX, PE, PEF(y));
            shift_ri(a, 7, RAX, 16, false);
            rr(a, 0x01, RDX, RAX, false);
            alu_ri(a, 4, RAX, 3);
            shift_ri(a, 4, RAX, 6, false);
            rex(a, true, RCX, 0, 0, false);
            b8(a, 0x8d);
            b8(a, ((RCX & 7) << 3) | 5);
            d32(a, (uint32_t) ((const uint8_t *) kc->dither - (a->p + 4)));
            rr(a, 0x01, RAX, RCX, true);
            rm(a, 0x89, RCX, CTX, NO_INDEX, CTXF(dither_row), true, false);
        }
    }
    if ((k->dst_mode & DST_RAW_TREJECT) && !raw_copy) {
        LD(RCX, PE, PEF(alpha_thres));
        alu_ri(a, 4, RCX, 255);
        ST(RCX, CTX, CTXF(thres));
    }
    if (func != 1 && !raw_copy) {
        sse_m(a, P66, MOVD_TO, 2, PE, NO_INDEX, 0, PEF(fg));
        sse(a, P66, PXOR, 1, 1);
        sse(a, P66, PUNPCKLBW, 2, 1);
        sse_i(a, PF2, PSHUFD, 2, 2, 0xc6);
        for (unsigned lane = 0; lane < 4; lane++)
            if (src_ch[lane] >= 0)
                sse_i(a, P66, PINSRW, 2, DREW, lane);
        sse_m(a, P66, MOVQ_ST, 2, CTX, NO_INDEX, 0, CTXF(fg));
    }
    if (fog && !raw_copy) {
        sse_m(a, P66, MOVD_TO, 0, PE, NO_INDEX, 0, PEF(fog_color));
        sse(a, P66, PXOR, 1, 1);
        sse(a, P66, PUNPCKLBW, 0, 1);
        sse_i(a, PF2, PSHUFD, 0, 0, 0xc6);
        sse_m(a, P66, MOVQ_ST, 0, CTX, NO_INDEX, 0, CTXF(fog));
        if (src_f < 0) {
            LD(RAX, PE, PEF(f));
            emit_sat(a);
            sse(a, P66, MOVD_TO, 0, RAX);
            sse_i(a, PF2, PSHUFD, 0, 0, 0);
            sse_m(a, P66, MOVQ_ST, 0, CTX, NO_INDEX, 0, CTXF(fog_f));
        }
    }
    if (blend && (k->dst_mode & DST_RAW_READ_DISABLE) && !raw_copy) {
        sse_m(a, P66, MOVD_TO, 0, PE, NO_INDEX, 0, PEF(dst_color));
        sse(a, P66, PXOR, 1, 1);
        sse(a, P66, PUNPCKLBW, 0, 1);
        sse_i(a, PF2, PSHUFD, 0, 0, 0xc6);
        sse_m(a, P66, MOVQ_ST, 0, CTX, NO_INDEX, 0, CTXF(dst_c));
    }
    if (dst_computed && !same_table && dst_16 && !raw_copy) {
        mov_ri64(a, RAX, (uint64_t) (uintptr_t) texel_table(k->dst_fmt, !!(k->dst_mode & DST_RAW_NO_PAD)));
        rm(a, 0x89, RAX, CTX, NO_INDEX, CTXF(dst_table), true, false);
    }
    if (func && !raw_copy) {
        if (src_16)
            mov_ri64(a, TABLE, (uint64_t) (uintptr_t) texel_table(fmt, !!(k->src_mode & SRC_NO_PAD)));
        else if (fmt >= 8 && fmt <= 10) {
            const uint32_t *ptab = texel_table(fmt - 4, !!(k->src_mode & SRC_NO_PAD));
            mov_ri64(a, RDX, (uint64_t) (uintptr_t) ptab);
            for (unsigned i = 0; i < 8; i++) {
                LD(RAX, PE, PEF(tex4pal[0]) + 4 * (int32_t) i);
                MOV(RCX, RAX);
                shift_ri(a, 5, RCX, 16, false);
                rmx(a, 0x8b, RCX, RDX, (int) RCX, 2, 0, false, false);
                ST(RCX, CTX, CTXF(pal[0]) + 4 * (int32_t) (2 * i));
                alu_ri(a, 4, RAX, 0xffff);
                rmx(a, 0x8b, RAX, RDX, (int) RAX, 2, 0, false, false);
                ST(RAX, CTX, CTXF(pal[0]) + 4 * (int32_t) (2 * i + 1));
            }
        }
        if (!single) {
            LD(RAX, PE, PEF(u_mask));
            alu_ri(a, 0, RAX, 1);
            if (fmt >= 8 && fmt <= 10)
                shift_ri(a, 5, RAX, 1, false);
            else if (fmt == 12)
                shift_ri(a, 4, RAX, 2, false);
            else if (fmt != 2 && fmt != 3)
                rr(a, 0x01, RAX, RAX, false);
            alu_ri(a, 0, RAX, 63);
            alu_ri(a, 4, RAX, ~63U);
            LD(RCX, PE, PEF(v_mask));
            alu_ri(a, 0, RCX, 1);
            rr(a, 0x0faf, RAX, RCX, false);
            rr(a, 0x31, RCX, RCX, false);
            alu_ri(a, 7, RAX, 16384);
            rr(a, 0x0f97, 0, RCX, false); /* seta cl */
            ST(RCX, CTX, CTXF(prefetch));
        }
    }

    uint8_t *loop = a->p;

    /* ESI = x, EDI = y (when needed); scissor if not preclipped. */
    uint8_t *skips[16];
    unsigned nskip = 0;
    if (!preclip) {
        emit_source(a, RSI, src_x, shadow ? PEF(x_shadow) : PEF(x));
        shift_ri(a, 7, RSI, 16, false);
    }
    if (src_y >= 0 || (dst_computed && src_y >= 0)) {
        emit_source(a, RDI, src_y, PEF(y));
        shift_ri(a, 7, RDI, 16, false);
    }
    if (!preclip) {
        for (unsigned axis = 0; axis < (src_y >= 0 ? 2U : 1U); axis++) {
            unsigned r = axis ? RDI : RSI;
            rm(a, 0x3b, r, CTX, NO_INDEX, CTXF(scissor[0]) + 8 * (int32_t) axis, false, false);
            skips[nskip++] = jcc(a, CC_L);
            rm(a, 0x3b, r, CTX, NO_INDEX, CTXF(scissor[1]) + 8 * (int32_t) axis, false, false);
            skips[nskip++] = jcc(a, CC_GE);
        }
    }
    if ((k->src_mode & SRC_Z_SCISSOR) && src_z >= 0) {
        rm(a, 0xf7, 0, CTX, NO_INDEX, CTXF(values[0]) + 4 * src_z, false, false);
        d32(a, 0xf8000000);
        skips[nskip++] = jcc(a, CC_NE);
    }

    if (depth_mode || zwrite) {
        if (src_y >= 0) {
            MOV(RAX, RDI);
            rm(a, 0x0faf, RAX, CTX, NO_INDEX, CTXF(z_stride), false, false);
            rm(a, 0x03, RAX, PE, NO_INDEX, PEF(z_base), false, false);
        } else
            LD(RAX, CTX, CTXF(z_row));
        rr(a, 0x01, RSI, RAX, false);
        rr(a, 0x01, RSI, RAX, false);
        ST(RAX, CTX, CTXF(z_addr));
        emit_source(a, RCX, src_z, PEF(z));
        shift_ri(a, 7, RCX, 11, false);
        alu_ri(a, 4, RCX, 0xffff);
        ST(RCX, CTX, CTXF(z));
        if (depth_mode) {
            static const uint8_t fail[] = { 0, CC_AE, CC_NE, CC_A, CC_BE, CC_E, CC_B };
            emit_load16(a, RDX, RAX);
            rr(a, 0x39, RDX, RCX, false);
            skips[nskip++] = jcc(a, fail[depth_mode]);
        }
    }

    if (func) {
        emit_source(a, R10, src_u, PEF(u));
        emit_source(a, R11, src_v, PEF(v));
        if (q_index >= 0)
            emit_divide_uv(a, kc);
    }
    if (raw_copy) {
        bool uc = !!(k->src_mode & SRC_U_CLAMP), vc = !!(k->src_mode & SRC_V_CLAMP);
        MOV(RAX, R11);
        shift_ri(a, 7, RAX, 16, false);
        emit_coord(a, PEF(v_mask), vc);
        rm(a, 0x0faf, RAX, CTX, NO_INDEX, CTXF(tex_stride), false, false);
        rm(a, 0x03, RAX, PE, NO_INDEX, PEF(src_base), false, false);
        MOV(R9, RAX);
        MOV(RAX, R10);
        shift_ri(a, 7, RAX, 16, false);
        emit_coord(a, PEF(u_mask), uc);
        rr(a, 0x01, RAX, RAX, false);
        rr(a, 0x01, R9, RAX, false);
        emit_load16(a, R8, RAX);
        if (k->src_mode & SRC_CHROMA) {
            rr(a, 0x69, RCX, R8, false);
            d32(a, 0x10001);
            rm(a, 0x33, RCX, PE, NO_INDEX, PEF(chroma_color), false, false);
            rm(a, 0x23, RCX, PE, NO_INDEX, PEF(chroma_mask), false, false);
            uint8_t *opaque = jcc(a, CC_NE);
            if (k->dst_mode & DST_RAW_TREJECT)
                skips[nskip++] = jmp(a);
            else
                mov_ri(a, R8, k->src_mode & SRC_CHROMA_BLACK_YUV ? 0x8090 : 0);
            land(opaque, a->p);
        }
    } else {
        if (func)
            emit_sample(a, fmt, k->src_mode, kc, single);
        if (func != 1) {
            sse_m(a, PF3, MOVD_FROM, 2, CTX, NO_INDEX, 0, CTXF(fg));
            for (unsigned ch = 0; ch < 4; ch++) {
                int  idx   = src_ch[ch];
                bool first = idx >= 0;
                for (unsigned prev = 0; prev < ch; prev++)
                    first = first && src_ch[prev] != idx;
                if (!first)
                    continue;
                unsigned order = 0;
                for (unsigned lane = 0; lane < 4; lane++)
                    order |= (src_ch[lane] == idx ? 0U : 1U) << (2 * lane);
                LD(RAX, CTX, CTXF(values[0]) + 4 * idx);
                emit_sat(a);
                sse(a, P66, MOVD_TO, 0, RAX);
                sse_i(a, PF2, PSHUFD, 0, 0, order);
                sse(a, P66, POR, 2, 0);
            }
        }
        if (func == 1)
            sse(a, P66, MOVDQA, 2, 3);
        else if (func == 3) {
            sse(a, P66, PMULLW, 2, 3);
            sse_rip(a, P66, PADDW, 2, kc->k127);
            sse_rip(a, P66, PMULHUW, 2, kc->k8081);
            sse_i(a, P66, PSHIFTW, SH_SRL, 2, 7);
        } else if (func == 2) {
            sse_i(a, PF2, PSHUFD, 0, 3, 0xff);
            sse_rip(a, P66, MOVDQA, 1, kc->k255);
            sse(a, P66, PSUBW, 1, 0);
            sse(a, P66, PMULLW, 0, 3);
            sse(a, P66, PMULLW, 1, 2);
            sse(a, P66, PADDW, 0, 1);
            sse_rip(a, P66, PADDW, 0, kc->k127);
            sse_rip(a, P66, PMULHUW, 0, kc->k8081);
            sse_i(a, P66, PSHIFTW, SH_SRL, 0, 7);
            sse_rip(a, P66, PAND, 0, kc->rgb);
            sse_rip(a, P66, PAND, 2, kc->alpha);
            sse(a, P66, POR, 2, 0);
        }

        if (k->dst_mode & DST_RAW_TREJECT) {
            sse_i(a, P66, PEXTRW, RAX, 2, 3);
            rm(a, 0x3b, RAX, CTX, NO_INDEX, CTXF(thres), false, false);
            skips[nskip++] = jcc(a, CC_LE);
        }

        if (fog)
            emit_fog(a, kc, src_f);
        if (blend)
            emit_blend(a, k, kc, src_y, same_table);

        if (k->dst_fmt == 12) {
            sse_i(a, PF2, PSHUFD, 2, 2, 0xc6); /* R, G, B, A -> B, G, R, A */
            sse(a, P66, PACKUSWB, 2, 2);
            sse(a, P66, MOVD_FROM, 2, R8);
        } else {
            if (dither && src_y < 0) {
                MOV(RAX, RSI);
                rm(a, 0x03, RAX, CTX, NO_INDEX, CTXF(dither_x), false, false);
                alu_ri(a, 4, RAX, 3);
                shift_ri(a, 4, RAX, 4, false);
                rm(a, 0x03, RAX, CTX, NO_INDEX, CTXF(dither_row), true, false);
                sse_m(a, P66, PADDW, 2, RAX, NO_INDEX, 0, 0);
                sse_rip(a, P66, PMINSW, 2, kc->k255);
            } else if (dither) {
                MOV(RAX, RSI);
                rm(a, 0x03, RAX, CTX, NO_INDEX, CTXF(dither_x), false, false);
                alu_ri(a, 4, RAX, 3);
                MOV(RCX, RDI);
                rm(a, 0x03, RCX, CTX, NO_INDEX, CTXF(dither_y), false, false);
                alu_ri(a, 4, RCX, 3);
                rmx(a, 0x8d, RAX, RAX, RCX, 2, 0, false, false);
                shift_ri(a, 4, RAX, 4, false);
                rex(a, true, RCX, 0, 0, false);
                b8(a, 0x8d);
                b8(a, ((RCX & 7) << 3) | 5);
                d32(a, (uint32_t) ((const uint8_t *) kc->dither - (a->p + 4)));
                sse_m(a, P66, PADDW, 2, RCX, RAX, 0, 0);
                sse_rip(a, P66, PMINSW, 2, kc->k255);
            }
            sse_rip(a, P66, PMULHUW, 2, kc->shift);
            sse_rip(a, P66, PMULLW, 2, kc->pos);
            sse_rip(a, P66, PMADDWD, 2, kc->ones);
            sse_i(a, P66, PSHUFD, 0, 2, 1);
            sse(a, P66, PADDD, 0, 2);
            sse(a, P66, MOVD_FROM, 0, R8);
        }
    }

    if (dst_computed) {
        LD(RAX, CTX, CTXF(dst_addr));
    } else {
        if (src_y >= 0) {
            MOV(RAX, RDI);
            rm(a, 0x0faf, RAX, CTX, NO_INDEX, CTXF(dst_stride), false, false);
            rm(a, 0x03, RAX, PE, NO_INDEX, PEF(dst_base), false, false);
        } else
            LD(RAX, CTX, CTXF(dst_row));
        if (k->dst_fmt == 12)
            rmx(a, 0x8d, RAX, RAX, RSI, 2, 0, false, false);
        else {
            rr(a, 0x01, RSI, RAX, false);
            rr(a, 0x01, RSI, RAX, false);
        }
    }
    if (k->dst_fmt == 12)
        emit_store32(a, R8, RAX, !unconditional);
    else
        emit_store16(a, R8, RAX, !unconditional, false);
    if (zwrite) {
        LD(RAX, CTX, CTXF(z_addr));
        LD(RCX, CTX, CTXF(z));
        emit_store16(a, RCX, RAX, !unconditional, true);
    }

    for (unsigned i = 0; i < nskip; i++)
        land(skips[i], a->p);
    if (q_index >= 0) {
        LD(RAX, CTX, CTXF(values[0]) + 4 * q_index);
        ST(RAX, PE, PEF(q));
    }
    if (!preclip && shadow) {
        LD(RAX, CTX, CTXF(x_step));
        rm(a, 0x01, RAX, PE, NO_INDEX, PEF(x_shadow), false, false);
    }
    for (unsigned i = 0; i < count; i++) {
        if (params[i] == P_NONE && !single)
            continue;
        LD(RAX, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
        rm(a, 0x01, RAX, CTX, NO_INDEX, CTXF(values[0]) + 4 * (int32_t) i, false, false);
    }
    if (!single) {
        if (preclip)
            rm(a, 0x03, RSI, CTX, NO_INDEX, CTXF(x_dir_step), false, false);
        rr(a, 0xff, 1, COUNT, false); /* dec ebx */
        land(jcc(a, CC_NE), loop);
    }

    if (unconditional) {
        /* Compute exact [lo, hi] once for the monotonic horizontal span. */
        if (!single)
            rm(a, 0x2b, RSI, CTX, NO_INDEX, CTXF(x_dir_step), false, false);
        LD(RAX, CTX, CTXF(x_first));
        MOV(RCX, RAX);
        rr(a, 0x39, RSI, RAX, false);
        rr(a, 0x0f4f, RAX, RSI, false); /* x_lo = min(x_first, x_last) */
        rr(a, 0x0f4c, RCX, RSI, false); /* x_hi = max(x_first, x_last) */
        LD(RDX, CTX, CTXF(dst_row));
        unsigned shift_b = k->dst_fmt == 12 ? 2 : 1;
        shift_ri(a, 4, RAX, shift_b, false);
        shift_ri(a, 4, RCX, shift_b, false);
        rr(a, 0x01, RDX, RAX, false);
        rr(a, 0x01, RDX, RCX, false);
        rr(a, 0x21, MASK, RAX, false);
        rr(a, 0x21, MASK, RCX, false);
        if (k->dst_fmt == 12) {
            alu_ri(a, 4, RAX, ~3U);
            alu_ri(a, 4, RCX, ~3U);
            alu_ri(a, 0, RCX, 2);
        }
        rr(a, 0x39, RCX, RAX, false);
        uint8_t *no_wrap = jcc(a, CC_BE);
        rr(a, 0x31, RAX, RAX, false);
        MOV(RCX, MASK);
        alu_ri(a, 1, RCX, 1);
        land(no_wrap, a->p);
        ST(RAX, CTX, CTXF(lo));
        ST(RCX, CTX, CTXF(hi));
        if (zwrite) {
            LD(RAX, CTX, CTXF(x_first));
            MOV(RCX, RAX);
            rr(a, 0x39, RSI, RAX, false);
            rr(a, 0x0f4f, RAX, RSI, false);
            rr(a, 0x0f4c, RCX, RSI, false);
            LD(RDX, CTX, CTXF(z_row));
            shift_ri(a, 4, RAX, 1, false);
            shift_ri(a, 4, RCX, 1, false);
            rr(a, 0x01, RDX, RAX, false);
            rr(a, 0x01, RDX, RCX, false);
            rr(a, 0x21, MASK, RAX, false);
            rr(a, 0x21, MASK, RCX, false);
            rr(a, 0x39, RCX, RAX, false);
            uint8_t *z_ok = jcc(a, CC_BE);
            rr(a, 0x31, RAX, RAX, false);
            MOV(RCX, MASK);
            alu_ri(a, 1, RCX, 1);
            land(z_ok, a->p);
            ST(RAX, CTX, CTXF(zlo));
            ST(RCX, CTX, CTXF(zhi));
        }
        mov_ri(a, DREW, 1);
    }

    if (preclip) {
        LD(RCX, CTX, CTXF(tail));
        rr(a, 0x85, RCX, RCX, false);
        uint8_t *no_tail = jcc(a, CC_E);
        for (unsigned i = 0; i < count; i++) {
            if (params[i] == P_NONE && !single)
                continue;
            LD(RAX, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
            rr(a, 0x0faf, RAX, RCX, false);
            rm(a, 0x01, RAX, CTX, NO_INDEX, CTXF(values[0]) + 4 * (int32_t) i, false, false);
        }
        if (q_index >= 0) {
            LD(RAX, CTX, CTXF(values[0]) + 4 * q_index);
            rm(a, 0x2b, RAX, CTX, NO_INDEX, CTXF(deltas[0]) + 4 * q_index, false, false);
            ST(RAX, PE, PEF(q));
        }
        land(no_tail, a->p);
    }

    uint8_t *set_state = jmp(a);

    /* O(1) fast-forward for rejected spans. */
    for (unsigned i = 0; i < nff; i++)
        land(ff[i], a->p);
    if (shadow) {
        rm(a, 0x0fb6, RAX, PE, NO_INDEX, PEF(x_dir), false, false);
        rr(a, 0xf7, 3, RAX, false);
        alu_ri(a, 4, RAX, 0xfffe0000);
        alu_ri(a, 0, RAX, 0x10000);
        rr(a, 0x0faf, RAX, COUNT, false);
        rm(a, 0x01, RAX, PE, NO_INDEX, PEF(x_shadow), false, false);
    }
    if (preclip_ff)
        land(preclip_ff, a->p);
    for (unsigned i = 0; i < count; i++) {
        if (params[i] == P_NONE && !single)
            continue;
        LD(RAX, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
        rr(a, 0x0faf, RAX, COUNT, false);
        rm(a, 0x01, RAX, CTX, NO_INDEX, CTXF(values[0]) + 4 * (int32_t) i, false, false);
    }
    if (q_index >= 0) {
        LD(RAX, CTX, CTXF(values[0]) + 4 * q_index);
        rm(a, 0x2b, RAX, CTX, NO_INDEX, CTXF(deltas[0]) + 4 * q_index, false, false);
        ST(RAX, PE, PEF(q));
    }

    land(set_state, a->p);
    for (unsigned i = 0; i < count; i++) {
        if (params[i] == P_NONE || params[i] == P_Q)
            continue;
        LD(RAX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
        rm(a, 0x2b, RAX, CTX, NO_INDEX, CTXF(deltas[0]) + 4 * (int32_t) i, false, false);
        emit_set(a, params[i], k->src_mode);
    }
    if (!unconditional) {
        rr(a, 0x31, DREW, DREW, false);
        LD(RAX, CTX, CTXF(lo));
        alu_ri(a, 7, RAX, UINT32_MAX);
        uint8_t *none_written = jcc(a, CC_E);
        mov_ri(a, DREW, 1);
        land(none_written, a->p);
    }

    if (single) {
        rm(a, 0x8b, RCX, CTX, NO_INDEX, CTXF(site), true, false);
        rm(a, 0x8b, RCX, RCX, NO_INDEX, (int32_t) offsetof(verite_draw_site, dest), true, false);
        rr(a, 0x85, RCX, RCX, true);
        uint8_t *no_dest = jcc(a, CC_E);
        for (unsigned i = 0; i < count; i++) {
            LD(RAX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
            ST(RAX, RCX, 4 * (int32_t) i);
        }
        land(no_dest, a->p);
        LD(RAX, CTX, CTXF(cnt));
        alu_ri(a, 5, RAX, 1);
        ST(RAX, PE, PEF(cnt));
    } else {
        rm(a, 0xc7, 0, PE, NO_INDEX, PEF(cnt), false, false);
        d32(a, 0);
    }
    rr(a, 0x85, DREW, DREW, false);
    uint8_t *none = jcc(a, CC_E);
    rm(a, 0xc6, 0, PE, NO_INDEX, PEF(pick_hit), false, false);
    b8(a, 1);
    rm(a, 0x8b, R8, CTX, NO_INDEX, CTXF(core), true, false);
    rm(a, 0xc6, 0, R8, NO_INDEX, (int32_t) offsetof(verite_core, drew), false, false);
    b8(a, 1);
    /* Mark dirty VRAM pages in core->page_gen / global_gen separately for color and Z. */
    emit_dirty_pages(a, CTXF(lo), CTXF(hi));
    if (zwrite) {
        emit_dirty_pages(a, CTXF(zlo), CTXF(zhi));
        LD(RAX, CTX, CTXF(zlo));
        LD(RCX, CTX, CTXF(zhi));
        rr(a, 0x39, RCX, RAX, false);
        uint8_t *no_z = jcc(a, CC_A);
        LD(RDX, CTX, CTXF(lo));
        rr(a, 0x39, RDX, RAX, false);
        rr(a, 0x0f42, RDX, RAX, false);
        ST(RDX, CTX, CTXF(lo));
        LD(RDX, CTX, CTXF(hi));
        rr(a, 0x39, RDX, RCX, false);
        rr(a, 0x0f47, RDX, RCX, false);
        ST(RDX, CTX, CTXF(hi));
        land(no_z, a->p);
    }
    land(none, a->p);

    land(empty, a->p);
    rm(a, 0x8b, RCX, CTX, NO_INDEX, CTXF(site), true, false);
    LD(RAX, CTX, CTXF(lo));
    ST(RAX, RCX, (int32_t) offsetof(verite_draw_site, lo));
    LD(RAX, CTX, CTXF(hi));
    ST(RAX, RCX, (int32_t) offsetof(verite_draw_site, hi));
    MOV(RAX, DREW);
    rr(a, 0x81, 0, RSP, true); /* add rsp, frame */
    d32(a, frame);
    for (int i = 7; i >= 0; i--)
        rex(a, false, 0, 0, saved[i], false), b8(a, 0x58 | (saved[i] & 7));
    b8(a, 0xc3);
    return start;
}

static uint8_t *
alloc_code(void)
{
#    ifdef _WIN32
    return VirtualAlloc(NULL, SPAN_ARENA, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
#    else
    int flags = MAP_PRIVATE | MAP_ANON;
#        if defined(__APPLE__) && defined(MAP_JIT)
    flags |= MAP_JIT;
#        endif
    void *code = mmap(NULL, SPAN_ARENA, PROT_READ | PROT_WRITE | PROT_EXEC, flags, -1, 0);
    return code == MAP_FAILED ? NULL : code;
#    endif
}

void
pixel_jit_free(verite_core *c)
{
    if (c->pixel_jit) {
#    ifdef _WIN32
        VirtualFree(c->pixel_jit->arena, 0, MEM_RELEASE);
#    else
        munmap(c->pixel_jit->arena, SPAN_ARENA);
#    endif
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
        if (!jit || !(jit->arena = alloc_code())) {
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
        /* Draw sites in RISC blocks point into the arena. */
        memset(jit->entries, 0, sizeof(jit->entries));
        jit->used  = 0;
        jit->count = 0;
        risc_jit_reset_sites(c);
    }
    as       a    = { jit->arena + jit->used };
    uint8_t *code = compile(&a, key); /* after its constants */
    size_t   size = (size_t) (a.p - (jit->arena + jit->used));
    if (size > SPAN_MAX_CODE)
        abort();
#    ifdef _WIN32
    FlushInstructionCache(GetCurrentProcess(), jit->arena + jit->used, size);
#    endif
    jit->used += (size + 15) & ~(size_t) 15;
    idx = span_hash(key) & (SPAN_CACHE - 1);
    while (jit->entries[idx].used)
        idx = (idx + 1) & (SPAN_CACHE - 1);
    jit->entries[idx] = (span_entry) { *key, (span_fn) (void *) code, true };
    jit->count++;
    return jit->entries[idx].fn;
}

/* The compiled span for the current render state, or NULL to use C. */
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
