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
    SPAN_CACHE      = 64,
    SPAN_ARENA      = 1024 * 1024,
    SPAN_MAX_CODE   = 12 * 1024,
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

/* The span's stack frame, addressed through R15. */
typedef struct {
    verite_core      *core;
    verite_draw_site *site;
    int32_t           cnt;
    uint32_t          x_step, dst_stride, z_stride, tex_stride;
    uint32_t          values[10], deltas[10];
    int32_t           c[4], t[4], tap[4], sum[4], w[4];
    uint32_t          z_addr, z;
    uint32_t          row0, row1, x0, x1;
    uint32_t          lo, hi;
    uint32_t          prev_u, prev_v; /* the last pixel's texel coordinates */
    int32_t           scissor[4];     /* x lo, x hi, y lo, y hi */
} span_ctx;

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
   decoded as texel() does and packed for the bilinear filter: entry 2n is
   R | B << 32 and entry 2n + 1 is G | A << 32. Built on first use. */
static uint64_t *packed_tables[2][2];

static const uint64_t *
packed_table(unsigned fmt, bool no_pad)
{
    uint64_t **table = &packed_tables[fmt == 6][no_pad];
    if (*table)
        return *table;
    uint64_t *t = malloc(65536 * 2 * sizeof(uint64_t));
    if (!t)
        abort();
    static const unsigned shift565[] = { 11, 5, 0 }, bits565[] = { 5, 6, 5 }, shift1555[] = { 10, 5, 0 };
    for (unsigned texel = 0; texel < 65536; texel++) {
        uint64_t ch[4];
        for (unsigned k = 0; k < 3; k++) {
            unsigned shift = fmt == 4 ? shift565[k] : shift1555[k];
            unsigned bits  = fmt == 4 ? bits565[k] : 5;
            unsigned v     = (texel >> shift) & ((1U << bits) - 1);
            ch[k]          = no_pad ? v << (8 - bits) : widen_table[bits - 4][v];
        }
        ch[3]            = fmt == 4 || (texel & 0x8000) ? 255 : 0;
        t[2 * texel]     = ch[0] | ch[2] << 32;
        t[2 * texel + 1] = ch[1] | ch[3] << 32;
    }
    return *table = t;
}

/* ---- Assembler ------------------------------------------------------------ */

enum { RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15 };
enum { CTX = R15, VRAM = R14, MASK = R13, PE = R12, COUNT = RBX, DREW = RBP };
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

/* op reg, [base + index + disp32]. byte forces a REX so SIL/DIL can be used. */
static void
rm(as *a, unsigned op, unsigned reg, unsigned base, int index, int32_t disp, bool w, bool byte)
{
    unsigned ix = index < 0 ? 4 : (unsigned) index;
    rex(a, w, reg, index < 0 ? 0 : ix, base, byte && reg >= 4);
    opcode(a, op);
    if (index < 0 && (base & 7) != 4) {
        b8(a, 0x80 | ((reg & 7) << 3) | (base & 7));
    } else {
        b8(a, 0x80 | ((reg & 7) << 3) | 4);
        b8(a, ((ix & 7) << 3) | (base & 7));
    }
    d32(a, (uint32_t) disp);
}

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

/* reg = clamp(reg, 0, 255) is unnecessary: every supported source is a byte. */

/* dst = 16-bit VRAM word at the address in reg (clobbers reg). */
static void
emit_load16(as *a, unsigned dst, unsigned addr)
{
    rr(a, 0x21, MASK, addr, false); /* and addr, mask */
    alu_ri(a, 4, addr, ~1U);
    alu_ri(a, 6, addr, 2);
    rm(a, 0x0fb7, dst, VRAM, (int) addr, 0, false, false);
}

/* Store the low 16 bits of val at the address in reg (clobbers reg), widening
   the written range in ctx.lo/hi. */
static void
emit_store16(as *a, unsigned val, unsigned addr)
{
    rr(a, 0x21, MASK, addr, false);
    alu_ri(a, 4, addr, ~1U);
    rm(a, 0x3b, addr, CTX, NO_INDEX, (int32_t) offsetof(span_ctx, lo), false, false); /* cmp addr, lo */
    uint8_t *above = jcc(a, CC_AE);
    rm(a, 0x89, addr, CTX, NO_INDEX, (int32_t) offsetof(span_ctx, lo), false, false);
    land(above, a->p);
    rm(a, 0x3b, addr, CTX, NO_INDEX, (int32_t) offsetof(span_ctx, hi), false, false);
    uint8_t *below = jcc(a, CC_BE);
    rm(a, 0x89, addr, CTX, NO_INDEX, (int32_t) offsetof(span_ctx, hi), false, false);
    land(below, a->p);
    alu_ri(a, 6, addr, 2);
    b8(a, 0x66);
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

/* ECX = widened channel of EAX: (EAX >> shift) & ((1 << bits) - 1), expanded to 8 bits. */
static void
emit_widen(as *a, unsigned shift, unsigned bits, bool no_pad)
{
    MOV(RCX, RAX);
    shift_ri(a, 5, RCX, shift, false);
    alu_ri(a, 4, RCX, (1U << bits) - 1);
    if (no_pad)
        shift_ri(a, 4, RCX, 8 - bits, false);
    else {
        mov_ri64(a, RDX, (uint64_t) (uintptr_t) widen_table[bits - 4]);
        rm(a, 0x0fb6, RCX, RDX, RCX, 0, false, false); /* movzx ecx, byte [rdx + rcx] */
    }
}

/* R8 = R | B << 32 and R9 = G | A << 32 for the 16-bit texel at row (ctx
   field) + x (ctx field) or, keyed, black or YUV black (128, 16, 128, 0).
   RDX holds packed_table(). Clobbers RAX, RCX. */
static void
emit_texel_packed(as *a, uint32_t src_mode, int32_t row, int32_t x)
{
    LD(RAX, CTX, x);
    rr(a, 0x01, RAX, RAX, false); /* add eax, eax */
    rm(a, 0x03, RAX, CTX, NO_INDEX, row, false, false);
    emit_load16(a, RAX, RAX);
    uint8_t *decode = NULL, *done = NULL;
    if (src_mode & SRC_CHROMA) {
        rr(a, 0x69, RCX, RAX, false); /* imul ecx, eax, 0x10001 */
        d32(a, 0x10001);
        rm(a, 0x33, RCX, PE, NO_INDEX, PEF(chroma_color), false, false);
        rm(a, 0x23, RCX, PE, NO_INDEX, PEF(chroma_mask), false, false);
        decode = jcc(a, CC_NE);
        if (src_mode & SRC_CHROMA_BLACK_YUV) {
            mov_ri64(a, R8, UINT64_C(128) | UINT64_C(128) << 32);
            mov_ri(a, R9, 16);
        } else {
            rr(a, 0x31, R8, R8, false);
            rr(a, 0x31, R9, R9, false);
        }
        done = jmp(a);
        land(decode, a->p);
    }
    shift_ri(a, 4, RAX, 4, false);             /* 16-byte entries */
    rm(a, 0x8b, R8, RDX, RAX, 0, true, false); /* mov r8, [rdx + rax] */
    rm(a, 0x8b, R9, RDX, RAX, 8, true, false); /* mov r9, [rdx + rax + 8] */
    if (done)
        land(done, a->p);
}

/* Prefetch the texel 1 << PREFETCH_SHIFT pixels on, extrapolated from how far u
   and v moved since the last pixel. Perspective keeps that step close to
   constant over a few pixels, and a wrong guess only costs the hint.
   R10D = u, R11D = v. Clobbers EAX, ECX. */
static void
emit_prefetch(as *a)
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
    rr(a, 0x01, RAX, RCX, false);
    rr(a, 0x01, RAX, RCX, false);
    rr(a, 0x21, MASK, RCX, false);
    rm(a, 0x0f18, 1, VRAM, RCX, 0, false, false); /* prefetcht0 [vram + rcx] */
}

/* sample() into ctx.t. R10D = u, R11D = v after any perspective division. */
static void
emit_sample(as *a, unsigned fmt, uint32_t src_mode)
{
    bool uc = !!(src_mode & SRC_U_CLAMP), vc = !!(src_mode & SRC_V_CLAMP);
    bool filter = !!(src_mode & SRC_FILTER);
    emit_prefetch(a);

    /* x0, row0 */
    MOV(RAX, R10);
    shift_ri(a, 7, RAX, 16, false);
    emit_coord(a, PEF(u_mask), uc);
    ST(RAX, CTX, CTXF(x0));
    MOV(RAX, R11);
    shift_ri(a, 7, RAX, 16, false);
    emit_coord(a, PEF(v_mask), vc);
    rm(a, 0x0faf, RAX, CTX, NO_INDEX, CTXF(tex_stride), false, false);
    rm(a, 0x03, RAX, PE, NO_INDEX, PEF(src_base), false, false);
    ST(RAX, CTX, CTXF(row0));
    mov_ri64(a, RDX, (uint64_t) (uintptr_t) packed_table(fmt, !!(src_mode & SRC_NO_PAD)));
    if (!filter) {
        emit_texel_packed(a, src_mode, CTXF(row0), CTXF(x0));
        ST(R8, CTX, CTXF(t[0]));
        ST(R9, CTX, CTXF(t[1]));
        shift_ri(a, 5, R8, 32, true);
        shift_ri(a, 5, R9, 32, true);
        ST(R8, CTX, CTXF(t[2]));
        ST(R9, CTX, CTXF(t[3]));
        return;
    }

    /* x1, row1 */
    MOV(RAX, R10);
    shift_ri(a, 7, RAX, 16, false);
    alu_ri(a, 0, RAX, 1);
    emit_coord(a, PEF(u_mask), uc);
    ST(RAX, CTX, CTXF(x1));
    MOV(RAX, R11);
    shift_ri(a, 7, RAX, 16, false);
    alu_ri(a, 0, RAX, 1);
    emit_coord(a, PEF(v_mask), vc);
    rm(a, 0x0faf, RAX, CTX, NO_INDEX, CTXF(tex_stride), false, false);
    rm(a, 0x03, RAX, PE, NO_INDEX, PEF(src_base), false, false);
    ST(RAX, CTX, CTXF(row1));

    /* Weights: fu = (u >> 8) & 255 in ESI-free registers R8D/R9D. */
    MOV(R8, R10);
    shift_ri(a, 5, R8, 8, false);
    alu_ri(a, 4, R8, 255);
    MOV(R9, R11);
    shift_ri(a, 5, R9, 8, false);
    alu_ri(a, 4, R9, 255);
    mov_ri(a, RAX, 256);
    rr(a, 0x29, R8, RAX, false); /* eax = 256 - fu */
    mov_ri(a, RCX, 256);
    rr(a, 0x29, R9, RCX, false); /* ecx = 256 - fv */
    MOV(RDX, RAX);
    rr(a, 0x0faf, RDX, RCX, false); /* (256-fu)(256-fv) */
    ST(RDX, CTX, CTXF(w[0]));
    MOV(RDX, R8);
    rr(a, 0x0faf, RDX, RCX, false); /* fu(256-fv) */
    ST(RDX, CTX, CTXF(w[1]));
    rr(a, 0x0faf, RAX, R9, false); /* (256-fu)fv */
    ST(RAX, CTX, CTXF(w[2]));
    MOV(RAX, R8);
    rr(a, 0x0faf, RAX, R9, false); /* fu fv */
    ST(RAX, CTX, CTXF(w[3]));

    static const int32_t rows[] = { CTXF(row0), CTXF(row0), CTXF(row1), CTXF(row1) };
    static const int32_t xs[]   = { CTXF(x0), CTXF(x1), CTXF(x0), CTXF(x1) };

    /* Four taps from the packed decode table, summed two channels to a
       register: R10 = R | B << 32, R11 = G | A << 32. The weights total 65536,
       so no lane passes 2^24 and lanes never carry into each other. For an
       unpadded 565 texel this also matches sample_bilinear_565()'s rounding:
       (8 sum + 32768) >> 16 == (sum + 4096) >> 13, and likewise for green. */
    mov_ri64(a, RDX, (uint64_t) (uintptr_t) packed_table(fmt, !!(src_mode & SRC_NO_PAD)));
    rr(a, 0x31, R10, R10, false);
    rr(a, 0x31, R11, R11, false);
    for (unsigned tap = 0; tap < 4; tap++) {
        emit_texel_packed(a, src_mode, rows[tap], xs[tap]);
        LD(RCX, CTX, CTXF(w[0]) + 4 * (int32_t) tap);
        rr(a, 0x0faf, R8, RCX, true);  /* imul r8, rcx */
        rr(a, 0x0faf, R9, RCX, true);  /* imul r9, rcx */
        rr(a, 0x01, R8, R10, true);    /* add r10, r8 */
        rr(a, 0x01, R9, R11, true);    /* add r11, r9 */
    }
    mov_ri64(a, RAX, UINT64_C(0x0000800000008000));
    rr(a, 0x01, RAX, R10, true);
    rr(a, 0x01, RAX, R11, true);
    for (unsigned k = 0; k < 4; k++) {
        static const int32_t out[] = { CTXF(t[0]), CTXF(t[2]), CTXF(t[1]), CTXF(t[3]) };
        rr(a, 0x89, k < 2 ? R10 : R11, RAX, true); /* mov rax, sum */
        shift_ri(a, 5, RAX, k & 1 ? 48 : 16, true);
        if (!(k & 1))
            alu_ri(a, 4, RAX, 255);
        ST(RAX, CTX, out[k]);
    }
}

/* R10D, R11D = u, v * 2^24 / q unless Q is 0 (exact truncating division).
   As perspective_uv() does, Q >= 64 multiplies by a double-precision 2^24 / q
   and corrects the truncated estimate by its remainder; two 64-bit idivs a
   pixel cost several times as much. Clobbers RAX, RCX, RDX, R8, XMM0-XMM1. */
static void
emit_divide_uv(as *a)
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
    rr(a, 0x0fef, 1, 1, false);   /* pxor xmm1, xmm1 */
    b8(a, 0xf2);
    rr(a, 0x0f2a, 1, R8, true);   /* cvtsi2sd xmm1, r8 */
    mov_ri64(a, RAX, UINT64_C(0x4170000000000000)); /* 2^24 */
    b8(a, 0x66);
    rr(a, 0x0f6e, 0, RAX, true);  /* movq xmm0, rax */
    b8(a, 0xf2);
    rr(a, 0x0f5e, 0, 1, false);   /* divsd xmm0, xmm1 */
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
        rr(a, 0x85, coord, coord, false);
        uint8_t *negative = jcc(a, CC_S);
        /* estimate += (remainder >= q) - (remainder < 0) */
        rr(a, 0x39, R8, RAX, true);     /* cmp rax, r8 */
        uint8_t *below = jcc(a, CC_L);
        rr(a, 0xff, 0, RCX, true);      /* inc rcx */
        land(below, a->p);
        rr(a, 0x85, RAX, RAX, true);
        uint8_t *nonneg = jcc(a, CC_NS);
        rr(a, 0xff, 1, RCX, true);      /* dec rcx */
        land(nonneg, a->p);
        uint8_t *next = jmp(a);
        /* estimate += (remainder > 0) - (remainder <= -q) */
        land(negative, a->p);
        rr(a, 0x85, RAX, RAX, true);
        uint8_t *not_above = jcc(a, CC_LE);
        rr(a, 0xff, 0, RCX, true);
        land(not_above, a->p);
        rr(a, 0x01, R8, RAX, true);     /* add rax, r8: remainder + q <= 0 */
        uint8_t *above = jcc(a, CC_G);
        rr(a, 0xff, 1, RCX, true);
        land(above, a->p);
        land(next, a->p);
        MOV(coord, RCX);
    }
    land(done, a->p);
    land(flat, a->p);
}

/* EAX = (EAX + 127) / 255 for EAX + 127 < 65536. */
static void
emit_div255(as *a)
{
    alu_ri(a, 0, RAX, 127);
    rr(a, 0x69, RAX, RAX, false);
    d32(a, 0x8081);
    shift_ri(a, 5, RAX, 23, false);
}

/* ECX = 255 - [CTX + field] */
static void
emit_inverse(as *a, int32_t field)
{
    mov_ri(a, RCX, 255);
    rm(a, 0x2b, RCX, CTX, NO_INDEX, field, false, false);
}

/* ECX = factor(func, source, k, s = ctx.c, d = ctx.tap), as in pixel.c. */
static void
emit_factor(as *a, unsigned func, bool source, unsigned k)
{
    int32_t s = CTXF(c[0]) + 4 * (int32_t) k, d = CTXF(tap[0]) + 4 * (int32_t) k;
    switch (func) {
        case 0:
            LD(RCX, CTX, source ? d : s);
            break;
        case 1:
            emit_inverse(a, source ? d : s);
            break;
        case 2:
            LD(RCX, CTX, CTXF(c[3]));
            break;
        case 3:
            emit_inverse(a, CTXF(c[3]));
            break;
        case 4:
            LD(RCX, CTX, CTXF(tap[3]));
            break;
        case 5:
            emit_inverse(a, CTXF(tap[3]));
            break;
        case 6:
            rr(a, 0x31, RCX, RCX, false);
            break;
        case 8:
        case 9:
            if (k == 3)
                mov_ri(a, RCX, 255);
            else {
                /* min(s.a, 255 - d.a); both are bytes. */
                emit_inverse(a, CTXF(tap[3]));
                LD(RDX, CTX, CTXF(c[3]));
                rr(a, 0x39, RCX, RDX, false);   /* cmp edx, ecx */
                rr(a, 0x0f4c, RCX, RDX, false); /* cmovl ecx, edx */
            }
            if (func == 9) {
                rr(a, 0xf7, 3, RCX, false); /* neg ecx */
                alu_ri(a, 0, RCX, 255);
            }
            break;
        case 10:
            LD(RCX, CTX, source ? s : d);
            break;
        case 11:
            emit_inverse(a, source ? s : d);
            break;
        default:
            mov_ri(a, RCX, 255);
            break;
    }
}

/* EAX = one side of blend(): the channel scaled by its factor, or less 128. */
static void
emit_blend_term(as *a, unsigned func, bool source, unsigned k)
{
    int32_t own = source ? CTXF(c[0]) + 4 * (int32_t) k : CTXF(tap[0]) + 4 * (int32_t) k;
    if (func == 15) {
        LD(RAX, CTX, own);
        alu_ri(a, 5, RAX, 128);
        return;
    }
    emit_factor(a, func, source, k);
    LD(RAX, CTX, own);
    rr(a, 0x0faf, RAX, RCX, false);
    emit_div255(a);
}

/* c = blend(c, d), d from DstColor or the decoded destination pixel. ESI = x,
   EDI = y. Clobbers EAX-EDX, R10D. */
static void
emit_blend(as *a, const span_key *k)
{
    if (k->dst_mode & DST_RAW_READ_DISABLE) {
        LD(RAX, PE, PEF(dst_color));
        for (unsigned ch = 0; ch < 4; ch++) {
            static const unsigned sh[] = { 16, 8, 0, 24 };
            MOV(RCX, RAX);
            shift_ri(a, 5, RCX, sh[ch], false);
            alu_ri(a, 4, RCX, 255);
            ST(RCX, CTX, CTXF(tap[0]) + 4 * (int32_t) ch);
        }
    } else {
        bool no_pad = !!(k->dst_mode & DST_RAW_NO_PAD);
        MOV(RAX, RDI);
        rm(a, 0x0faf, RAX, CTX, NO_INDEX, CTXF(dst_stride), false, false);
        rm(a, 0x03, RAX, PE, NO_INDEX, PEF(dst_base), false, false);
        rr(a, 0x01, RSI, RAX, false);
        rr(a, 0x01, RSI, RAX, false);
        emit_load16(a, RAX, RAX);
        static const unsigned shift565[] = { 11, 5, 0 }, bits565[] = { 5, 6, 5 }, shift1555[] = { 10, 5, 0 };
        for (unsigned ch = 0; ch < 3; ch++) {
            if (k->dst_fmt == 4)
                emit_widen(a, shift565[ch], bits565[ch], no_pad);
            else
                emit_widen(a, shift1555[ch], 5, no_pad);
            ST(RCX, CTX, CTXF(tap[0]) + 4 * (int32_t) ch);
        }
        if (k->dst_fmt == 4)
            mov_ri(a, RCX, 255);
        else {
            MOV(RCX, RAX);
            shift_ri(a, 5, RCX, 15, false);
            alu_ri(a, 4, RCX, 1);
            rr(a, 0xf7, 3, RCX, false); /* neg ecx */
            alu_ri(a, 4, RCX, 255);
        }
        ST(RCX, CTX, CTXF(tap[3]));
    }
    unsigned sf = (k->dst_mode >> 4) & 15, df = (k->dst_mode >> 8) & 15;
    for (unsigned ch = 0; ch < 4; ch++) {
        emit_blend_term(a, sf, true, ch);
        MOV(R10, RAX);
        emit_blend_term(a, df, false, ch);
        rr(a, 0x01, R10, RAX, false);
        rr(a, 0x31, RCX, RCX, false);
        rr(a, 0x85, RAX, RAX, false);
        rr(a, 0x0f4c, RAX, RCX, false); /* cmovl eax, 0 */
        mov_ri(a, RCX, 255);
        rr(a, 0x39, RCX, RAX, false);
        rr(a, 0x0f4f, RAX, RCX, false); /* cmovg eax, 255 */
        ST(RAX, CTX, CTXF(sum[0]) + 4 * (int32_t) ch);
    }
    for (unsigned ch = 0; ch < 4; ch++) {
        LD(RAX, CTX, CTXF(sum[0]) + 4 * (int32_t) ch);
        ST(RAX, CTX, CTXF(c[0]) + 4 * (int32_t) ch);
    }
}

/* ECX = quantize(channel in EAX, n bits) with the dither term (d << 4) >> n in R9D. */
static void
emit_quantize(as *a, unsigned n, bool dither)
{
    MOV(RCX, RAX);
    if (dither) {
        MOV(RDX, R9);
        shift_ri(a, 5, RDX, n, false);
        rr(a, 0x01, RDX, RCX, false);
        mov_ri(a, RDX, 255);
        rr(a, 0x39, RDX, RCX, false);   /* cmp ecx, edx */
        rr(a, 0x0f47, RCX, RDX, false); /* cmova ecx, edx */
    }
    shift_ri(a, 5, RCX, 8 - n, false);
}

/* c = f * (c - fog) / 255 + fog for red, green and blue, with f = sat(F)
   and C's truncating division. Clobbers EAX, ECX, EDX, R8D, R10D, R11D. */
static void
emit_fog(as *a)
{
    LD(RAX, PE, PEF(f));
    emit_sat(a);
    MOV(R10, RAX);
    LD(R11, PE, PEF(fog_color));
    for (unsigned ch = 0; ch < 3; ch++) {
        static const unsigned sh[] = { 16, 8, 0 };
        int32_t c = CTXF(c[0]) + 4 * (int32_t) ch;
        MOV(R8, R11);
        shift_ri(a, 5, R8, sh[ch], false);
        alu_ri(a, 4, R8, 255);
        LD(RAX, CTX, c);
        rr(a, 0x29, R8, RAX, false);     /* sub eax, r8d */
        rr(a, 0x0faf, RAX, R10, false);  /* imul eax, r10d */
        /* |product| <= 255 * 255, where (n * 0x8081) >> 23 == n / 255. */
        b8(a, 0x99);                     /* cdq */
        rr(a, 0x31, RDX, RAX, false);    /* xor eax, edx */
        rr(a, 0x29, RDX, RAX, false);    /* sub eax, edx */
        rr(a, 0x69, RAX, RAX, false);    /* imul eax, eax, 0x8081 */
        d32(a, 0x8081);
        shift_ri(a, 5, RAX, 23, false);
        rr(a, 0x31, RDX, RAX, false);
        rr(a, 0x29, RDX, RAX, false);
        rr(a, 0x01, R8, RAX, false);     /* add eax, r8d */
        ST(RAX, CTX, c);
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

/* ctx.field = line_stride(pe->stride >> shift). */
static void
emit_stride(as *a, unsigned shift, int32_t field)
{
    LD(RAX, PE, PEF(stride));
    shift_ri(a, 5, RAX, shift, false);
    MOV(RCX, RAX);
    shift_ri(a, 5, RCX, 4, false);
    alu_ri(a, 4, RCX, 7);
    shift_ri(a, 4, RCX, 2, false);
    alu_ri(a, 4, RAX, 7);
    shift_ri(a, 4, RAX, 2, false);
    mov_ri64(a, RDX, (uint64_t) (uintptr_t) stride_high);
    rm(a, 0x8b, RCX, RDX, RCX, 0, false, false);
    mov_ri64(a, RDX, (uint64_t) (uintptr_t) stride_low);
    rm(a, 0x03, RCX, RDX, RAX, 0, false, false);
    ST(RCX, CTX, field);
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

/* Generates a span_fn. op holds the register count, plus 0x100 for a single
   draw that writes back its stepped values and decrements Cnt. */
static uint8_t *
compile(as *a, const span_key *k)
{
    uint8_t     *start = a->p;
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
    for (unsigned i = 0; i < count; i++) {
        if (params[i] == P_Q)
            q_index = (int) i;
        if (params[i] == P_X)
            shadow = false;
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

    /* Spans call nothing, so the frame needs no particular alignment. */
    static const unsigned saved[] = { RBX, RBP, RSI, RDI, R12, R13, R14, R15 };
    uint32_t frame = (sizeof(span_ctx) + 15) & ~15U;
    for (unsigned i = 0; i < 8; i++)
        rex(a, false, 0, 0, saved[i], false), b8(a, 0x50 | (saved[i] & 7));
    rr(a, 0x81, 5, RSP, true); /* sub rsp, frame */
    d32(a, frame);
    rr(a, 0x89, RSP, CTX, true);
#    ifdef _WIN32
    unsigned arg_core = RCX, arg_site = RDX;
#    else
    unsigned arg_core = RDI, arg_site = RSI;
#    endif
    rm(a, 0x89, arg_core, CTX, NO_INDEX, CTXF(core), true, false);
    rm(a, 0x89, arg_site, CTX, NO_INDEX, CTXF(site), true, false);
    rm(a, 0x8d, PE, arg_core, NO_INDEX, (int32_t) offsetof(verite_core, pe), true, false); /* lea */
    rm(a, 0x8b, RAX, arg_core, NO_INDEX, (int32_t) offsetof(verite_core, jit_mem), true, false);
    rm(a, 0x8b, VRAM, RAX, NO_INDEX, (int32_t) offsetof(verite_jit_memory, vram), true, false);
    LD(MASK, RAX, (int32_t) offsetof(verite_jit_memory, mask));
    rr(a, 0x31, DREW, DREW, false);
    rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(lo), false, false);
    d32(a, UINT32_MAX);
    rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(hi), false, false);
    d32(a, 0);

    /* cnt = (int16_t) Cnt; nothing happens unless it is positive. */
    rm(a, 0x0fbf, RAX, PE, NO_INDEX, PEF(cnt), false, false); /* movsx eax, word [cnt] */
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

    if (shadow) {
        /* x_step = x_dir ? -1.0 : 1.0 in 16.16 */
        rm(a, 0x0fb6, RAX, PE, NO_INDEX, PEF(x_dir), false, false);
        rr(a, 0xf7, 3, RAX, false); /* neg eax */
        alu_ri(a, 4, RAX, 0xfffe0000);
        alu_ri(a, 0, RAX, 0x10000);
        ST(RAX, CTX, CTXF(x_step));
    }
    emit_stride(a, 8, CTXF(dst_stride));
    if (depth_mode || zwrite)
        emit_stride(a, 16, CTXF(z_stride));
    if (func) {
        emit_stride(a, 0, CTXF(tex_stride));
        /* The first pixel has no step to extrapolate; its prefetch is a guess. */
        rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(prev_u), false, false);
        d32(a, 0);
        rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(prev_v), false, false);
        d32(a, 0);
    }

    /* Loop-invariant state: U and V masks, and the scissor bounds
       [lo, min(hi, 2048)) for each axis. */
    for (unsigned i = 0; i < count; i++)
        if (params[i] == P_U || params[i] == P_V)
            emit_uv_mask(a, params[i], k->src_mode);
    for (unsigned axis = 0; axis < 2; axis++) {
        LD(RAX, PE, axis ? PEF(scissor_y) : PEF(scissor_x));
        MOV(RCX, RAX);
        shift_ri(a, 5, RCX, 16, false);
        alu_ri(a, 4, RCX, 0xfff);
        ST(RCX, CTX, CTXF(scissor[0]) + 8 * (int32_t) axis);
        alu_ri(a, 4, RAX, 0xfff);
        mov_ri(a, RCX, 2048);
        rr(a, 0x39, RCX, RAX, false);   /* cmp eax, ecx */
        rr(a, 0x0f4f, RAX, RCX, false); /* cmovg eax, ecx */
        ST(RAX, CTX, CTXF(scissor[1]) + 8 * (int32_t) axis);
    }

    uint8_t *loop = a->p;
    for (unsigned i = 0; i < count; i++) {
        if (params[i] == P_NONE || params[i] == P_Q)
            continue;
        LD(RAX, CTX, CTXF(values[0]) + 4 * (int32_t) i);
        emit_set(a, params[i], k->src_mode);
    }

    /* ESI = x, EDI = y; scissor. */
    uint8_t *skips[16];
    unsigned nskip = 0;
    LD(RSI, PE, shadow ? PEF(x_shadow) : PEF(x));
    shift_ri(a, 7, RSI, 16, false);
    LD(RDI, PE, PEF(y));
    shift_ri(a, 7, RDI, 16, false);
    for (unsigned axis = 0; axis < 2; axis++) {
        unsigned r = axis ? RDI : RSI;
        rm(a, 0x3b, r, CTX, NO_INDEX, CTXF(scissor[0]) + 8 * (int32_t) axis, false, false); /* cmp r, lo */
        skips[nskip++] = jcc(a, CC_L);
        rm(a, 0x3b, r, CTX, NO_INDEX, CTXF(scissor[1]) + 8 * (int32_t) axis, false, false); /* cmp r, hi */
        skips[nskip++] = jcc(a, CC_GE);
    }
    if (k->src_mode & SRC_Z_SCISSOR) {
        rm(a, 0x80, 7, PE, NO_INDEX, PEF(z_ovf), false, false); /* cmp byte [z_ovf], 0 */
        b8(a, 0);
        skips[nskip++] = jcc(a, CC_NE);
    }

    if (depth_mode || zwrite) {
        MOV(RAX, RDI);
        rm(a, 0x0faf, RAX, CTX, NO_INDEX, CTXF(z_stride), false, false);
        rm(a, 0x03, RAX, PE, NO_INDEX, PEF(z_base), false, false);
        rr(a, 0x01, RSI, RAX, false);
        rr(a, 0x01, RSI, RAX, false);
        ST(RAX, CTX, CTXF(z_addr));
        LD(RCX, PE, PEF(z));
        shift_ri(a, 7, RCX, 11, false);
        alu_ri(a, 4, RCX, 0xffff);
        ST(RCX, CTX, CTXF(z));
        if (depth_mode == 7)
            skips[nskip++] = jmp(a);
        else if (depth_mode) {
            static const uint8_t fail[] = { 0, CC_AE, CC_NE, CC_A, CC_BE, CC_E, CC_B };
            emit_load16(a, RDX, RAX);
            rr(a, 0x39, RDX, RCX, false); /* cmp z, old */
            skips[nskip++] = jcc(a, fail[depth_mode]);
        }
    }

    if (raw_copy) {
        LD(R10, PE, PEF(u));
        LD(R11, PE, PEF(v));
    }
    if (raw_copy && q_index >= 0)
        emit_divide_uv(a);
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
        uint8_t *opaque = NULL;
        if (k->src_mode & SRC_CHROMA) {
            rr(a, 0x69, RCX, R8, false); /* imul ecx, r8d, 0x10001 */
            d32(a, 0x10001);
            rm(a, 0x33, RCX, PE, NO_INDEX, PEF(chroma_color), false, false);
            rm(a, 0x23, RCX, PE, NO_INDEX, PEF(chroma_mask), false, false);
            opaque = jcc(a, CC_NE);
            /* Keyed: alpha 0 is always rejected; otherwise it encodes as 0. */
            if (k->dst_mode & DST_RAW_TREJECT)
                skips[nskip++] = jmp(a);
            else
                rr(a, 0x31, R8, R8, false);
            uint8_t *keyed = jmp(a);
            land(opaque, a->p);
            opaque = keyed;
        }
        if (k->dst_mode & DST_RAW_TREJECT) {
            /* Alpha 255 is rejected only by a threshold of 255. */
            LD(RCX, PE, PEF(alpha_thres));
            alu_ri(a, 4, RCX, 255);
            alu_ri(a, 7, RCX, 255);
            skips[nskip++] = jcc(a, CC_E);
        }
        if (opaque)
            land(opaque, a->p);
    }

    /* c = argb(fg), unless a replaced texel overwrites all of it. */
    if (!raw_copy) {
    LD(RAX, PE, PEF(fg));
    for (unsigned ch = 0; ch < 4 && func != 1; ch++) {
        static const unsigned sh[] = { 16, 8, 0, 24 };
        MOV(RCX, RAX);
        shift_ri(a, 5, RCX, sh[ch], false);
        alu_ri(a, 4, RCX, 255);
        ST(RCX, CTX, CTXF(c[0]) + 4 * (int32_t) ch);
    }

    if (func) {
        LD(R10, PE, PEF(u));
        LD(R11, PE, PEF(v));
        if (q_index >= 0)
            emit_divide_uv(a);
        emit_sample(a, fmt, k->src_mode);
        for (unsigned ch = 0; ch < 4; ch++) {
            int32_t c = CTXF(c[0]) + 4 * (int32_t) ch, t = CTXF(t[0]) + 4 * (int32_t) ch;
            if (func == 1) {
                LD(RAX, CTX, t);
                ST(RAX, CTX, c);
            } else if (func == 3) {
                LD(RAX, CTX, c);
                rm(a, 0x0faf, RAX, CTX, NO_INDEX, t, false, false);
                emit_div255(a);
                ST(RAX, CTX, c);
            } else if (ch < 3) {
                /* Decal: c * (255 - ta) + t * ta. */
                LD(RCX, CTX, CTXF(t[3]));
                LD(RAX, CTX, t);
                rr(a, 0x0faf, RAX, RCX, false);
                mov_ri(a, RDX, 255);
                rr(a, 0x29, RCX, RDX, false);
                rm(a, 0x0faf, RDX, CTX, NO_INDEX, c, false, false);
                rr(a, 0x01, RDX, RAX, false);
                emit_div255(a);
                ST(RAX, CTX, c);
            }
        }
    }

    if (k->dst_mode & DST_RAW_TREJECT) {
        LD(RAX, CTX, CTXF(c[3]));
        LD(RCX, PE, PEF(alpha_thres));
        alu_ri(a, 4, RCX, 255);
        rr(a, 0x39, RCX, RAX, false);
        skips[nskip++] = jcc(a, CC_LE);
    }

    if (k->dst_mode & DST_RAW_FOG)
        emit_fog(a);

    if (k->dst_mode & DST_RAW_BLEND)
        emit_blend(a, k);

    /* Dither term d << 4 in R9D. */
    if (dither) {
        LD(RDX, PE, PEF(dither_offset));
        MOV(RAX, RDX);
        alu_ri(a, 4, RAX, 3);
        rr(a, 0x01, RSI, RAX, false);
        alu_ri(a, 4, RAX, 3);
        shift_ri(a, 5, RDX, 4, false);
        alu_ri(a, 4, RDX, 3);
        rr(a, 0x01, RDI, RDX, false);
        alu_ri(a, 4, RDX, 3);
        shift_ri(a, 4, RDX, 2, false);
        rr(a, 0x01, RDX, RAX, false); /* index = dy * 4 + dx */
        mov_ri64(a, RDX, (uint64_t) (uintptr_t) dither_table);
        rm(a, 0x0fb6, R9, RDX, RAX, 0, false, false);
        shift_ri(a, 4, R9, 4, false);
    }

    }

    /* Encode into R8D. */
    if (!raw_copy) {
    rr(a, 0x31, R8, R8, false);
    static const struct {
        unsigned ch, n, pos;
    } fields565[] = { { 0, 5, 11 }, { 1, 6, 5 }, { 2, 5, 0 } },
      fields1555[] = { { 3, 1, 15 }, { 0, 5, 10 }, { 1, 5, 5 }, { 2, 5, 0 } };
    unsigned nf = k->dst_fmt == 4 ? 3 : 4;
    for (unsigned f = 0; f < nf; f++) {
        unsigned ch  = k->dst_fmt == 4 ? fields565[f].ch : fields1555[f].ch;
        unsigned n   = k->dst_fmt == 4 ? fields565[f].n : fields1555[f].n;
        unsigned pos = k->dst_fmt == 4 ? fields565[f].pos : fields1555[f].pos;
        LD(RAX, CTX, CTXF(c[0]) + 4 * (int32_t) ch);
        emit_quantize(a, n, dither);
        shift_ri(a, 4, RCX, pos, false);
        rr(a, 0x09, RCX, R8, false);
    }
    }

    /* addr = dst_base + dst_stride * y + 2x */
    MOV(RAX, RDI);
    rm(a, 0x0faf, RAX, CTX, NO_INDEX, CTXF(dst_stride), false, false);
    rm(a, 0x03, RAX, PE, NO_INDEX, PEF(dst_base), false, false);
    rr(a, 0x01, RSI, RAX, false);
    rr(a, 0x01, RSI, RAX, false);
    emit_store16(a, R8, RAX);
    if (zwrite) {
        LD(RAX, CTX, CTXF(z_addr));
        LD(RCX, CTX, CTXF(z));
        emit_store16(a, RCX, RAX);
    }
    mov_ri(a, DREW, 1);

    for (unsigned i = 0; i < nskip; i++)
        land(skips[i], a->p);
    if (q_index >= 0) {
        LD(RAX, CTX, CTXF(values[0]) + 4 * q_index);
        ST(RAX, PE, PEF(q));
    }
    if (shadow) {
        LD(RAX, CTX, CTXF(x_step));
        rm(a, 0x01, RAX, PE, NO_INDEX, PEF(x_shadow), false, false);
    }
    for (unsigned i = 0; i < count; i++) {
        LD(RAX, CTX, CTXF(deltas[0]) + 4 * (int32_t) i);
        rm(a, 0x01, RAX, CTX, NO_INDEX, CTXF(values[0]) + 4 * (int32_t) i, false, false);
    }
    rr(a, 0xff, 1, COUNT, false); /* dec ebx */
    land(jcc(a, CC_NE), loop);

    if (single) {
        /* dest[k] = value + delta (already stepped); Cnt decrements. */
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
    rm(a, 0x8b, RAX, CTX, NO_INDEX, CTXF(core), true, false);
    rm(a, 0xc6, 0, RAX, NO_INDEX, (int32_t) offsetof(verite_core, drew), false, false);
    b8(a, 1);
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
    as a = { jit->arena + jit->used };
    uint8_t *code = compile(&a, key);
    size_t   size = (size_t) (a.p - code);
    if (size > SPAN_MAX_CODE)
        abort();
#    ifdef _WIN32
    FlushInstructionCache(GetCurrentProcess(), code, size);
#    endif
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
