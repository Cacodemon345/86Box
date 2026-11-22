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
    DST_RAW_ZWRITE  = 1 << 16,
    DST_RAW_YUV     = 1 << 18,
    DST_RAW_BLEND   = 1 << 19,
    DST_RAW_DITHER  = 1 << 20,
    DST_RAW_FOG     = 1 << 21,
    DST_RAW_BGR     = 1 << 24,
    DST_RAW_TREJECT = 1 << 25,
};

/* Shared with the generated code, which addresses it through R15. */
typedef struct {
    uint8_t *vram;
    uint32_t mask;
    uint32_t count;
    uint32_t x_step, dst_stride, z_stride, tex_stride;
    uint32_t values[10], deltas[10];
    int32_t  c[4], t[4], tap[4], sum[4], w[4];
    uint32_t z_addr, z;
    uint32_t row0, row1, x0, x1;
} span_ctx;

typedef unsigned (*span_fn)(span_ctx *ctx, pixel_engine *pe);

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

/* Store the low 16 bits of val at the address in reg (clobbers reg). */
static void
emit_store16(as *a, unsigned val, unsigned addr)
{
    rr(a, 0x21, MASK, addr, false);
    alu_ri(a, 4, addr, ~1U);
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
                ST(RAX, PE, u ? PEF(u) : PEF(v));
                LD(RCX, PE, PEF(src_mask));
                if (!u)
                    shift_ri(a, 5, RCX, 16, false);
                alu_ri(a, 4, RCX, 0x7ff);
                ST(RCX, PE, u ? PEF(u_mask) : PEF(v_mask));
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

/* texel() for a 16-bit format at row (ctx field) + x (ctx field), into out[4]. */
static void
emit_texel(as *a, unsigned fmt, uint32_t src_mode, int32_t row, int32_t x, int32_t out)
{
    bool no_pad = !!(src_mode & SRC_NO_PAD);
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
        bool yuv = !!(src_mode & SRC_CHROMA_BLACK_YUV);
        static const uint32_t black[] = { 0, 0, 0, 0 }, yuv_black[] = { 128, 16, 128, 0 };
        for (unsigned k = 0; k < 4; k++) {
            rm(a, 0xc7, 0, CTX, NO_INDEX, out + 4 * (int32_t) k, false, false);
            d32(a, yuv ? yuv_black[k] : black[k]);
        }
        done = jmp(a);
        land(decode, a->p);
    }
    if (fmt == 4) {
        emit_widen(a, 11, 5, no_pad);
        ST(RCX, CTX, out);
        emit_widen(a, 5, 6, no_pad);
        ST(RCX, CTX, out + 4);
        emit_widen(a, 0, 5, no_pad);
        ST(RCX, CTX, out + 8);
        rm(a, 0xc7, 0, CTX, NO_INDEX, out + 12, false, false);
        d32(a, 255);
    } else {
        emit_widen(a, 10, 5, no_pad);
        ST(RCX, CTX, out);
        emit_widen(a, 5, 5, no_pad);
        ST(RCX, CTX, out + 4);
        emit_widen(a, 0, 5, no_pad);
        ST(RCX, CTX, out + 8);
        shift_ri(a, 5, RAX, 15, false);
        alu_ri(a, 4, RAX, 1);
        rr(a, 0xf7, 3, RAX, false); /* neg eax */
        alu_ri(a, 4, RAX, 255);
        ST(RAX, CTX, out + 12);
    }
    if (done)
        land(done, a->p);
}

/* sample() into ctx.t. R10D = u, R11D = v after any perspective division. */
static void
emit_sample(as *a, unsigned fmt, uint32_t src_mode)
{
    bool uc = !!(src_mode & SRC_U_CLAMP), vc = !!(src_mode & SRC_V_CLAMP);
    bool filter = !!(src_mode & SRC_FILTER);

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
    if (!filter) {
        emit_texel(a, fmt, src_mode, CTXF(row0), CTXF(x0), CTXF(t));
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

    if (fmt == 4 && !(src_mode & SRC_CHROMA)) {
        /* sample_bilinear_565() */
        bool no_pad = !!(src_mode & SRC_NO_PAD);
        for (unsigned k = 0; k < 3; k++)
            rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(sum[0]) + 4 * (int32_t) k, false, false), d32(a, 0);
        static const unsigned shifts[] = { 11, 5, 0 }, bits[] = { 5, 6, 5 };
        for (unsigned tap = 0; tap < 4; tap++) {
            LD(RAX, CTX, xs[tap]);
            rr(a, 0x01, RAX, RAX, false);
            rm(a, 0x03, RAX, CTX, NO_INDEX, rows[tap], false, false);
            emit_load16(a, RAX, RAX);
            for (unsigned k = 0; k < 3; k++) {
                if (no_pad) {
                    MOV(RCX, RAX);
                    shift_ri(a, 5, RCX, shifts[k], false);
                    alu_ri(a, 4, RCX, (1U << bits[k]) - 1);
                } else
                    emit_widen(a, shifts[k], bits[k], false);
                rm(a, 0x0faf, RCX, CTX, NO_INDEX, CTXF(w[0]) + 4 * (int32_t) tap, false, false);
                rm(a, 0x01, RCX, CTX, NO_INDEX, CTXF(sum[0]) + 4 * (int32_t) k, false, false);
            }
        }
        static const uint32_t round_np[] = { 4096, 8192, 4096 }, shift_np[] = { 13, 14, 13 };
        for (unsigned k = 0; k < 3; k++) {
            LD(RAX, CTX, CTXF(sum[0]) + 4 * (int32_t) k);
            alu_ri(a, 0, RAX, no_pad ? round_np[k] : 32768);
            shift_ri(a, 7, RAX, no_pad ? shift_np[k] : 16, false);
            ST(RAX, CTX, CTXF(t[0]) + 4 * (int32_t) k);
        }
        rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(t[3]), false, false);
        d32(a, 255);
        return;
    }

    /* Generic four-tap filter over decoded (and keyed) texels. */
    for (unsigned k = 0; k < 4; k++)
        rm(a, 0xc7, 0, CTX, NO_INDEX, CTXF(sum[0]) + 4 * (int32_t) k, false, false), d32(a, 0);
    for (unsigned tap = 0; tap < 4; tap++) {
        emit_texel(a, fmt, src_mode, rows[tap], xs[tap], CTXF(tap));
        for (unsigned k = 0; k < 4; k++) {
            LD(RAX, CTX, CTXF(tap[0]) + 4 * (int32_t) k);
            rm(a, 0x0faf, RAX, CTX, NO_INDEX, CTXF(w[0]) + 4 * (int32_t) tap, false, false);
            rm(a, 0x01, RAX, CTX, NO_INDEX, CTXF(sum[0]) + 4 * (int32_t) k, false, false);
        }
    }
    for (unsigned k = 0; k < 4; k++) {
        LD(RAX, CTX, CTXF(sum[0]) + 4 * (int32_t) k);
        alu_ri(a, 0, RAX, 32768);
        shift_ri(a, 7, RAX, 16, false);
        ST(RAX, CTX, CTXF(t[0]) + 4 * (int32_t) k);
    }
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

static bool
key_supported(const span_key *k)
{
    unsigned func = (k->src_mode >> 4) & 7;
    unsigned fmt  = k->src_mode & 15 ? k->src_mode & 15 : k->dst_fmt;
    if ((k->dst_fmt != 4 && k->dst_fmt != 6) || (k->dst_mode & 15) != 12)
        return false;
    if (k->dst_mode & (DST_RAW_YUV | DST_RAW_BLEND | DST_RAW_FOG | DST_RAW_BGR))
        return false;
    if (func > 3)
        return false;
    if (func && ((fmt != 4 && fmt != 6) || (k->src_mode & SRC_BGR)))
        return false;
    return true;
}

/* Generates unsigned fn(span_ctx *ctx, pixel_engine *pe); returns whether a pixel was written. */
static uint8_t *
compile(as *a, const span_key *k)
{
    uint8_t     *start = a->p;
    unsigned     count = k->op;
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

    static const unsigned saved[] = { RBX, RBP, RSI, RDI, R12, R13, R14, R15 };
    for (unsigned i = 0; i < 8; i++)
        rex(a, false, 0, 0, saved[i], false), b8(a, 0x50 | (saved[i] & 7));
#    ifdef _WIN32
    rr(a, 0x89, RCX, CTX, true);
    rr(a, 0x89, RDX, PE, true);
#    else
    rr(a, 0x89, RDI, CTX, true);
    rr(a, 0x89, RSI, PE, true);
#    endif
    rm(a, 0x8b, VRAM, CTX, NO_INDEX, CTXF(vram), true, false);
    LD(MASK, CTX, CTXF(mask));
    LD(COUNT, CTX, CTXF(count));
    rr(a, 0x31, DREW, DREW, false);

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
        LD(RAX, PE, axis ? PEF(scissor_y) : PEF(scissor_x));
        MOV(RCX, RAX);
        shift_ri(a, 5, RCX, 16, false);
        alu_ri(a, 4, RCX, 0xfff);
        rr(a, 0x39, RCX, r, false); /* cmp r, ecx */
        skips[nskip++] = jcc(a, CC_L);
        alu_ri(a, 4, RAX, 0xfff);
        rr(a, 0x39, RAX, r, false);
        skips[nskip++] = jcc(a, CC_GE);
        alu_ri(a, 7, r, 2048);
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

    /* c = argb(fg) */
    LD(RAX, PE, PEF(fg));
    for (unsigned ch = 0; ch < 4; ch++) {
        static const unsigned sh[] = { 16, 8, 0, 24 };
        MOV(RCX, RAX);
        shift_ri(a, 5, RCX, sh[ch], false);
        alu_ri(a, 4, RCX, 255);
        ST(RCX, CTX, CTXF(c[0]) + 4 * (int32_t) ch);
    }

    if (func) {
        LD(R10, PE, PEF(u));
        LD(R11, PE, PEF(v));
        if (q_index >= 0) {
            LD(R8, PE, PEF(q));
            rr(a, 0x85, R8, R8, false);
            uint8_t *flat = jcc(a, CC_E);
            for (unsigned axis = 0; axis < 2; axis++) {
                rr(a, 0x63, RAX, axis ? R11 : R10, true); /* movsxd rax, r */
                shift_ri(a, 4, RAX, 24, true);
                b8(a, 0x48);
                b8(a, 0x99);                  /* cqo */
                rr(a, 0xf7, 7, R8, true);     /* idiv r8 */
                MOV(axis ? R11 : R10, RAX);
            }
            land(flat, a->p);
        }
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

    /* Encode into R8D. */
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

    MOV(RAX, DREW);
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
        memset(jit->entries, 0, sizeof(jit->entries));
        jit->used = 0;
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

bool
pixel_jit_span(verite_core *c, unsigned count, const uint32_t values[], const uint32_t deltas[],
               int cnt, uint32_t x_step, verite_memory mem)
{
    pixel_engine *pe = &c->pe;
    if (c->jit_disabled || c->jit_unavailable || (pe->pat_mode & 0x10000))
        return false;
    span_key key = {
        .op       = count,
        .drawctl  = pe->drawctl & 0xffff,
        .src_mode = pe->src_mode & 0x1ffff,
        .dst_mode = pe->dst_mode & 0x03ff7fff,
        .dst_fmt  = pe->dst_fmt & 15,
    };
    if ((pe->pmask & 0xffff) != 0xffff || !key_supported(&key) || mem.mask > UINT32_MAX)
        return false;
    span_fn fn = lookup(c, &key);
    if (!fn)
        return false;
    span_ctx ctx = {
        .vram       = mem.data,
        .mask       = (uint32_t) mem.mask,
        .count      = (uint32_t) cnt,
        .x_step     = x_step,
        .dst_stride = verite_line_stride(pe->stride >> 8),
        .z_stride   = verite_line_stride(pe->stride >> 16),
        .tex_stride = verite_line_stride(pe->stride),
    };
    memcpy(ctx.values, values, count * sizeof(*values));
    memcpy(ctx.deltas, deltas, count * sizeof(*deltas));
    if (fn(&ctx, pe))
        pe->pick_hit = c->drew = true;
    return true;
}

#endif
