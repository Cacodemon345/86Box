/* SPDX-License-Identifier: GPL-2.0-or-later */
/* A small AArch64 assembler for the Vérité recompilers. */
#ifndef VERITE_A64_H
#define VERITE_A64_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint8_t *p;
} a64;

/* X16 and X17 are the assembler's scratch registers: X16 for an address
   offset, X17 for an immediate operand. Neither lives across a helper. */
enum { A64_IP0 = 16, A64_IP1 = 17, A64_LR = 30, A64_ZR = 31, A64_SP = 31 };

enum { A64_EQ, A64_NE, A64_HS, A64_LO, A64_MI, A64_PL, A64_VS, A64_VC,
       A64_HI, A64_LS, A64_GE, A64_LT, A64_GT, A64_LE, A64_AL };

static inline void
a64_put(a64 *a, uint32_t insn)
{
    memcpy(a->p, &insn, 4);
    a->p += 4;
}

/* ---- Immediates --------------------------------------------------------- */

static inline void
a64_mov_imm(a64 *a, bool sf, unsigned rd, uint64_t v)
{
    unsigned chunks = sf ? 4 : 2;
    if (!sf)
        v &= UINT32_MAX;
    uint64_t inv = sf ? ~v : (~v & UINT32_MAX);
    /* One MOVZ or MOVN when the value allows it. */
    for (unsigned hw = 0; hw < chunks; hw++) {
        if (!(v & ~((uint64_t) 0xffff << (16 * hw)))) {
            a64_put(a, 0x52800000 | (sf << 31) | (hw << 21) | (uint32_t) ((v >> (16 * hw)) & 0xffff) << 5 | rd);
            return;
        }
        if (!(inv & ~((uint64_t) 0xffff << (16 * hw)))) {
            a64_put(a, 0x12800000 | (sf << 31) | (hw << 21) | (uint32_t) ((inv >> (16 * hw)) & 0xffff) << 5 | rd);
            return;
        }
    }
    bool first = true;
    for (unsigned hw = 0; hw < chunks; hw++) {
        uint32_t part = (uint32_t) (v >> (16 * hw)) & 0xffff;
        if (!part)
            continue;
        a64_put(a, (first ? 0x52800000 : 0x72800000) | (sf << 31) | (hw << 21) | part << 5 | rd);
        first = false;
    }
}

/* The N:immr:imms field of a logical immediate, or -1. */
static inline int
a64_bitmask(uint64_t imm, bool sf)
{
    unsigned width = sf ? 64 : 32;
    if (!sf)
        imm = (imm & UINT32_MAX) | (imm << 32);
    if (!imm || !~imm)
        return -1;
    unsigned size = 64;
    while (size > 2) {
        unsigned half = size / 2;
        uint64_t mask = (UINT64_C(1) << half) - 1;
        if ((imm & mask) != ((imm >> half) & mask))
            break;
        size = half;
    }
    if (size > width)
        return -1;
    uint64_t mask = size == 64 ? ~UINT64_C(0) : (UINT64_C(1) << size) - 1;
    uint64_t elem = imm & mask;
    unsigned ones = 0;
    for (uint64_t t = elem; t; t &= t - 1)
        ones++;
    uint64_t run = ones == 64 ? ~UINT64_C(0) : (UINT64_C(1) << ones) - 1;
    for (unsigned r = 0; r < size; r++) {
        /* rotl(elem, r) == run means ROR(run, r) == elem */
        uint64_t rot = r ? ((elem << r) | (elem >> (size - r))) & mask : elem;
        if (rot == run) {
            unsigned n    = size == 64;
            unsigned imms = ((~(size - 1) << 1) & 0x3f) | (ones - 1);
            return (int) (n << 12 | r << 6 | imms);
        }
    }
    return -1;
}

/* ---- Integer data processing ------------------------------------------- */

enum { A64_ADD, A64_ADDS, A64_SUB, A64_SUBS, A64_AND, A64_ORR, A64_EOR, A64_ANDS, A64_BIC, A64_ORN };
enum { A64_LSL, A64_LSR, A64_ASR, A64_ROR };

static inline void
a64_alu(a64 *a, unsigned op, bool sf, unsigned rd, unsigned rn, unsigned rm, unsigned shift, unsigned amount)
{
    static const uint32_t base[] = { 0x0b000000, 0x2b000000, 0x4b000000, 0x6b000000, 0x0a000000,
                                     0x2a000000, 0x4a000000, 0x6a000000, 0x0a200000, 0x2a200000 };
    a64_put(a, base[op] | (sf << 31) | (shift << 22) | (rm << 16) | ((amount & 63) << 10) | (rn << 5) | rd);
}

static inline void
a64_rrr(a64 *a, unsigned op, bool sf, unsigned rd, unsigned rn, unsigned rm)
{
    a64_alu(a, op, sf, rd, rn, rm, A64_LSL, 0);
}

static inline void
a64_mov(a64 *a, bool sf, unsigned rd, unsigned rm)
{
    a64_rrr(a, A64_ORR, sf, rd, A64_ZR, rm);
}

/* rd = rn op imm. ADD and SUB may use SP as rn and, without S, as rd. */
static inline void
a64_alu_imm(a64 *a, unsigned op, bool sf, unsigned rd, unsigned rn, uint64_t imm)
{
    if (!sf)
        imm &= UINT32_MAX;
    if (op <= A64_SUBS) {
        static const uint32_t base[] = { 0x11000000, 0x31000000, 0x51000000, 0x71000000 };
        uint64_t neg = sf ? (uint64_t) -imm : (uint32_t) -(uint32_t) imm;
        for (unsigned pass = 0; pass < 2; pass++) {
            uint64_t v  = pass ? neg : imm;
            unsigned o  = pass ? op ^ 2 : op; /* ADD <-> SUB */
            if (pass && !imm)
                break;
            if (v < 4096) {
                a64_put(a, base[o] | (sf << 31) | (uint32_t) v << 10 | (rn << 5) | rd);
                return;
            }
            if (!(v & 0xfff) && v < (1 << 24)) {
                a64_put(a, base[o] | (sf << 31) | 1 << 22 | (uint32_t) (v >> 12) << 10 | (rn << 5) | rd);
                return;
            }
        }
    } else {
        static const uint32_t base[] = { [A64_AND] = 0x12000000, [A64_ORR] = 0x32000000,
                                         [A64_EOR] = 0x52000000, [A64_ANDS] = 0x72000000 };
        int field = op <= A64_ANDS ? a64_bitmask(imm, sf) : -1;
        if (field >= 0) {
            a64_put(a, base[op] | (sf << 31) | (uint32_t) field << 10 | (rn << 5) | rd);
            return;
        }
    }
    a64_mov_imm(a, sf, A64_IP1, imm);
    a64_rrr(a, op, sf, rd, rn, A64_IP1);
}

static inline void
a64_cmp_imm(a64 *a, bool sf, unsigned rn, uint64_t imm)
{
    a64_alu_imm(a, A64_SUBS, sf, A64_ZR, rn, imm);
}

static inline void
a64_cmp(a64 *a, bool sf, unsigned rn, unsigned rm)
{
    a64_rrr(a, A64_SUBS, sf, A64_ZR, rn, rm);
}

static inline void
a64_tst_imm(a64 *a, bool sf, unsigned rn, uint64_t imm)
{
    a64_alu_imm(a, A64_ANDS, sf, A64_ZR, rn, imm);
}

/* rd = rn + (rm extended), for a 64-bit base and a 32-bit index: UXTW is 2,
   SXTW 6. */
static inline void
a64_add_ext(a64 *a, unsigned rd, unsigned rn, unsigned rm, unsigned option, unsigned amount)
{
    a64_put(a, 0x8b200000 | (rm << 16) | (option << 13) | (amount << 10) | (rn << 5) | rd);
}

static inline void
a64_bfm(a64 *a, uint32_t base, bool sf, unsigned rd, unsigned rn, unsigned immr, unsigned imms)
{
    a64_put(a, base | (sf << 31) | (sf << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd);
}

enum { A64_SBFM = 0x13000000, A64_BFM = 0x33000000, A64_UBFM = 0x53000000 };

static inline void
a64_shift_imm(a64 *a, unsigned kind, bool sf, unsigned rd, unsigned rn, unsigned n)
{
    unsigned bits = sf ? 64 : 32;
    n &= bits - 1;
    switch (kind) {
        case A64_LSL:
            a64_bfm(a, A64_UBFM, sf, rd, rn, (bits - n) & (bits - 1), bits - 1 - n);
            break;
        case A64_LSR:
            a64_bfm(a, A64_UBFM, sf, rd, rn, n, bits - 1);
            break;
        case A64_ASR:
            a64_bfm(a, A64_SBFM, sf, rd, rn, n, bits - 1);
            break;
        default: /* EXTR rd, rn, rn, #n */
            a64_put(a, 0x13800000 | (sf << 31) | (sf << 22) | (rn << 16) | (n << 10) | (rn << 5) | rd);
            break;
    }
}

static inline void
a64_ubfx(a64 *a, bool sf, unsigned rd, unsigned rn, unsigned lsb, unsigned width)
{
    a64_bfm(a, A64_UBFM, sf, rd, rn, lsb, lsb + width - 1);
}

static inline void
a64_sxtw(a64 *a, unsigned rd, unsigned rn)
{
    a64_bfm(a, A64_SBFM, true, rd, rn, 0, 31);
}

static inline void
a64_shiftv(a64 *a, unsigned kind, bool sf, unsigned rd, unsigned rn, unsigned rm)
{
    a64_put(a, 0x1ac02000 | (sf << 31) | (rm << 16) | (kind << 10) | (rn << 5) | rd);
}

static inline void
a64_madd(a64 *a, bool sf, unsigned rd, unsigned rn, unsigned rm, unsigned ra)
{
    a64_put(a, 0x1b000000 | (sf << 31) | (rm << 16) | (ra << 10) | (rn << 5) | rd);
}

static inline void
a64_mul(a64 *a, bool sf, unsigned rd, unsigned rn, unsigned rm)
{
    a64_madd(a, sf, rd, rn, rm, A64_ZR);
}

static inline void
a64_smull(a64 *a, unsigned rd, unsigned rn, unsigned rm)
{
    a64_put(a, 0x9b200000 | (rm << 16) | (A64_ZR << 10) | (rn << 5) | rd);
}

static inline void
a64_sdiv(a64 *a, bool sf, unsigned rd, unsigned rn, unsigned rm)
{
    a64_put(a, 0x1ac00c00 | (sf << 31) | (rm << 16) | (rn << 5) | rd);
}

static inline void
a64_csel(a64 *a, bool sf, unsigned rd, unsigned rn, unsigned rm, unsigned cond)
{
    a64_put(a, 0x1a800000 | (sf << 31) | (rm << 16) | (cond << 12) | (rn << 5) | rd);
}

static inline void
a64_cset(a64 *a, bool sf, unsigned rd, unsigned cond)
{
    /* CSINC rd, zr, zr, !cond */
    a64_put(a, 0x1a800400 | (sf << 31) | (A64_ZR << 16) | ((cond ^ 1) << 12) | (A64_ZR << 5) | rd);
}

static inline void
a64_cneg(a64 *a, bool sf, unsigned rd, unsigned rn, unsigned cond)
{
    /* CSNEG rd, rn, rn, !cond */
    a64_put(a, 0x5a800400 | (sf << 31) | (rn << 16) | ((cond ^ 1) << 12) | (rn << 5) | rd);
}

/* ---- Branches ------------------------------------------------------------ */

static inline uint8_t *
a64_b(a64 *a)
{
    uint8_t *at = a->p;
    a64_put(a, 0x14000000);
    return at;
}

static inline uint8_t *
a64_bcond(a64 *a, unsigned cond)
{
    uint8_t *at = a->p;
    a64_put(a, 0x54000000 | cond);
    return at;
}

static inline uint8_t *
a64_cbz(a64 *a, bool nonzero, bool sf, unsigned rt)
{
    uint8_t *at = a->p;
    a64_put(a, (nonzero ? 0x35000000 : 0x34000000) | (sf << 31) | rt);
    return at;
}

static inline uint8_t *
a64_tbz(a64 *a, bool nonzero, unsigned rt, unsigned bit)
{
    uint8_t *at = a->p;
    a64_put(a, (nonzero ? 0x37000000 : 0x36000000) | ((bit >> 5) << 31) | ((bit & 31) << 19) | rt);
    return at;
}

/* Points the branch at at to target. */
static inline void
a64_land(uint8_t *at, const uint8_t *target)
{
    uint32_t insn;
    memcpy(&insn, at, 4);
    int64_t words = (target - at) / 4;
    if ((insn & 0x7c000000) == 0x14000000)
        insn = (insn & 0xfc000000) | ((uint32_t) words & 0x3ffffff);
    else if ((insn & 0x7e000000) == 0x36000000)
        insn = (insn & ~(0x3fffU << 5)) | ((uint32_t) words & 0x3fff) << 5;
    else /* B.cond, CBZ, CBNZ */
        insn = (insn & ~(0x7ffffU << 5)) | ((uint32_t) words & 0x7ffff) << 5;
    memcpy(at, &insn, 4);
}

static inline void
a64_bcond_to(a64 *a, unsigned cond, const uint8_t *target)
{
    a64_land(a64_bcond(a, cond), target);
}

static inline void
a64_blr(a64 *a, unsigned rn)
{
    a64_put(a, 0xd63f0000 | (rn << 5));
}

static inline void
a64_ret(a64 *a)
{
    a64_put(a, 0xd65f03c0);
}

static inline void
a64_adr(a64 *a, unsigned rd, const void *target)
{
    int64_t  off = (const uint8_t *) target - a->p;
    uint32_t imm = (uint32_t) off & 0x1fffff;
    a64_put(a, 0x10000000 | (imm & 3) << 29 | (imm >> 2) << 5 | rd);
}

/* ---- Loads and stores ---------------------------------------------------- */

/* size:opc:V for each access, with its width in bytes. */
enum { A64_LDRB, A64_LDRH, A64_LDRSH, A64_LDR_W, A64_LDR_X,
       A64_STRB, A64_STRH, A64_STR_W, A64_STR_X,
       A64_LDR_S, A64_LDR_D, A64_LDR_Q, A64_STR_S, A64_STR_D, A64_STR_Q };

static const struct {
    uint8_t size, opc, simd, bytes;
} a64_access[] = {
    [A64_LDRB] = { 0, 1, 0, 1 }, [A64_LDRH] = { 1, 1, 0, 2 }, [A64_LDRSH] = { 1, 3, 0, 2 },
    [A64_LDR_W] = { 2, 1, 0, 4 }, [A64_LDR_X] = { 3, 1, 0, 8 },
    [A64_STRB] = { 0, 0, 0, 1 }, [A64_STRH] = { 1, 0, 0, 2 }, [A64_STR_W] = { 2, 0, 0, 4 },
    [A64_STR_X] = { 3, 0, 0, 8 },
    [A64_LDR_S] = { 2, 1, 1, 4 }, [A64_LDR_D] = { 3, 1, 1, 8 }, [A64_LDR_Q] = { 0, 3, 1, 16 },
    [A64_STR_S] = { 2, 0, 1, 4 }, [A64_STR_D] = { 3, 0, 1, 8 }, [A64_STR_Q] = { 0, 2, 1, 16 },
};

enum { A64_UXTW = 2, A64_XLSL = 3, A64_SXTW = 6 };

/* rt <-> [rn + extend(rm) << (scaled ? log2 width : 0)] */
static inline void
a64_ldst_reg(a64 *a, unsigned kind, unsigned rt, unsigned rn, unsigned rm, unsigned option, bool scaled)
{
    uint32_t insn = 0x38200800 | (uint32_t) a64_access[kind].size << 30 | (uint32_t) a64_access[kind].simd << 26
        | (uint32_t) a64_access[kind].opc << 22;
    a64_put(a, insn | (rm << 16) | (option << 13) | (scaled << 12) | (rn << 5) | rt);
}

/* rt <-> [rn + disp]; rn may be SP. Uses X16 when disp does not encode. */
static inline void
a64_ldst(a64 *a, unsigned kind, unsigned rt, unsigned rn, int64_t disp)
{
    unsigned bytes = a64_access[kind].bytes;
    if (disp >= 0 && !(disp % bytes) && disp / bytes < 4096) {
        uint32_t insn = 0x39000000 | (uint32_t) a64_access[kind].size << 30 | (uint32_t) a64_access[kind].simd << 26
            | (uint32_t) a64_access[kind].opc << 22;
        a64_put(a, insn | (uint32_t) (disp / bytes) << 10 | (rn << 5) | rt);
        return;
    }
    a64_mov_imm(a, true, A64_IP0, (uint64_t) disp);
    a64_ldst_reg(a, kind, rt, rn, A64_IP0, A64_XLSL, false);
}

/* STP/LDP of X registers: mode 1 post-index, 3 pre-index, 2 signed offset. */
static inline void
a64_pair(a64 *a, bool load, unsigned mode, unsigned rt, unsigned rt2, unsigned rn, int disp)
{
    a64_put(a, 0xa8000000 | (mode << 23) | (load << 22) | ((uint32_t) (disp / 8) & 0x7f) << 15 | (rt2 << 10) | (rn << 5) | rt);
}

/* PRFM PLDL1KEEP, [rn, wm, uxtw] */
static inline void
a64_prefetch(a64 *a, unsigned rn, unsigned rm)
{
    a64_put(a, 0xf8a04800 | (rm << 16) | (rn << 5));
}

/* ---- Advanced SIMD ------------------------------------------------------- */

/* Arrangements as Q << 2 | size. */
enum { V8B = 0, V4H = 1, V2S = 2, V16B = 4, V8H = 5, V4S = 6 };

static inline void
a64_v3(a64 *a, unsigned arr, bool u, unsigned opcode, unsigned rd, unsigned rn, unsigned rm)
{
    a64_put(a, 0x0e200400 | (arr >> 2) << 30 | (u << 29) | (arr & 3) << 22 | (rm << 16) | (opcode << 11) | (rn << 5) | rd);
}

#define A64_VADD(a, t, d, n, m)  a64_v3(a, t, false, 0x10, d, n, m)
#define A64_VSUB(a, t, d, n, m)  a64_v3(a, t, true, 0x10, d, n, m)
#define A64_VMUL(a, t, d, n, m)  a64_v3(a, t, false, 0x13, d, n, m)
#define A64_VMLA(a, t, d, n, m)  a64_v3(a, t, false, 0x12, d, n, m)
#define A64_VSMIN(a, t, d, n, m) a64_v3(a, t, false, 0x0d, d, n, m)
#define A64_VUSHL(a, t, d, n, m) a64_v3(a, t, true, 0x08, d, n, m)
/* Bitwise operations take the size field as the operation. */
#define A64_VAND(a, d, n, m) a64_v3(a, 4 | 0, false, 0x03, d, n, m)
#define A64_VBIC(a, d, n, m) a64_v3(a, 4 | 1, false, 0x03, d, n, m)
#define A64_VORR(a, d, n, m) a64_v3(a, 4 | 2, false, 0x03, d, n, m)
#define A64_VEOR(a, d, n, m) a64_v3(a, 4 | 0, true, 0x03, d, n, m)

static inline void
a64_vnot(a64 *a, unsigned rd, unsigned rn)
{
    a64_put(a, 0x6e205800 | (rn << 5) | rd);
}

static inline void
a64_vmov(a64 *a, unsigned rd, unsigned rn)
{
    A64_VORR(a, rd, rn, rn);
}

/* 4H x 4H -> 4S */
static inline void
a64_umull(a64 *a, unsigned rd, unsigned rn, unsigned rm)
{
    a64_put(a, 0x2e60c000 | (rm << 16) | (rn << 5) | rd);
}

static inline void
a64_umlal(a64 *a, unsigned rd, unsigned rn, unsigned rm)
{
    a64_put(a, 0x2e608000 | (rm << 16) | (rn << 5) | rd);
}

/* UXTL 8B -> 8H, or 4H -> 4S with words set. */
static inline void
a64_uxtl(a64 *a, bool words, unsigned rd, unsigned rn)
{
    a64_put(a, (words ? 0x2f10a400 : 0x2f08a400) | (rn << 5) | rd);
}

/* XTN 4S -> 4H */
static inline void
a64_xtn(a64 *a, unsigned rd, unsigned rn)
{
    a64_put(a, 0x0e612800 | (rn << 5) | rd);
}

/* SHRN or RSHRN 4S -> 4H */
static inline void
a64_shrn(a64 *a, bool round, unsigned rd, unsigned rn, unsigned n)
{
    a64_put(a, (round ? 0x0f008c00 : 0x0f008400) | (32 - n) << 16 | (rn << 5) | rd);
}

/* USHR 4H */
static inline void
a64_ushr4h(a64 *a, unsigned rd, unsigned rn, unsigned n)
{
    a64_put(a, 0x2f000400 | (32 - n) << 16 | (rn << 5) | rd);
}

/* ADDV Hd, Vn.4H */
static inline void
a64_addv4h(a64 *a, unsigned rd, unsigned rn)
{
    a64_put(a, 0x0e71b800 | (rn << 5) | rd);
}

/* DUP Vd.4H or .2S, Wn */
static inline void
a64_dup(a64 *a, bool words, unsigned rd, unsigned rn)
{
    a64_put(a, (words ? 0x0e040c00 : 0x0e020c00) | (rn << 5) | rd);
}

/* DUP Vd.4H, Vn.H[lane] */
static inline void
a64_dup_lane(a64 *a, unsigned rd, unsigned rn, unsigned lane)
{
    a64_put(a, 0x0e000400 | ((lane << 2) | 2) << 16 | (rn << 5) | rd);
}

/* INS Vd.H[lane], Wn */
static inline void
a64_ins(a64 *a, unsigned rd, unsigned lane, unsigned rn)
{
    a64_put(a, 0x4e001c00 | ((lane << 2) | 2) << 16 | (rn << 5) | rd);
}

/* UMOV Wd, Vn.H[lane] */
static inline void
a64_umov(a64 *a, unsigned rd, unsigned rn, unsigned lane)
{
    a64_put(a, 0x0e003c00 | ((lane << 2) | 2) << 16 | (rn << 5) | rd);
}

/* FMOV Sd, Wn */
static inline void
a64_fmov_from(a64 *a, unsigned rd, unsigned rn)
{
    a64_put(a, 0x1e270000 | (rn << 5) | rd);
}

/* ---- Executable memory --------------------------------------------------- */

#ifdef _WIN32
#    include <windows.h>
#else
#    include <sys/mman.h>
#    if defined(__APPLE__)
#        include <pthread.h>
#        include <libkern/OSCacheControl.h>
#    endif
#endif

static inline uint8_t *
a64_alloc_code(size_t size)
{
#ifdef _WIN32
    return VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
#else
    int flags = MAP_PRIVATE | MAP_ANON;
#    if defined(__APPLE__) && defined(MAP_JIT)
    flags |= MAP_JIT;
#    endif
    void *code = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC, flags, -1, 0);
    return code == MAP_FAILED ? NULL : code;
#endif
}

static inline void
a64_free_code(uint8_t *code, size_t size)
{
#ifdef _WIN32
    (void) size;
    VirtualFree(code, 0, MEM_RELEASE);
#else
    munmap(code, size);
#endif
}

/* Brackets writes to code memory; Apple silicon maps a MAP_JIT region either
   writable or executable for each thread. */
static inline void
a64_write_begin(void)
{
#if defined(__APPLE__) && defined(MAP_JIT)
    pthread_jit_write_protect_np(0);
#endif
}

static inline void
a64_write_end(uint8_t *code, size_t size)
{
#if defined(__APPLE__) && defined(MAP_JIT)
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(code, size);
#elif defined(_WIN32)
    FlushInstructionCache(GetCurrentProcess(), code, size);
#else
    __builtin___clear_cache((char *) code, (char *) code + size);
#endif
}

#endif
