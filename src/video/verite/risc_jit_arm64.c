/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Native basic blocks for the Vérité RISC on AArch64, after risc_jit_x64.c. */

#include "internal.h"

#ifdef VERITE_JIT_ARM64

#    include <stdlib.h>
#    include "a64.h"

enum {
    BLOCK_WORDS = 32,
    /* Includes two segmented lanes or a memory access with its exit stub. */
    MAX_BLOCK_CODE = BLOCK_WORDS * 400 + 128,
    /* Room kept free for the largest single instruction (a paired step with
       two U/V lanes and a Q call) and for each epilogue. */
    MAX_INSN_CODE = 1280,
    FINISH_CODE   = 128,
    /* The prologue's room for the memory argument or the loop budget. */
    PROLOGUE_SLOTS = 6,
    CACHE_SIZE     = 4096,
    ARENA_SIZE     = 4 * 1024 * 1024,
    SITE_COUNT     = 8192,
};

typedef verite_jit_memory jit_memory;

typedef unsigned (*block_fn)(verite_core *, const jit_memory *, unsigned);

typedef struct {
    block_fn fn;
    uint32_t pc;
    unsigned count;
    bool     fast_memory;
    uint32_t words[BLOCK_WORDS];
} jit_block;

struct verite_jit {
    uint8_t         *arena;
    size_t           used;
    unsigned         sites_used;
    jit_block        blocks[CACHE_SIZE];
    verite_draw_site sites[SITE_COUNT];
};

typedef a64 emitter;

/* W0 and W1 work as EAX and ECX do in the x64 blocks, W3 holds a pair's first
   lane and X3 the VRAM base in a memory access. X9 holds the core, X10 the
   memory argument or, in a loop, the remaining budget, W11 the next PC across
   a branch's delay slot, X12 the base of a segmented operand and X8 a draw
   site. All are caller-saved; calls save X9-X11 and the link register. */
enum { AX    = 0,
       CX    = 1,
       DX2   = 2,
       PAIR  = 3,
       VRAM  = 3,
       SITE  = 8,
       CORE  = 9,
       MEM   = 10,
       NPC   = 11,
       INDEX = 12 };

#    define CORE_FIELD(field) ((uint32_t) offsetof(verite_core, field))
#    define GPR(field)        (CORE_FIELD(reg) + (field) * 4U)
#    define PE(field)         CORE_FIELD(pe.field)

typedef struct {
    unsigned base;
    uint32_t disp;
} operand;

static void
core_ld(emitter *e, unsigned kind, unsigned reg, uint32_t disp)
{
    a64_ldst(e, kind, reg, CORE, disp);
}

static void
core_imm(emitter *e, uint32_t disp, uint32_t value)
{
    a64_mov_imm(e, false, A64_IP1, value);
    core_ld(e, A64_STR_W, A64_IP1, disp);
}

static bool
segmented(unsigned field)
{
    return (field & 0xc0) == 0x40;
}

static operand
reg_address(emitter *e, unsigned field)
{
    if (!segmented(field))
        return (operand) { CORE, GPR(field) };

    unsigned bank = (field >> 4) & 3;
    core_ld(e, A64_LDR_W, INDEX, CORE_FIELD(seg));
    a64_ubfx(e, false, INDEX, INDEX, bank * 4, 3);
    /* 16 registers of 4 bytes for each step of the segment */
    a64_alu(e, A64_ADD, true, INDEX, CORE, INDEX, A64_LSL, 6);
    return (operand) { INDEX, GPR(128 + (field & 15)) };
}

static void
read_reg(emitter *e, unsigned host, unsigned field)
{
    if (field < 32)
        a64_mov_imm(e, false, host, risc_constant(field));
    else {
        operand o = reg_address(e, field);
        a64_ldst(e, A64_LDR_W, host, o.base, o.disp);
    }
}

/* Destinations in the constant bank discard writes. */
static void
write_reg(emitter *e, unsigned field, unsigned host)
{
    if (field >= 64) {
        operand o = reg_address(e, field);
        a64_ldst(e, A64_STR_W, host, o.base, o.disp);
    }
}

static bool
plain_reg(unsigned field)
{
    return field < 32 || field >= 64;
}

/* Dn <-> the register pair at field & ~1 */
static void
packed_reg(emitter *e, unsigned vreg, unsigned field, bool write)
{
    operand o = reg_address(e, field & ~1U);
    a64_ldst(e, write ? A64_STR_D : A64_LDR_D, vreg, o.base, o.disp);
}

static void
emit_clip(emitter *e)
{
    a64_cmp(e, false, AX, CX);
    a64_cset(e, false, AX, A64_GT);
    core_ld(e, A64_LDR_W, CX, CORE_FIELD(clip));
    a64_shift_imm(e, A64_LSL, false, CX, CX, 1);
    a64_alu_imm(e, A64_AND, false, CX, CX, 0xfc);
    a64_alu(e, A64_ORR, false, CX, CX, AX, A64_LSL, 2);
    core_ld(e, A64_STR_W, CX, CORE_FIELD(clip));
}

static void
emit_alu(emitter *e, unsigned op, unsigned s2, unsigned s1, bool immediate)
{
    static const uint8_t binary[]        = { A64_ADD, A64_SUB, A64_BIC, A64_SUB, A64_AND, A64_ORR, A64_ORR, A64_EOR };
    static const uint8_t product_shift[] = { 0, 8, 16, 24, 32, 31 };

    read_reg(e, AX, s2);
    if (op < 8) {
        if (immediate && op != 3) {
            static const uint8_t group[] = { A64_ADD, A64_SUB, A64_AND, 0, A64_AND, A64_ORR, A64_ORR, A64_EOR };
            a64_alu_imm(e, group[op], false, AX, AX, op == 2 ? ~s1 : s1);
        } else {
            if (immediate)
                a64_mov_imm(e, false, CX, s1);
            else
                read_reg(e, CX, s1);
            if (op == 3)
                a64_rrr(e, A64_SUB, false, AX, CX, AX);
            else
                a64_rrr(e, binary[op], false, AX, AX, CX);
        }
        if (op == 6)
            a64_rrr(e, A64_ORN, false, AX, A64_ZR, AX);
        return;
    }

    if (immediate)
        a64_mov_imm(e, false, CX, s1);
    else
        read_reg(e, CX, s1);
    if (op == 14) {
        emit_clip(e);
        return;
    }
    if (op == 8) {
        a64_mul(e, false, AX, AX, CX);
        return;
    }
    a64_smull(e, AX, AX, CX);
    a64_shift_imm(e, A64_ASR, true, AX, AX, product_shift[op - 8]);
}

static void
emit_packed_alu(emitter *e, unsigned sub, unsigned d, unsigned s2, unsigned s1, bool pair_b)
{
    packed_reg(e, 0, s2, false);
    if (pair_b) {
        packed_reg(e, 1, s1, false);
    } else {
        read_reg(e, AX, s1);
        a64_dup(e, true, 1, AX);
    }
    switch (sub) {
        case 0:
            A64_VADD(e, V2S, 0, 0, 1);
            break;
        case 1:
            A64_VSUB(e, V2S, 0, 0, 1);
            break;
        case 2:
            A64_VBIC(e, 0, 0, 1);
            break;
        case 3:
            A64_VSUB(e, V2S, 0, 1, 0);
            break;
        case 4:
            A64_VAND(e, 0, 0, 1);
            break;
        case 5:
        case 6:
            A64_VORR(e, 0, 0, 1);
            if (sub == 6)
                a64_vnot(e, 0, 0);
            break;
        default:
            A64_VEOR(e, 0, 0, 1);
            break;
    }
    packed_reg(e, 0, d, true);
}

/* Call fn(core, arg1, memory). W0 supplies arg1 when from_ax is set. The
   link register, the core, the memory argument and the next PC survive. */
static void
emit_call(emitter *e, const void *fn, bool from_ax, uint32_t arg1)
{
    a64_pair(e, false, 3, A64_LR, NPC, A64_SP, -32);
    a64_pair(e, false, 2, CORE, MEM, A64_SP, 16);
    if (from_ax)
        a64_mov(e, false, 1, AX);
    else
        a64_mov_imm(e, false, 1, arg1);
    a64_mov(e, true, 0, CORE);
    a64_mov(e, true, 2, MEM);
    a64_mov_imm(e, true, A64_IP0, (uint64_t) (uintptr_t) fn);
    a64_blr(e, A64_IP0);
    a64_pair(e, true, 2, CORE, MEM, A64_SP, 16);
    a64_pair(e, true, 1, A64_LR, NPC, A64_SP, 32);
}

/* W0 = clamp(W0 >> 16, 0, 255). Clobbers W1. */
static void
emit_sat(emitter *e)
{
    a64_shift_imm(e, A64_ASR, false, AX, AX, 16);
    a64_alu(e, A64_BIC, false, AX, AX, AX, A64_ASR, 31); /* negative to 0 */
    a64_mov_imm(e, false, CX, 255);
    a64_cmp(e, false, AX, CX);
    a64_csel(e, false, AX, CX, AX, A64_GT);
}

/* fg = (fg & ~mask) | (W0 & mask). Clobbers W1. */
static void
emit_fg_merge(emitter *e, uint32_t mask)
{
    a64_alu_imm(e, A64_AND, false, AX, AX, mask);
    core_ld(e, A64_LDR_W, CX, PE(fg));
    a64_alu_imm(e, A64_AND, false, CX, CX, ~mask);
    a64_rrr(e, A64_ORR, false, AX, AX, CX);
    core_ld(e, A64_STR_W, AX, PE(fg));
}

static void
emit_uv_store(emitter *e, bool u)
{
    core_ld(e, A64_STR_W, AX, u ? PE(u) : PE(v));
    core_ld(e, A64_LDR_W, CX, PE(src_mask));
    a64_ubfx(e, false, CX, CX, u ? 0 : 16, 11);
    core_ld(e, A64_STR_W, CX, u ? PE(u_mask) : PE(v_mask));
}

/* Mirrors step_param() and pixel_set() with the value in W0 and the delta in
   W1. Clobbers W0-W2, and every caller-saved register for Q. */
static void
emit_param(emitter *e, pixel_param param)
{
    static const uint32_t channel[][2] = {
        [P_R] = { 16, 0x00ff0000 },
        [P_G] = { 8, 0x0000ff00 },
        [P_B] = { 0, 0x000000ff },
        [P_A] = { 24, 0xff000000 },
    };
    switch (param) {
        case P_NONE:
            break;
        case P_XSHADOW:
            a64_shift_imm(e, A64_LSR, false, DX2, CX, 31);
            core_ld(e, A64_STRB, DX2, PE(x_dir));
            core_ld(e, A64_STR_W, AX, PE(x_shadow));
            break;
        case P_X:
            core_ld(e, A64_STR_W, AX, PE(x));
            break;
        case P_Y:
            core_ld(e, A64_STR_W, AX, PE(y));
            break;
        case P_F:
            core_ld(e, A64_STR_W, AX, PE(f));
            break;
        case P_CNT:
            a64_shift_imm(e, A64_LSR, false, AX, AX, 16);
            core_ld(e, A64_STR_W, AX, PE(cnt));
            break;
        case P_Z:
            core_ld(e, A64_STR_W, AX, PE(z));
            a64_tst_imm(e, false, AX, 0xf8000000);
            a64_cset(e, false, DX2, A64_NE);
            core_ld(e, A64_LDR_W, CX, PE(src_mode));
            a64_tst_imm(e, false, CX, SRC_Z_SCISSOR);
            a64_cset(e, false, CX, A64_NE);
            a64_rrr(e, A64_AND, false, DX2, DX2, CX);
            core_ld(e, A64_STRB, DX2, PE(z_ovf));
            break;
        case P_U:
        case P_V:
            {
                a64_shift_imm(e, A64_LSR, false, DX2, CX, 31);
                core_ld(e, A64_STRB, DX2, param == P_U ? PE(u_dir) : PE(v_dir));
                core_ld(e, A64_LDR_W, DX2, PE(src_mode));
                uint8_t *swapped = a64_tbz(e, true, DX2, 10); /* SRC_SWAP_UV */
                emit_uv_store(e, param == P_U);
                uint8_t *done = a64_b(e);
                a64_land(swapped, e->p);
                emit_uv_store(e, param != P_U);
                a64_land(done, e->p);
                break;
            }
        case P_Q:
            emit_call(e, (const void *) risc_seed_q, true, 0);
            break;
        case P_R:
        case P_G:
        case P_B:
        case P_A:
            core_ld(e, A64_STR_W, AX, param == P_R ? PE(r) : param == P_G ? PE(g)
                                          : param == P_B   ? PE(b)
                                                           : PE(a));
            emit_sat(e);
            a64_shift_imm(e, A64_LSL, false, AX, AX, channel[param][0]);
            emit_fg_merge(e, channel[param][1]);
            break;
        case P_I:
            core_ld(e, A64_STR_W, AX, PE(r));
            core_ld(e, A64_STR_W, AX, PE(g));
            core_ld(e, A64_STR_W, AX, PE(b));
            emit_sat(e);
            a64_mov_imm(e, false, CX, 0x010101);
            a64_mul(e, false, AX, AX, CX);
            emit_fg_merge(e, 0xffffff);
            break;
        default:
            break;
    }
}

static bool
step_operands(unsigned d, unsigned s2, unsigned s1)
{
    return plain_reg(d) && plain_reg(s2) && plain_reg(s1);
}

/* Parameter steps (0xd0-0xdf). The interpreter reads every source before it
   writes the destination, and the parameter side effects never touch the
   register file, so emit all side effects first and the sums last. */
static bool
emit_step(emitter *e, uint32_t ir, bool *calls)
{
    static const pixel_param scalar[8][2] = {
        { P_XSHADOW, P_Y   },
        { P_Q,       P_Q   },
        { P_R,       P_V   },
        { P_I,       P_U   },
        { P_F,       P_F   },
        { P_G,       P_B   },
        { P_Z,       P_A   },
        { P_CNT,     P_CNT }
    };
    static const pixel_param pair[8][2] = {
        { P_XSHADOW, P_Y    },
        { P_Q,       P_U    },
        { P_R,       P_V    },
        { P_I,       P_V    },
        { P_Z,       P_V    },
        { P_G,       P_B    },
        { P_Z,       P_A    },
        { P_CNT,     P_NONE }
    };
    unsigned op = ir >> 24;
    unsigned d  = (uint8_t) (ir >> 16);
    unsigned s2 = (uint8_t) (ir >> 8);
    unsigned s1 = (uint8_t) ir;
    if (op < 0xd0 || op > 0xdf)
        return false;
    if (op < 0xd8) {
        if (!step_operands(d, s2, s1))
            return false;
        pixel_param param = scalar[op & 7][(s2 | s1) & 1];
        read_reg(e, AX, s2);
        read_reg(e, CX, s1);
        emit_param(e, param);
        *calls |= param == P_Q;
        emit_alu(e, 0, s2, s1, false);
        write_reg(e, d, AX);
        return true;
    }
    /* Pairs from SWFIFO/NPSWFIFO (32-35) are excluded by plain_reg(). */
    if (!step_operands(d & ~1U, s2 & ~1U, s1 & ~1U))
        return false;
    for (unsigned lane = 0; lane < 2; lane++) {
        pixel_param param = pair[op & 7][lane];
        read_reg(e, AX, lane ? s2 | 1 : s2 & ~1U);
        read_reg(e, CX, lane ? s1 | 1 : s1 & ~1U);
        emit_param(e, param);
        *calls |= param == P_Q;
    }
    emit_alu(e, 0, s2 & ~1U, s1 & ~1U, false);
    a64_mov(e, false, PAIR, AX);
    emit_alu(e, 0, s2 | 1, s1 | 1, false);
    write_reg(e, d & ~1U, PAIR);
    write_reg(e, d | 1, AX);
    return true;
}

static void
jit_draw(verite_core *c, uint32_t ir)
{
    verite_memory mem = { c->jit_mem->vram, c->jit_mem->mask };
    uint8_t       op  = ir >> 24;
    if (op >= 0xf0)
        pixel_draw_param(c, op, ir >> 16, ir >> 8, ir, mem);
    else
        pixel_draw_parallel(c, op, ir >> 16, ir >> 8, ir, mem);
}

static bool
draw_operand(unsigned field)
{
    /* Groups of up to ten registers stay inside one aligned bank of 16. */
    return plain_reg(field) && plain_reg(field | 15);
}

/* Parameter draws whose registers sit outside the constant and segmented
   banks get a site: an inline check of the render state against the span the
   site used last, and a direct call to it. A draw can overwrite the following
   microcode, so with check set the block leaves through the returned exit when
   the draw wrote inside [code_start, code_last] or ran in C. */
static bool
param_draw(uint32_t ir)
{
    unsigned op = ir >> 24;
    return (op >= 0xf0 && op <= 0xf3) || (op >= 0xf8 && op <= 0xfd);
}

static verite_draw_site *
draw_site(verite_core *c, uint32_t ir)
{
    static const uint8_t align[] = { 2, 2, 4, 4, 0, 0, 0, 0, 8, 8, 8, 8, 16, 16 };
    unsigned op = ir >> 24, d = (uint8_t) (ir >> 16), s2 = (uint8_t) (ir >> 8), s1 = (uint8_t) ir;
    struct verite_jit *jit = c->jit;
    if (!param_draw(ir) || jit->sites_used >= SITE_COUNT)
        return NULL;
    unsigned n = align[op - 0xf0];
    d &= ~(n - 1);
    s2 &= ~(n - 1);
    s1 &= ~(n - 1);
    /* Only single draws write back; a destination in the constant bank discards. */
    bool single = !(op & 1);
    if (s2 < 128 || s1 < 128 || (single && d >= 32 && d < 128))
        return NULL;
    verite_draw_site *s = &jit->sites[jit->sites_used++];
    *s = (verite_draw_site) {
        .ir     = ir,
        .fn     = (void *) pixel_jit_site_miss,
        .values = &c->reg[s2],
        .deltas = &c->reg[s1],
        .dest   = single && d >= 128 ? &c->reg[d] : NULL,
    };
    return s;
}

static uint8_t *
emit_site(emitter *e, verite_draw_site *site, bool check, uint32_t code_start, uint32_t code_last)
{
    static const uint32_t keys[][2] = {
        { PE(drawctl), offsetof(verite_draw_site, drawctl) },
        { PE(src_mode), offsetof(verite_draw_site, src_mode) },
        { PE(dst_mode), offsetof(verite_draw_site, dst_mode) },
        { PE(dst_fmt), offsetof(verite_draw_site, dst_fmt) },
        { PE(pmask), offsetof(verite_draw_site, pmask) },
        { PE(pat_mode), offsetof(verite_draw_site, pat_mode) },
    };
    uint64_t address = (uint64_t) (uintptr_t) site;
    uint8_t *miss[6];
    a64_mov_imm(e, true, SITE, address);
    for (unsigned i = 0; i < 6; i++) {
        core_ld(e, A64_LDR_W, AX, keys[i][0]);
        a64_ldst(e, A64_LDR_W, CX, SITE, keys[i][1]);
        a64_cmp(e, false, AX, CX);
        miss[i] = a64_bcond(e, A64_NE);
    }
    a64_ldst(e, A64_LDR_X, A64_IP0, SITE, offsetof(verite_draw_site, fn));
    uint8_t *hit = a64_b(e);
    for (unsigned i = 0; i < 6; i++)
        a64_land(miss[i], e->p);
    a64_mov_imm(e, true, A64_IP0, (uint64_t) (uintptr_t) pixel_jit_site_miss);
    a64_land(hit, e->p);

    a64_pair(e, false, 3, A64_LR, NPC, A64_SP, -32);
    a64_pair(e, false, 2, CORE, MEM, A64_SP, 16);
    a64_mov(e, true, 0, CORE);
    a64_mov(e, true, 1, SITE); /* second argument: the site */
    a64_blr(e, A64_IP0);
    a64_pair(e, true, 2, CORE, MEM, A64_SP, 16);
    a64_pair(e, true, 1, A64_LR, NPC, A64_SP, 32);
    if (!check)
        return NULL;

    a64_mov_imm(e, true, SITE, address);
    a64_ldst(e, A64_LDR_W, AX, SITE, offsetof(verite_draw_site, lo));
    a64_cmp_imm(e, false, AX, code_last);
    uint8_t *after = a64_bcond(e, A64_HI);
    a64_ldst(e, A64_LDR_W, AX, SITE, offsetof(verite_draw_site, hi));
    a64_cmp_imm(e, false, AX, code_start);
    uint8_t *below = a64_bcond(e, A64_LO);
    uint8_t *exit  = a64_b(e);
    a64_land(after, e->p);
    a64_land(below, e->p);
    return exit;
}

/* Other draws run the interpreter's pixel engine through a call, and the
   block ends after one. */
static bool
emit_draw(emitter *e, uint32_t ir)
{
    unsigned op = ir >> 24;
    bool     parallel = op == 0xe0 || op == 0xe1 || op == 0xe4 || op == 0xe5;
    bool     param    = (op >= 0xf0 && op <= 0xf3) || (op >= 0xf8 && op <= 0xfd);
    if ((!parallel && !param) || !draw_operand((uint8_t) (ir >> 16)) || !draw_operand((uint8_t) (ir >> 8)) || !draw_operand((uint8_t) ir))
        return false;
    emit_call(e, (const void *) jit_draw, false, ir);
    return true;
}

void
risc_jit_reset_sites(verite_core *c)
{
    if (c->jit)
        for (unsigned i = 0; i < c->jit->sites_used; i++)
            c->jit->sites[i].fn = (void *) pixel_jit_site_miss;
}

static bool
emit_integer(emitter *e, uint32_t ir, uint32_t pc)
{
    unsigned op  = ir >> 24;
    unsigned d   = (uint8_t) (ir >> 16);
    unsigned s2  = (uint8_t) (ir >> 8);
    unsigned s1  = (uint8_t) ir;
    unsigned sub = op & 15;

    if (op == 0x6b && plain_reg(d)) {
        if (d >= 64) {
            a64_mov_imm(e, false, AX, risc_relative(pc, 2));
            write_reg(e, d, AX);
        }
        return true;
    }
    if (op == 0x76 || op == 0x77) {
        if (!plain_reg(d))
            return false;
        if (d >= 64) {
            a64_mov_imm(e, false, AX, op == 0x76 ? ir & 0xffff : ir << 16);
            write_reg(e, d, AX);
        }
        return true;
    }
    if (op <= 0x3f) {
        bool pair      = op >= 0x20;
        bool immediate = op < 0x10;
        if (pair && sub == 15) {
            /* The interpreter still runs the FIFO readiness check for these NOPs. */
            return plain_reg(d) && plain_reg(s2) && plain_reg(s1);
        }
        if (sub == 15 || !plain_reg(s2) || (!immediate && !plain_reg(s1)) || (sub != 14 && !plain_reg(d)))
            return false;
        if (immediate && d == s2 && d >= 64 && (sub == 0 || sub == 1)) {
            if (s1) {
                operand o = reg_address(e, d);
                a64_ldst(e, A64_LDR_W, AX, o.base, o.disp);
                a64_alu_imm(e, sub == 0 ? A64_ADD : A64_SUB, false, AX, AX, s1);
                a64_ldst(e, A64_STR_W, AX, o.base, o.disp);
            }
            return true;
        }
        if (pair && sub < 8 && d >= 64 && s2 >= 64 && (op >= 0x30 || s1 >= 64)) {
            emit_packed_alu(e, sub, d, s2, s1, op < 0x30);
            return true;
        }
        if (pair) {
            emit_alu(e, sub, s2 & ~1U, op < 0x30 ? s1 & ~1U : s1, false);
            if (sub != 14 && d >= 64)
                a64_mov(e, false, PAIR, AX);
            emit_alu(e, sub, s2 | 1, op < 0x30 ? s1 | 1 : s1, false);
            if (sub != 14) {
                write_reg(e, d & ~1U, PAIR);
                write_reg(e, d | 1, AX);
            }
        } else {
            emit_alu(e, sub, s2, s1, immediate);
            if (sub != 14)
                write_reg(e, d, AX);
        }
        return true;
    }

    if (!((op >= 0x40 && op <= 0x4d && op != 0x42) || (op >= 0x52 && op <= 0x5e && op != 0x53 && op != 0x5b)) || !plain_reg(d) || !plain_reg(s2) || (op >= 0x50 && !plain_reg(s1)))
        return false;

    read_reg(e, AX, op == 0x52 ? s1 : s2);
    switch (op) {
        case 0x40:
        case 0x41:
        case 0x4b:
            a64_alu_imm(e, op == 0x41 ? A64_SUB : A64_ADD, false, AX, AX, s1 << (op == 0x4b ? 8 : 16));
            break;
        case 0x43:
            a64_mov_imm(e, false, CX, s1 << 16);
            a64_rrr(e, A64_SUB, false, AX, CX, AX);
            break;
        case 0x44:
        case 0x45:
        case 0x46:
        case 0x47:
        case 0x54:
        case 0x55:
        case 0x56:
        case 0x57:
            {
                static const uint8_t kinds[] = { A64_ROR, A64_LSL, A64_ASR, A64_LSR };
                unsigned             kind    = kinds[op & 3];
                if (op < 0x50)
                    a64_shift_imm(e, kind, false, AX, AX, s1 & 31);
                else {
                    /* Like x86, the variable shifts count modulo 32. */
                    read_reg(e, CX, s1);
                    a64_shiftv(e, kind, false, AX, AX, CX);
                }
                break;
            }
        case 0x48:
        case 0x49:
        case 0x4a:
        case 0x58:
        case 0x59:
        case 0x5a:
            {
                static const uint8_t cc[] = { A64_LT, A64_LO, A64_EQ };
                if (op < 0x50)
                    a64_cmp_imm(e, false, AX, s1);
                else {
                    read_reg(e, CX, s1);
                    a64_cmp(e, false, AX, CX);
                }
                a64_cset(e, false, AX, cc[(op & 15) - 8]);
                break;
            }
        case 0x4c:
        case 0x4d:
        case 0x5c:
        case 0x5d:
            if (op < 0x50)
                a64_mov_imm(e, false, CX, s1);
            else
                read_reg(e, CX, s1);
            a64_cmp(e, false, AX, CX);
            a64_csel(e, false, AX, CX, AX, (op & 1) ? A64_LT : A64_GT);
            break;
        case 0x52:
            /* abs, which leaves INT32_MIN alone as neg/cmovl does */
            a64_cmp_imm(e, false, AX, 0);
            a64_cneg(e, false, AX, AX, A64_LT);
            break;
        case 0x5e:
            read_reg(e, CX, s1);
            a64_rrr(e, A64_ADD, false, AX, AX, CX);
            a64_alu_imm(e, A64_AND, false, AX, AX, 0xffff0000);
            break;
    }
    write_reg(e, d, AX);
    return true;
}

static bool
emit_branch(emitter *e, uint32_t ir, uint32_t pc)
{
    unsigned op   = ir >> 24;
    unsigned d    = (uint8_t) (ir >> 16);
    unsigned s1   = (uint8_t) ir;
    uint32_t slot = risc_next(pc);
    uint32_t link = risc_next(slot);

    if (op >= 0x60 && op <= 0x65 && plain_reg(s1)) {
        static const uint8_t cc[] = { A64_EQ, A64_NE, A64_PL, A64_MI, A64_GT, A64_LE };
        read_reg(e, AX, s1);
        a64_mov_imm(e, false, NPC, link);
        a64_mov_imm(e, false, CX, risc_relative(slot, (int16_t) (ir >> 8)));
        a64_cmp_imm(e, false, AX, 0);
        a64_csel(e, false, NPC, CX, NPC, cc[op - 0x60]);
    } else if (op == 0x6f && plain_reg(s1) && plain_reg(d)) {
        read_reg(e, NPC, s1);
        a64_mov_imm(e, false, AX, link);
        write_reg(e, d, AX);
    } else if (op == 0x68 || op == 0x6a || op == 0x6c || op == 0x6e) {
        if (op == 0x6a || op == 0x6e)
            core_imm(e, GPR(RISC_RA), link);
        a64_mov_imm(e, false, NPC, op < 0x6c ? risc_relative(slot, (int32_t) (ir << 8) >> 8) : risc_absolute(ir));
    } else {
        return false;
    }
    return true;
}

/* A failed address guard branches to an epilogue for the completed prefix.
   The interpreter then executes the original load/store with full bus timing. */
static uint8_t *
emit_memory(emitter *e, uint32_t ir, bool *store)
{
    unsigned op    = ir >> 24;
    unsigned d     = (uint8_t) (ir >> 16);
    unsigned s2    = (uint8_t) (ir >> 8);
    unsigned s1    = (uint8_t) ir;
    bool     write = (op >= 0x78 && op <= 0x7a) || op == 0x7c;
    bool     load  = (op >= 0x70 && op <= 0x72) || op == 0x74 || op == 0x75;
    bool     pair  = op == 0x74 || op == 0x75 || op == 0x7c;
    if ((!write && !load) || !plain_reg(s1) || (pair ? (write ? s2 : d) < 64 : !plain_reg(write ? s2 : d)))
        return NULL;

    unsigned size   = pair ? 8 : 1U << (op & 3);
    unsigned offset = size * (write ? d : s2);
    read_reg(e, AX, s1);
    if (offset)
        a64_alu_imm(e, A64_ADD, false, AX, AX, offset);
    a64_cmp_imm(e, false, AX, LOCAL_END);
    uint8_t *exit = a64_bcond(e, A64_HS); /* slow path */

    a64_ldst(e, A64_LDR_W, CX, MEM, offsetof(jit_memory, mask));
    a64_rrr(e, A64_AND, false, AX, AX, CX);
    a64_ldst(e, A64_LDR_X, VRAM, MEM, offsetof(jit_memory, vram));
    if (size != 1)
        a64_alu_imm(e, A64_AND, false, AX, AX, ~(size - 1));
    if (size < 4)
        a64_alu_imm(e, A64_EOR, false, AX, AX, 4 - size);

    if (pair) {
        if (write) {
            packed_reg(e, 0, s2, false);
            a64_ldst_reg(e, A64_STR_D, 0, VRAM, AX, A64_UXTW, false);
        } else {
            a64_ldst_reg(e, A64_LDR_D, 0, VRAM, AX, A64_UXTW, false);
            packed_reg(e, 0, d, true);
        }
    } else if (write) {
        read_reg(e, CX, s2);
        a64_ldst_reg(e, size == 1 ? A64_STRB : size == 2 ? A64_STRH : A64_STR_W, CX, VRAM, AX, A64_UXTW, false);
    } else {
        a64_ldst_reg(e, size == 1 ? A64_LDRB : size == 2 ? A64_LDRH : A64_LDR_W, CX, VRAM, AX, A64_UXTW, false);
        write_reg(e, d, CX);
    }

    if (write) {
        a64_mov_imm(e, false, CX, 1);
        core_ld(e, A64_STRB, CX, CORE_FIELD(drew));
        /* A store may overwrite the next RISC instruction. End the block. */
        *store = true;
    }
    return exit;
}

/* timer += W0 */
static void
add_timer(emitter *e)
{
    core_ld(e, A64_LDR_W, CX, CORE_FIELD(timer));
    a64_rrr(e, A64_ADD, false, CX, CX, AX);
    core_ld(e, A64_STR_W, CX, CORE_FIELD(timer));
}

static void
emit_finish(emitter *e, uint32_t pc, unsigned count, bool branch, bool delay_in_block, bool loop)
{
    if (!count) {
        a64_mov_imm(e, false, AX, 0);
        a64_ret(e);
        return;
    }
    if (branch && delay_in_block) {
        core_ld(e, A64_STR_W, NPC, CORE_FIELD(pc));
        /* npc = risc_next(npc) */
        a64_alu_imm(e, A64_ADD, false, AX, NPC, 4);
        a64_alu_imm(e, A64_AND, false, NPC, NPC, 0xff000000);
        a64_alu_imm(e, A64_AND, false, AX, AX, 0x00ffffff);
        a64_rrr(e, A64_ORR, false, NPC, NPC, AX);
    } else {
        core_imm(e, CORE_FIELD(pc), pc);
    }
    if (branch)
        core_ld(e, A64_STR_W, NPC, CORE_FIELD(npc));
    else
        core_imm(e, CORE_FIELD(npc), risc_next(pc));

    if (loop) {
        core_ld(e, A64_LDR_W, AX, CORE_FIELD(jit_budget));
        a64_rrr(e, A64_SUB, false, AX, AX, MEM);
    } else
        a64_mov_imm(e, false, AX, count);
    add_timer(e);
    a64_ret(e);
}

/* Leave after instruction count - 1 of a possibly looping block. A loop has
   already run jit_budget - W10 cycles in earlier iterations. */
static void
emit_side_finish(emitter *e, uint32_t pc, unsigned count, bool loop)
{
    if (!loop) {
        emit_finish(e, pc, count, false, false, false);
        return;
    }
    core_imm(e, CORE_FIELD(pc), pc);
    core_imm(e, CORE_FIELD(npc), risc_next(pc));
    core_ld(e, A64_LDR_W, AX, CORE_FIELD(jit_budget));
    a64_rrr(e, A64_SUB, false, AX, AX, MEM);
    a64_alu_imm(e, A64_ADD, false, AX, AX, count);
    add_timer(e);
    a64_ret(e);
}

void
risc_jit_free(verite_core *c)
{
    pixel_jit_free(c);
    if (c->jit) {
        a64_free_code(c->jit->arena, ARENA_SIZE);
        free(c->jit);
        c->jit = NULL;
    }
}

static struct verite_jit *
jit_create(void)
{
    struct verite_jit *jit = calloc(1, sizeof(*jit));
    if (jit && !(jit->arena = a64_alloc_code(ARENA_SIZE))) {
        free(jit);
        return NULL;
    }
    return jit;
}

static jit_block *
jit_lookup(struct verite_jit *jit, uint32_t pc)
{
    return &jit->blocks[((pc >> 2) ^ (pc >> 14) ^ (pc >> 24)) & (CACHE_SIZE - 1)];
}

/* Keep the emission buffer in the cold compilation path. */
#    if defined(_MSC_VER)
__declspec(noinline)
#    else
__attribute__((noinline))
#    endif
static jit_block *
compile_block(verite_core *c, verite_memory mem, unsigned limit, bool fast_memory)
{
    struct verite_jit *jit = c->jit;
    uint32_t           code_words[MAX_BLOCK_CODE / 4];
    uint8_t           *code = (uint8_t *) code_words;
    emitter            e    = { code };
    uint32_t           words[BLOCK_WORDS];
    uint8_t           *exits[BLOCK_WORDS] = { 0 };
    uint8_t           *after[BLOCK_WORDS] = { 0 };
    uint32_t           start              = c->pc;
    bool               local              = start < LOCAL_END;
    uint32_t           code_start         = (uint32_t) (start & mem.mask);
    uint32_t           code_last          = code_start + limit * 4 - 1;
    uint32_t           pc                 = start;
    unsigned           count              = 0;
    bool               branch = false, delay_in_block = false;
    bool               uses_memory = false, calls = false;
    unsigned           exit_count  = 0;
    bool               backedge = false, conditional = false;

    a64_mov(&e, true, CORE, 0);
    /* Reserve space for the memory argument or loop budget, filled in
       once we know the shape of the block. */
    uint8_t *prologue = e.p;
    e.p += PROLOGUE_SLOTS * 4;
    uint8_t *body = e.p;
    while (count < limit) {
        if ((size_t) (e.p - code) + MAX_INSN_CODE + (exit_count + 2) * FINISH_CODE > MAX_BLOCK_CODE)
            break;
        uint32_t ir  = risc_fetch(c, pc, mem);
        words[count] = ir;
        if (emit_integer(&e, ir, pc) || emit_step(&e, ir, &calls)) {
            pc = risc_next(pc);
            count++;
            if (branch) {
                delay_in_block = true;
                break;
            }
        } else if (draw_site(c, ir)) {
            /* The site was taken from the pool above; a check is needed only
               when instructions follow it in the block. */
            verite_draw_site *site = &jit->sites[jit->sites_used - 1];
            after[count]           = emit_site(&e, site, local && !branch, code_start, code_last);
            exit_count += after[count] != NULL;
            calls = true;
            pc    = risc_next(pc);
            count++;
            if (branch) {
                delay_in_block = true;
                break;
            }
        } else if (emit_draw(&e, ir)) {
            calls = true;
            pc    = risc_next(pc);
            count++;
            delay_in_block = branch;
            break;
        } else if (!branch && emit_branch(&e, ir, pc)) {
            pc = risc_next(pc);
            count++;
            branch      = true;
            unsigned op = ir >> 24;
            conditional = op >= 0x60 && op <= 0x65;
            if (conditional)
                backedge = risc_relative(pc, (int16_t) (ir >> 8)) == start;
            else if (op == 0x68 || op == 0x6a)
                backedge = risc_relative(pc, (int32_t) (ir << 8) >> 8) == start;
            else if (op == 0x6c || op == 0x6e)
                backedge = risc_absolute(ir) == start;
        } else {
            bool store = false;
            if (branch || !fast_memory || !(exits[count] = emit_memory(&e, ir, &store)))
                break;
            uses_memory = true;
            exit_count++;
            pc          = risc_next(pc);
            count++;
            if (store)
                break;
        }
    }
    (void) calls;

    jit_block *block   = jit_lookup(jit, start);
    block->pc          = start;
    block->fn          = NULL;
    block->count       = count < 2 ? 1 : count;
    block->fast_memory = fast_memory;
    memcpy(block->words, words, block->count * sizeof(*words));
    if (count < 2)
        return block;

    bool    loop  = backedge && delay_in_block && !uses_memory;
    emitter setup = { prologue };
    if (uses_memory) {
        a64_mov(&setup, true, MEM, 1);
    } else if (loop) {
        a64_mov(&setup, false, MEM, 2);
        core_ld(&setup, A64_STR_W, MEM, CORE_FIELD(jit_budget));

        /* W10 holds remaining cycles across iterations. Draw sites leave the
           loop when they write into this block, so its snapshot stays valid. */
        a64_alu_imm(&e, A64_SUB, false, MEM, MEM, count);
        a64_cmp_imm(&e, false, MEM, count);
        if (conditional) {
            uint8_t *skip_loop = a64_bcond(&e, A64_LO);
            a64_cmp_imm(&e, false, NPC, start);
            a64_bcond_to(&e, A64_EQ, body);
            a64_land(skip_loop, e.p);
        } else
            a64_bcond_to(&e, A64_HS, body);
    }
    if ((size_t) (setup.p - prologue) > PROLOGUE_SLOTS * 4)
        abort();
    emit_finish(&e, pc, count, branch, delay_in_block, loop);
    for (unsigned i = 0; i < count; i++) {
        if (exits[i]) {
            a64_land(exits[i], e.p);
            emit_finish(&e, risc_relative(start, i), i, false, false, false);
        }
        if (after[i]) {
            a64_land(after[i], e.p);
            emit_side_finish(&e, risc_relative(start, i + 1), i + 1, loop);
        }
    }

    /* Branches stay inside the body, so it moves up over the unused slots. */
    size_t   prefix = setup.p - code;
    size_t   size   = prefix + (e.p - body);
    uint8_t *dest   = jit->arena + jit->used;
    a64_write_begin();
    memcpy(dest, code, prefix);
    memcpy(dest + prefix, body, e.p - body);
    a64_write_end(dest, size);
    jit->used += (size + 15) & ~(size_t) 15;
    block->fn = (block_fn) (void *) dest;
    return block;
}

unsigned
risc_jit_run(verite_core *c, verite_memory mem, unsigned budget)
{
    uint32_t pc = c->pc;
    if (c->jit_disabled || c->jit_unavailable || budget < 2 || (pc & 3) || c->npc != risc_next(pc))
        return 0;

    bool local            = pc < LOCAL_END;
    bool canonical_memory = mem.mask >= 3 && !(mem.mask & (mem.mask + 1));
    if (local && !canonical_memory)
        return 0;
    if (!c->jit) {
        c->jit = jit_create();
        if (!c->jit) {
            c->jit_unavailable = true;
            return 0;
        }
    }

    struct verite_jit *jit   = c->jit;
    jit_block         *block = jit_lookup(jit, pc);
    uint32_t           addr  = local ? pc & mem.mask : pc;
    unsigned           limit = (0x1000 - (addr & 0xfff)) / 4;
    if (local && mem.mask < 0xfff)
        limit = (mem.mask + 1 - addr) / 4;
    if (limit > BLOCK_WORDS)
        limit = BLOCK_WORDS;
    if (limit < 2)
        return 0;

    /* Validate once per entry for writes from the CPU, RISC and renderer.
       A native store can run only as the final instruction. */
    bool fast_memory = !c->slow_precharge && canonical_memory && mem.mask >= 7;
    bool match       = block->count && block->pc == pc && block->count <= limit && block->fast_memory == fast_memory;
    if (match && local) {
        /* Interpreter-only sites cache just the first instruction. */
        if (block->count == 1)
            match = !memcmp(mem.data + addr, block->words, sizeof(uint32_t));
        else
            match = !memcmp(mem.data + addr, block->words, block->count * sizeof(uint32_t));
    }
    if (!match) {
        if (jit->used > ARENA_SIZE - MAX_BLOCK_CODE - 15 || jit->sites_used > SITE_COUNT - BLOCK_WORDS) {
            memset(jit->blocks, 0, sizeof(jit->blocks));
            jit->used       = 0;
            jit->sites_used = 0;
        }
        block = compile_block(c, mem, limit, fast_memory);
    }

    unsigned wait   = local && c->slow_precharge && (addr >> 12) != c->dram_row;
    unsigned cycles = block->count + wait;
    if (!block->fn || cycles > budget)
        return 0;
    jit_memory memory   = { mem.data, (uint32_t) mem.mask };
    c->jit_mem          = &memory;
    unsigned executed   = block->fn(c, &memory, budget - wait);
    c->jit_mem          = NULL;
    if (!executed)
        return 0;
    if (wait) {
        c->dram_row = addr >> 12;
        c->timer++;
    }
    c->dram_wait = executed == 1 ? wait : 0;
    return executed + wait;
}

#endif
