/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Native basic blocks for the Vérité RISC. */

#include "internal.h"

#ifdef VERITE_JIT_X64

#    include <stdlib.h>
#    ifdef _WIN32
#        include <windows.h>
#    else
#        include <sys/mman.h>
#    endif

enum {
    BLOCK_WORDS = 32,
    /* Includes two segmented lanes or a memory access with its exit stub. */
    MAX_BLOCK_CODE = BLOCK_WORDS * 256 + 128,
    /* Room kept free for the largest single instruction and each epilogue. */
    MAX_INSN_CODE = 1024,
    FINISH_CODE   = 96,
    CACHE_SIZE    = 4096,
    ARENA_SIZE    = 4 * 1024 * 1024,
    SITE_COUNT    = 8192,
};

typedef verite_jit_memory jit_memory;

typedef unsigned (*block_fn)(verite_core *, const jit_memory *, unsigned);

typedef struct {
    block_fn fn;
    uint32_t pc;
    unsigned count;
    uint32_t page_gen, global_gen;
    uint8_t  mem_mode, max_wait;
    uint32_t words[BLOCK_WORDS];
} jit_block;

struct verite_jit {
    uint8_t         *arena;
    size_t           used;
    unsigned         sites_used;
    jit_block        blocks[CACHE_SIZE];
    verite_draw_site sites[SITE_COUNT];
};

typedef struct {
    uint8_t *pos;
} emitter;

/* R10 holds the core. R11 indexes segmented operands. EDX is reserved for
   the next PC across a branch's delay slot. R9 holds the loop budget or
   slow-precharge extra_wait. All are caller-saved in both ABIs. */
enum { AX    = 0,
       CX    = 1,
       DX    = 2,
       SP    = 4,
       SI    = 6,
       DI    = 7,
       R8    = 8,
       MEM   = 9,
       CORE  = 10,
       INDEX = 11 };

#    ifdef _WIN32
enum { ARG_CORE   = CX,
       ARG_MEMORY = DX,
       ARG_BUDGET = R8 };
#    else
enum { ARG_CORE   = DI,
       ARG_MEMORY = SI,
       ARG_BUDGET = DX };
#    endif

#    define CORE_FIELD(field) ((uint32_t) offsetof(verite_core, field))
#    define GPR(field)        (CORE_FIELD(reg) + (field) * 4U)
#    define PE(field)         CORE_FIELD(pe.field)

static void
emit_byte(emitter *e, uint8_t value)
{
    *e->pos++ = value;
}

static void
dword(emitter *e, uint32_t value)
{
    memcpy(e->pos, &value, sizeof(value));
    e->pos += sizeof(value);
}

static void
rel32(uint8_t *operand, const uint8_t *target)
{
    uint32_t displacement = (uint32_t) (target - (operand + 4));
    memcpy(operand, &displacement, sizeof(displacement));
}

static void
rex(emitter *e, bool wide, unsigned reg, unsigned index, unsigned base)
{
    unsigned prefix = 0x40 | (wide << 3) | ((reg >> 3) << 2) | ((index >> 3) << 1) | (base >> 3);
    if (prefix != 0x40)
        emit_byte(e, prefix);
}

static void
opcode(emitter *e, unsigned op)
{
    if (op > 0xff)
        emit_byte(e, op >> 8);
    emit_byte(e, op);
}

static void
rr(emitter *e, unsigned op, unsigned reg, unsigned rm, bool wide)
{
    rex(e, wide, reg, 0, rm);
    opcode(e, op);
    emit_byte(e, 0xc0 | ((reg & 7) << 3) | (rm & 7));
}

static void
core_mem(emitter *e, unsigned op, unsigned reg, uint32_t disp, bool indexed)
{
    rex(e, false, reg, indexed ? INDEX : 0, CORE);
    opcode(e, op);
    emit_byte(e, 0x80 | ((reg & 7) << 3) | (indexed ? 4 : (CORE & 7)));
    if (indexed)
        emit_byte(e, 0x80 | ((INDEX & 7) << 3) | (CORE & 7));
    dword(e, disp);
}

static void
core_mem64(emitter *e, unsigned op, unsigned reg, uint32_t disp)
{
    rex(e, true, reg, 0, CORE);
    opcode(e, op);
    emit_byte(e, 0x80 | ((reg & 7) << 3) | (CORE & 7));
    dword(e, disp);
}

static void
imm(emitter *e, unsigned reg, uint32_t value)
{
    rex(e, false, 0, 0, reg);
    emit_byte(e, 0xb8 | (reg & 7));
    dword(e, value);
}

static void
alu_imm(emitter *e, unsigned op, unsigned reg, uint32_t value)
{
    int32_t signed_value = (int32_t) value;
    if (signed_value >= -128 && signed_value <= 127) {
        rr(e, 0x83, op, reg, false);
        emit_byte(e, value);
    } else {
        rr(e, 0x81, op, reg, false);
        dword(e, value);
    }
}

static void
shift(emitter *e, unsigned op, unsigned reg, uint8_t count, bool wide)
{
    if (count) {
        rr(e, 0xc1, op, reg, wide);
        emit_byte(e, count);
    }
}

static void
core_imm(emitter *e, uint32_t disp, uint32_t value)
{
    core_mem(e, 0xc7, 0, disp, false);
    dword(e, value);
}

static bool
segmented(unsigned field)
{
    return (field & 0xc0) == 0x40;
}

static uint32_t
reg_address(emitter *e, unsigned field)
{
    if (!segmented(field))
        return GPR(field);

    unsigned bank = (field >> 4) & 3;
    core_mem(e, 0x8b, INDEX, CORE_FIELD(seg), false);
    shift(e, bank ? 5 : 4, INDEX, bank ? (bank - 1) * 4 : 4, false);
    alu_imm(e, 4, INDEX, 0x70);
    return GPR(128 + (field & 15));
}

static bool
is_fifo_field(unsigned field)
{
    return field == 32 || field == 33;
}

static bool
readable_reg(unsigned field)
{
    if (field < 32 || field >= 64)
        return true;
    return (field >= 32 && field <= 38) || field == 40 || field == 41 || field == 46 || field == 47
        || field == 56 || field == 57;
}

static bool
readable_nonfifo(unsigned field)
{
    return readable_reg(field) && !is_fifo_field(field);
}

static bool
writable_reg(unsigned field)
{
    return field != 33 && field != 39;
}

/* Pop from input FIFO into host. Caller must have checked in_count >= pops.
   Clobbers R8 (and host). */
static void
emit_pop_fifo(emitter *e, unsigned host, bool swap)
{
    core_mem(e, 0x8b, R8, CORE_FIELD(in_head), false);
    rex(e, false, host, R8, CORE);
    opcode(e, 0x8b);
    emit_byte(e, 0x84 | ((host & 7) << 3));
    emit_byte(e, (2 << 6) | ((R8 & 7) << 3) | (CORE & 7));
    dword(e, CORE_FIELD(input));
    alu_imm(e, 0, R8, 1);
    alu_imm(e, 4, R8, IN_FIFO - 1);
    core_mem(e, 0x89, R8, CORE_FIELD(in_head), false);
    core_mem(e, 0x8b, R8, CORE_FIELD(in_count), false);
    alu_imm(e, 5, R8, 1);
    core_mem(e, 0x89, R8, CORE_FIELD(in_count), false);

    /* if (IN_FIFO - in_count == (regs[VIO_LOW_WATER_MARK] & 31) + 1) intr |= VINTR_FIFO_LOW */
    imm(e, INDEX, IN_FIFO);
    rr(e, 0x29, R8, INDEX, false);
    core_mem(e, 0x0fb6, R8, CORE_FIELD(regs) + VIO_LOW_WATER_MARK, false);
    alu_imm(e, 4, R8, 31);
    alu_imm(e, 0, R8, 1);
    rr(e, 0x39, R8, INDEX, false);
    opcode(e, 0x0f85);
    uint8_t *no_low = e->pos;
    dword(e, 0);
    emit_byte(e, 0x66);
    core_mem(e, 0x83, 1, CORE_FIELD(intr), false);
    emit_byte(e, VINTR_FIFO_LOW);
    rel32(no_low, e->pos);
    if (swap) {
        rex(e, false, 0, 0, host);
        emit_byte(e, 0x0f);
        emit_byte(e, 0xc8 | (host & 7)); /* bswap host */
    }
}

/* Read special register (34..57, non-pop) into host. Clobbers R8. */
static void
emit_special_read(emitter *e, unsigned host, unsigned field)
{
    switch (field) {
        case 34: /* SEED */
            core_mem(e, 0x8b, host, CORE_FIELD(seed), false);
            break;
        case 35: /* SEG */
            core_mem(e, 0x8b, host, CORE_FIELD(seg), false);
            alu_imm(e, 4, host, 0xffff);
            core_mem(e, 0x8b, R8, CORE_FIELD(seed_index), false);
            shift(e, 4, R8, 19, false);
            rr(e, 0x09, R8, host, false);
            alu_imm(e, 1, host, 0x8888);
            break;
        case 36: /* SCALE */
            core_mem(e, 0x8b, host, CORE_FIELD(scale), false);
            break;
        case 37: /* FLAG */
            core_mem(e, 0x8b, host, CORE_FIELD(flag), false);
            core_mem(e, 0x8b, R8, CORE_FIELD(out_count), false);
            alu_imm(e, 7, R8, OUT_FIFO);
            rr(e, 0x0f92, 0, R8, false);
            rr(e, 0x0fb6, R8, R8, false);
            shift(e, 4, R8, 2, false);
            rr(e, 0x09, R8, host, false);
            core_mem(e, 0x0fb6, R8, CORE_FIELD(regs) + VIO_COMM, false);
            alu_imm(e, 4, R8, 15);
            shift(e, 4, R8, 8, false);
            rr(e, 0x09, R8, host, false);
            core_mem(e, 0x0fb6, R8, PE(pick_valid), false);
            shift(e, 4, R8, 16, false);
            rr(e, 0x09, R8, host, false);
            core_mem(e, 0x0fb6, R8, PE(pick_hit), false);
            shift(e, 4, R8, 17, false);
            rr(e, 0x09, R8, host, false);
            core_mem(e, 0x8b, R8, CORE_FIELD(in_count), false);
            rr(e, 0x85, R8, R8, false);
            rr(e, 0x0f95, 0, R8, false);
            rr(e, 0x0fb6, R8, R8, false);
            shift(e, 4, R8, 31, false);
            rr(e, 0x09, R8, host, false);
            break;
        case 38: /* CLIP */
            core_mem(e, 0x8b, host, CORE_FIELD(clip), false);
            alu_imm(e, 4, host, 0xfc);
            break;
        case 40: /* NPSWFIFO */
        case 41: /* NPFIFO */
            core_mem(e, 0x8b, R8, CORE_FIELD(in_count), false);
            rr(e, 0x85, R8, R8, false);
            opcode(e, 0x0f84);
            uint8_t *empty = e->pos;
            dword(e, 0);
            core_mem(e, 0x8b, R8, CORE_FIELD(in_head), false);
            rex(e, false, host, R8, CORE);
            opcode(e, 0x8b);
            emit_byte(e, 0x84 | ((host & 7) << 3));
            emit_byte(e, (2 << 6) | ((R8 & 7) << 3) | (CORE & 7));
            dword(e, CORE_FIELD(input));
            if (field == 40) {
                rex(e, false, 0, 0, host);
                emit_byte(e, 0x0f);
                emit_byte(e, 0xc8 | (host & 7));
            }
            emit_byte(e, 0xeb);
            uint8_t *done = e->pos++;
            rel32(empty, e->pos);
            rr(e, 0x31, host, host, false);
            *done = (uint8_t) (e->pos - (done + 1));
            break;
        case 46: /* WINCLIP */
            core_mem(e, 0x8b, host, CORE_FIELD(clip), false);
            alu_imm(e, 4, host, 0x3c);
            break;
        case 47: /* RISCINTR */
            core_mem(e, 0x8b, host, CORE_FIELD(riscintr), false);
            rr(e, 0x89, host, R8, false);
            shift(e, 5, R8, 16, false);
            rr(e, 0x21, host, R8, false);
            alu_imm(e, 4, R8, 0x3f);
            rr(e, 0x0f95, 0, R8, false);
            rr(e, 0x0fb6, R8, R8, false);
            shift(e, 4, R8, 31, false);
            rr(e, 0x09, R8, host, false);
            break;
        case 56:
            imm(e, host, 0xfffff000U);
            break;
        case 57:
            imm(e, host, 0xfffff800U);
            break;
        default:
            rr(e, 0x31, host, host, false);
            break;
    }
}

static void
read_reg(emitter *e, unsigned host, unsigned field)
{
    if (field < 32) {
        uint32_t value = risc_constant(field);
        if (value)
            imm(e, host, value);
        else
            rr(e, 0x31, host, host, false);
    } else if (field >= 64) {
        uint32_t disp = reg_address(e, field);
        core_mem(e, 0x8b, host, disp, segmented(field));
    } else if (is_fifo_field(field)) {
        emit_pop_fifo(e, host, field == 32);
    } else {
        emit_special_read(e, host, field);
    }
}

/* Destinations in the constant bank or non-side-effect special regs discard. */
static void
write_reg(emitter *e, unsigned field, unsigned host)
{
    if (field >= 64) {
        uint32_t disp = reg_address(e, field);
        core_mem(e, 0x89, host, disp, segmented(field));
        return;
    }
    switch (field) {
        case 35: /* SEG */
            rr(e, 0x89, host, R8, false);
            alu_imm(e, 4, R8, 0xffff);
            core_mem(e, 0x89, R8, CORE_FIELD(seg), false);
            break;
        case 37: /* FLAG */
            rr(e, 0xf7, 0, host, false);
            dword(e, 1);
            opcode(e, 0x0f84);
            uint8_t *no_intr = e->pos;
            dword(e, 0);
            emit_byte(e, 0x66);
            core_mem(e, 0x83, 1, CORE_FIELD(intr), false);
            emit_byte(e, VINTR_RISC);
            rel32(no_intr, e->pos);
            core_mem(e, 0x8b, R8, CORE_FIELD(flag), false);
            alu_imm(e, 4, R8, 1);
            rr(e, 0x89, host, INDEX, false);
            alu_imm(e, 4, INDEX, 0x400000f9U);
            rr(e, 0x09, R8, INDEX, false);
            core_mem(e, 0x89, INDEX, CORE_FIELD(flag), false);
            break;
        case 40: /* DRAWCTL */
            rr(e, 0x89, host, R8, false);
            alu_imm(e, 4, R8, 0xffff);
            core_mem(e, 0x89, R8, PE(drawctl), false);
            break;
        case 47: /* RISCINTR */
            rr(e, 0x89, host, R8, false);
            alu_imm(e, 4, R8, 0x3f);
            rr(e, 0xf7, 2, R8, false); /* not r8d */
            core_mem(e, 0x8b, INDEX, CORE_FIELD(riscintr), false);
            rr(e, 0x21, R8, INDEX, false);
            alu_imm(e, 4, INDEX, 0xffff);
            rr(e, 0x89, host, R8, false);
            alu_imm(e, 4, R8, 0x3f0000);
            rr(e, 0x09, R8, INDEX, false);
            core_mem(e, 0x89, INDEX, CORE_FIELD(riscintr), false);
            break;
        default:
            break;
    }
}

static bool
plain_reg(unsigned field)
{
    return field < 32 || field >= 64;
}

static uint8_t *
emit_fifo_guard(emitter *e, unsigned pops)
{
    core_mem(e, 0x83, 7, CORE_FIELD(in_count), false);
    emit_byte(e, pops);
    opcode(e, 0x0f82); /* jb stall_exit */
    uint8_t *at = e->pos;
    dword(e, 0);
    return at;
}

static void
packed_reg(emitter *e, unsigned xmm, unsigned field, bool write)
{
    uint32_t disp = reg_address(e, field & ~1U);
    emit_byte(e, write ? 0x66 : 0xf3);
    core_mem(e, write ? 0x0fd6 : 0x0f7e, xmm, disp, segmented(field));
}

static void
emit_clip(emitter *e)
{
    rr(e, 0x39, CX, AX, false);   /* cmp eax, ecx */
    rr(e, 0x0f9f, 0, AX, false);  /* setg al */
    rr(e, 0x0fb6, AX, AX, false); /* movzx eax, al */
    core_mem(e, 0x8b, CX, CORE_FIELD(clip), false);
    shift(e, 4, CX, 1, false);
    alu_imm(e, 4, CX, 0xfc);
    shift(e, 4, AX, 2, false);
    rr(e, 0x09, AX, CX, false);
    core_mem(e, 0x89, CX, CORE_FIELD(clip), false);
}

static void
emit_alu_regs(emitter *e, unsigned op, bool immediate, uint32_t imm_val)
{
    static const uint8_t binary[]        = { 0x01, 0x29, 0x21, 0x29, 0x21, 0x09, 0x09, 0x31 };
    static const uint8_t group[]         = { 0, 5, 4, 0, 4, 1, 1, 6 };
    static const uint8_t product_shift[] = { 0, 8, 16, 24, 32, 31 };

    if (op < 8) {
        if (immediate && op != 3)
            alu_imm(e, group[op], AX, op == 2 ? ~imm_val : imm_val);
        else {
            if (immediate)
                imm(e, CX, imm_val);
            if (op == 2)
                rr(e, 0xf7, 2, CX, false);
            rr(e, binary[op], op == 3 ? AX : CX, op == 3 ? CX : AX, false);
            if (op == 3)
                rr(e, 0x89, CX, AX, false);
        }
        if (op == 6)
            rr(e, 0xf7, 2, AX, false);
        return;
    }
    if (immediate)
        imm(e, CX, imm_val);
    if (op == 14) {
        emit_clip(e);
        return;
    }
    bool wide = op != 8;
    if (wide) {
        rr(e, 0x63, AX, AX, true);
        rr(e, 0x63, CX, CX, true);
    }
    rr(e, 0x0faf, AX, CX, wide);
    shift(e, 7, AX, product_shift[op - 8], wide);
}

static void
emit_alu(emitter *e, unsigned op, unsigned s2, unsigned s1, bool immediate)
{
    if (op == 14 && !immediate && is_fifo_field(s1)) {
        read_reg(e, CX, s1);
        read_reg(e, AX, s2);
    } else {
        read_reg(e, AX, s2);
        if (!immediate && !(op < 8 && immediate))
            read_reg(e, CX, s1);
    }
    emit_alu_regs(e, op, immediate, s1);
}

static void
emit_packed_alu(emitter *e, unsigned sub, unsigned d, unsigned s2, unsigned s1, bool pair_b)
{
    static const uint16_t ops[] = { 0x0ffe, 0x0ffa, 0x0fdf, 0x0ffa, 0x0fdb, 0x0feb, 0x0feb, 0x0fef };
    packed_reg(e, 0, s2, false);
    if (pair_b) {
        packed_reg(e, 1, s1, false);
    } else {
        read_reg(e, AX, s1);
        emit_byte(e, 0x66);
        rr(e, 0x0f6e, 1, AX, false);
        emit_byte(e, 0x66);
        rr(e, 0x0f70, 1, 1, false);
        emit_byte(e, 0);
    }
    unsigned result = sub == 2 || sub == 3;
    emit_byte(e, 0x66);
    rr(e, ops[sub], result, result ^ 1, false);
    if (sub == 6) {
        emit_byte(e, 0x66);
        rr(e, 0x0f76, 1, 1, false);
        emit_byte(e, 0x66);
        rr(e, 0x0fef, 0, 1, false);
    }
    packed_reg(e, result, d, true);
}

/* Call fn(core, arg1, core->jit_mem). RDX keeps the next PC in a delay slot. */
static void
emit_call(emitter *e, const void *fn, bool from_ax, uint32_t arg1)
{
    emit_byte(e, 0x41); /* push r10 */
    emit_byte(e, 0x50 | (CORE & 7));
    emit_byte(e, 0x41); /* push r9 */
    emit_byte(e, 0x50 | (MEM & 7));
    emit_byte(e, 0x50 | DX); /* push rdx */
    rr(e, 0x83, 5, SP, true); /* sub rsp, 32 */
    emit_byte(e, 32);
    core_mem64(e, 0x8b, ARG_BUDGET, CORE_FIELD(jit_mem));
    if (from_ax)
        rr(e, 0x89, AX, ARG_MEMORY, false);
    else
        imm(e, ARG_MEMORY, arg1);
    rr(e, 0x89, CORE, ARG_CORE, true);
    emit_byte(e, 0x48); /* mov rax, fn */
    emit_byte(e, 0xb8);
    uint64_t address = (uint64_t) (uintptr_t) fn;
    memcpy(e->pos, &address, sizeof(address));
    e->pos += sizeof(address);
    rr(e, 0xff, 2, AX, false); /* call rax */
    rr(e, 0x83, 0, SP, true);  /* add rsp, 32 */
    emit_byte(e, 32);
    emit_byte(e, 0x58 | DX);
    emit_byte(e, 0x41);
    emit_byte(e, 0x58 | (MEM & 7));
    emit_byte(e, 0x41);
    emit_byte(e, 0x58 | (CORE & 7));
}

/* Call fn(eax, ecx) -> eax. */
static void
emit_call2_ret(emitter *e, const void *fn)
{
    emit_byte(e, 0x41); /* push r10 */
    emit_byte(e, 0x50 | (CORE & 7));
    emit_byte(e, 0x41); /* push r9 */
    emit_byte(e, 0x50 | (MEM & 7));
    emit_byte(e, 0x50 | DX); /* push rdx */
    rr(e, 0x83, 5, SP, true);
    emit_byte(e, 32);
#    ifdef _WIN32
    /* Win64: arg0 = ECX, arg1 = EDX. Move ECX -> EDX first! */
    rr(e, 0x89, CX, DX, false);
    rr(e, 0x89, AX, CX, false);
#    else
    rr(e, 0x89, AX, DI, false);
    rr(e, 0x89, CX, SI, false);
#    endif
    emit_byte(e, 0x48);
    emit_byte(e, 0xb8);
    uint64_t address = (uint64_t) (uintptr_t) fn;
    memcpy(e->pos, &address, sizeof(address));
    e->pos += sizeof(address);
    rr(e, 0xff, 2, AX, false);
    rr(e, 0x83, 0, SP, true);
    emit_byte(e, 32);
    emit_byte(e, 0x58 | DX);
    emit_byte(e, 0x41);
    emit_byte(e, 0x58 | (MEM & 7));
    emit_byte(e, 0x41);
    emit_byte(e, 0x58 | (CORE & 7));
}

/* Inline seed_op(c, R8D). Clobbers RAX, R8 (and RDX if !save_dx, else preserves
   RDX). Note: caller puts x in R8D! */
static void
emit_seed_op(emitter *e, bool save_dx)
{
    if (save_dx)
        emit_byte(e, 0x50 | DX);
    rr(e, 0x85, R8, R8, false);
    opcode(e, 0x0f85);
    uint8_t *nonzero = e->pos;
    dword(e, 0);
    core_imm(e, CORE_FIELD(scale), 0);
    core_imm(e, CORE_FIELD(seed_index), 0);
    core_imm(e, CORE_FIELD(seed), UINT32_MAX);
    emit_byte(e, 0xe9);
    uint8_t *done = e->pos;
    dword(e, 0);

    rel32(nonzero, e->pos);
    rr(e, 0x0fbd, CX, R8, false);  /* bsr ecx, r8d: 31 - leading */
    rr(e, 0x89, CX, AX, false);
    alu_imm(e, 0, AX, 1);          /* scale = 32 - leading */
    core_mem(e, 0x89, AX, CORE_FIELD(scale), false);
    alu_imm(e, 6, CX, 31);         /* ecx = leading */
    rr(e, 0xd3, 4, R8, false);     /* shl r8d, cl */
    rr(e, 0x89, R8, AX, false);
    shift(e, 5, AX, 19, false);
    core_mem(e, 0x89, AX, CORE_FIELD(seed_index), false);
    imm(e, DX, 0x40000000);
    rr(e, 0x31, AX, AX, false);
    rr(e, 0xf7, 6, R8, false);     /* div r8d -> eax = (1ULL << 62) / x_norm */
    core_mem(e, 0x89, AX, CORE_FIELD(seed), false);
    rel32(done, e->pos);
    if (save_dx)
        emit_byte(e, 0x58 | DX);
}

/* EAX = clamp(EAX >> 16, 0, 255). Clobbers ECX. */
static void
emit_sat(emitter *e)
{
    shift(e, 7, AX, 16, false);
    imm(e, CX, 255);
    rr(e, 0x39, CX, AX, false);
    rr(e, 0x0f4f, AX, CX, false);
    rr(e, 0x31, CX, CX, false);
    rr(e, 0x85, AX, AX, false);
    rr(e, 0x0f4c, AX, CX, false);
}

static void
emit_fg_merge(emitter *e, uint32_t mask)
{
    alu_imm(e, 4, AX, mask);
    core_mem(e, 0x8b, CX, PE(fg), false);
    alu_imm(e, 4, CX, ~mask);
    rr(e, 0x09, CX, AX, false);
    core_mem(e, 0x89, AX, PE(fg), false);
}

static void
emit_uv_store(emitter *e, bool u)
{
    core_mem(e, 0x89, AX, u ? PE(u) : PE(v), false);
    core_mem(e, 0x8b, CX, PE(src_mask), false);
    if (!u)
        shift(e, 5, CX, 16, false);
    alu_imm(e, 4, CX, 0x7ff);
    core_mem(e, 0x89, CX, u ? PE(u_mask) : PE(v_mask), false);
}

/* Mirrors step_param() and pixel_set() with value in EAX and delta in ECX.
   Preserves RDX when save_dx is true. */
static void
emit_param(emitter *e, pixel_param param, bool save_dx)
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
            rr(e, 0x85, CX, CX, false);
            core_mem(e, 0x0f98, 0, PE(x_dir), false);
            core_mem(e, 0x89, AX, PE(x_shadow), false);
            break;
        case P_X:
            core_mem(e, 0x89, AX, PE(x), false);
            break;
        case P_Y:
            core_mem(e, 0x89, AX, PE(y), false);
            break;
        case P_F:
            core_mem(e, 0x89, AX, PE(f), false);
            break;
        case P_CNT:
            shift(e, 5, AX, 16, false);
            core_mem(e, 0x89, AX, PE(cnt), false);
            break;
        case P_Z:
            core_mem(e, 0x89, AX, PE(z), false);
            emit_byte(e, 0xa9);
            dword(e, 0xf8000000);
            rr(e, 0x0f95, 0, CX, false);
            core_mem(e, 0xf7, 0, PE(src_mode), false);
            dword(e, SRC_Z_SCISSOR);
            rr(e, 0x0f95, 0, AX, false);
            rr(e, 0x20, AX, CX, false);
            core_mem(e, 0x88, CX, PE(z_ovf), false);
            break;
        case P_U:
        case P_V:
            {
                rr(e, 0x85, CX, CX, false);
                core_mem(e, 0x0f98, 0, param == P_U ? PE(u_dir) : PE(v_dir), false);
                core_mem(e, 0xf7, 0, PE(src_mode), false);
                dword(e, SRC_SWAP_UV);
                opcode(e, 0x0f85);
                uint8_t *swapped = e->pos;
                dword(e, 0);
                emit_uv_store(e, param == P_U);
                emit_byte(e, 0xe9);
                uint8_t *done = e->pos;
                dword(e, 0);
                rel32(swapped, e->pos);
                emit_uv_store(e, param != P_U);
                rel32(done, e->pos);
                break;
            }
        case P_Q:
            core_mem(e, 0x89, AX, PE(q), false);
            rr(e, 0x89, AX, R8, false);
            emit_seed_op(e, save_dx);
            break;
        case P_R:
        case P_G:
        case P_B:
        case P_A:
            core_mem(e, 0x89, AX, param == P_R ? PE(r) : param == P_G ? PE(g)
                                                    : param == P_B   ? PE(b)
                                                                     : PE(a),
                     false);
            emit_sat(e);
            shift(e, 4, AX, channel[param][0], false);
            emit_fg_merge(e, channel[param][1]);
            break;
        case P_I:
            core_mem(e, 0x89, AX, PE(r), false);
            core_mem(e, 0x89, AX, PE(g), false);
            core_mem(e, 0x89, AX, PE(b), false);
            emit_sat(e);
            rr(e, 0x69, AX, AX, false);
            dword(e, 0x010101);
            emit_fg_merge(e, 0xffffff);
            break;
        default:
            break;
    }
}

static bool
step_operands(unsigned d, unsigned s2, unsigned s1)
{
    return writable_reg(d) && readable_reg(s2) && readable_reg(s1);
}

/* Parameter steps (0xd0-0xdf). Reads each operand ONCE! */
static bool
emit_step(emitter *e, uint32_t ir, bool save_dx, uint8_t **fifo_exit)
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
        unsigned pops = is_fifo_field(s2) + is_fifo_field(s1);
        if (pops)
            *fifo_exit = emit_fifo_guard(e, pops);
        pixel_param param = scalar[op & 7][(s2 | s1) & 1];
        read_reg(e, AX, s2);
        read_reg(e, CX, s1);
        if (writable_reg(d)) {
            rr(e, 0x89, AX, R8, false);
            rr(e, 0x01, CX, R8, false);
            emit_byte(e, 0x66);
            rr(e, 0x0f6e, 0, R8, false); /* movd xmm0, r8d */
        }
        emit_param(e, param, save_dx);
        if (writable_reg(d)) {
            emit_byte(e, 0x66);
            rr(e, 0x0f7e, 0, AX, false); /* movd eax, xmm0 */
            write_reg(e, d, AX);
        }
        return true;
    }
    if (!plain_reg(d & ~1U) || !plain_reg(s2 & ~1U) || !plain_reg(s1 & ~1U) || !writable_reg(d & ~1U))
        return false;
    bool need_scalar_write = d >= 64 && !(s2 >= 64 && s1 >= 64);
    if (d >= 64 && s2 >= 64 && s1 >= 64) {
        packed_reg(e, 0, s2, false);
        packed_reg(e, 1, s1, false);
        emit_byte(e, 0x66);
        rr(e, 0x0ffe, 0, 1, false); /* paddd xmm0, xmm1 */
    }
    for (unsigned lane = 0; lane < 2; lane++) {
        pixel_param param = pair[op & 7][lane];
        read_reg(e, AX, lane ? s2 | 1 : s2 & ~1U);
        read_reg(e, CX, lane ? s1 | 1 : s1 & ~1U);
        if (need_scalar_write) {
            rr(e, 0x89, AX, R8, false);
            rr(e, 0x01, CX, R8, false);
            emit_byte(e, 0x66);
            rr(e, 0x0f6e, lane, R8, false); /* movd xmm<lane>, r8d */
            if (lane) {
                emit_byte(e, 0x66);
                rr(e, 0x0f62, 0, 1, false); /* punpckldq xmm0, xmm1 */
            }
        }
        emit_param(e, param, save_dx);
    }
    if (d >= 64)
        packed_reg(e, 0, d, true);
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
    return plain_reg(field) && plain_reg(field | 15);
}

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

/* 3 x 64-bit XOR-OR key check + single branch. */
static uint8_t *
emit_site(emitter *e, verite_draw_site *site, bool check, uint32_t code_start, uint32_t code_last)
{
    uint64_t address = (uint64_t) (uintptr_t) site;
    emit_byte(e, 0x49); /* mov r11, site */
    emit_byte(e, 0xbb);
    memcpy(e->pos, &address, 8);
    e->pos += 8;
    for (unsigned i = 0; i < 3; i++) {
        core_mem64(e, 0x8b, AX, PE(drawctl) + 8 * i);
        rex(e, true, AX, 0, INDEX);
        opcode(e, 0x33);
        emit_byte(e, 0x40 | ((AX & 7) << 3) | (INDEX & 7));
        emit_byte(e, (uint8_t) (8 * i));
        if (i == 0)
            rr(e, 0x89, AX, CX, true);
        else
            rr(e, 0x09, AX, CX, true);
    }
    opcode(e, 0x0f85); /* jnz miss */
    uint8_t *miss = e->pos;
    dword(e, 0);

    emit_byte(e, 0x49); /* mov rax, [r11 + fn] */
    emit_byte(e, 0x8b);
    emit_byte(e, 0x43);
    emit_byte(e, offsetof(verite_draw_site, fn));
    emit_byte(e, 0xeb);
    uint8_t *hit = e->pos++;
    rel32(miss, e->pos);
    emit_byte(e, 0x48); /* mov rax, pixel_jit_site_miss */
    emit_byte(e, 0xb8);
    uint64_t handler = (uint64_t) (uintptr_t) pixel_jit_site_miss;
    memcpy(e->pos, &handler, 8);
    e->pos += 8;
    *hit = (uint8_t) (e->pos - (hit + 1));

    emit_byte(e, 0x41);
    emit_byte(e, 0x50 | (CORE & 7));
    emit_byte(e, 0x41);
    emit_byte(e, 0x50 | (MEM & 7));
    emit_byte(e, 0x50 | DX);
    rr(e, 0x83, 5, SP, true);
    emit_byte(e, 32);
    rr(e, 0x89, CORE, ARG_CORE, true);
    rr(e, 0x89, INDEX, ARG_MEMORY, true);
    rr(e, 0xff, 2, AX, false);
    rr(e, 0x83, 0, SP, true);
    emit_byte(e, 32);
    emit_byte(e, 0x58 | DX);
    emit_byte(e, 0x41);
    emit_byte(e, 0x58 | (MEM & 7));
    emit_byte(e, 0x41);
    emit_byte(e, 0x58 | (CORE & 7));
    if (!check)
        return NULL;

    emit_byte(e, 0x49);
    emit_byte(e, 0xbb);
    memcpy(e->pos, &address, 8);
    e->pos += 8;
    emit_byte(e, 0x41);
    emit_byte(e, 0x8b);
    emit_byte(e, 0x43);
    emit_byte(e, offsetof(verite_draw_site, lo));
    alu_imm(e, 7, AX, code_last);
    opcode(e, 0x0f87);
    uint8_t *after = e->pos;
    dword(e, 0);
    emit_byte(e, 0x41);
    emit_byte(e, 0x8b);
    emit_byte(e, 0x43);
    emit_byte(e, offsetof(verite_draw_site, hi));
    alu_imm(e, 7, AX, code_start);
    opcode(e, 0x0f82);
    uint8_t *below = e->pos;
    dword(e, 0);
    emit_byte(e, 0xe9);
    uint8_t *exit = e->pos;
    dword(e, 0);
    rel32(after, e->pos);
    rel32(below, e->pos);
    return exit;
}

static bool
emit_draw(emitter *e, uint32_t ir)
{
    unsigned op       = ir >> 24;
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

static int32_t
spr_direct_offset(unsigned idx)
{
    switch (idx) {
        case 0: return (int32_t) PE(src_base);
        case 1: return (int32_t) PE(src_mode);
        case 2: return (int32_t) PE(src_mask);
        case 3: return (int32_t) PE(stride);
        case 4: return (int32_t) PE(dst_base);
        case 5: return (int32_t) PE(dst_mode);
        case 6: return (int32_t) PE(dst_fmt);
        case 7: return (int32_t) PE(pmask);
        case 12: return (int32_t) PE(pat_mode);
        case 14: return (int32_t) PE(scissor_x);
        case 15: return (int32_t) PE(scissor_y);
        case 16: return (int32_t) PE(z_base);
        case 17: return (int32_t) PE(dither_offset);
        case 20: return (int32_t) PE(bg);
        case 21: return (int32_t) PE(fog_color);
        case 22: return (int32_t) PE(dst_color);
        case 23: return (int32_t) PE(chroma_color);
        case 24: return (int32_t) PE(yuv_contrast);
        case 26: return (int32_t) PE(chroma_mask);
        case 29: return (int32_t) PE(alpha_thres);
        case 30: return (int32_t) PE(spec_color);
        default:
            if (idx >= 32 && idx <= 39)
                return (int32_t) (PE(tex4pal) + 4U * (idx - 32));
            return -1;
    }
}

static bool
emit_integer(emitter *e, uint32_t ir, uint32_t pc, bool save_dx, uint8_t **fifo_exit)
{
    unsigned op  = ir >> 24;
    unsigned d   = (uint8_t) (ir >> 16);
    unsigned s2  = (uint8_t) (ir >> 8);
    unsigned s1  = (uint8_t) ir;
    unsigned sub = op & 15;

    if (op == 0x6b && writable_reg(d)) {
        imm(e, AX, risc_relative(pc, 2));
        write_reg(e, d, AX);
        return true;
    }
    if (op == 0x76 || op == 0x77) {
        if (!writable_reg(d))
            return false;
        imm(e, AX, op == 0x76 ? ir & 0xffff : ir << 16);
        write_reg(e, d, AX);
        return true;
    }
    if (op == 0xc4) {
        if (!plain_reg(d) || !plain_reg(s2) || !plain_reg(s1))
            return false;
        emit_call(e, (const void *) risc_sort_segments, false, ir);
        return true;
    }
    if (op >= 0xc0 && op <= 0xc3) {
        if (!plain_reg(d & ~1U) || !writable_reg(d & ~1U) || !readable_reg(s1))
            return false;
        if (is_fifo_field(s1))
            *fifo_exit = emit_fifo_guard(e, 1);
        read_reg(e, AX, s1);
        rr(e, 0x89, AX, CX, false);
        switch (op) {
            case 0xc0:
                alu_imm(e, 4, AX, 0xffff0000U);
                shift(e, 4, CX, 16, false);
                break;
            case 0xc1:
                shift(e, 4, AX, 16, false);
                alu_imm(e, 4, CX, 0xffff0000U);
                break;
            case 0xc2:
                alu_imm(e, 4, AX, 0x00ff0000U);
                shift(e, 5, CX, 8, false);
                alu_imm(e, 4, CX, 0x00ff0000U);
                break;
            case 0xc3:
                shift(e, 4, AX, 8, false);
                alu_imm(e, 4, AX, 0x00ff0000U);
                shift(e, 4, CX, 16, false);
                alu_imm(e, 4, CX, 0x00ff0000U);
                break;
        }
        write_reg(e, d & ~1U, AX);
        write_reg(e, d | 1, CX);
        return true;
    }
    if (op == 0x4e || op == 0x4f || op == 0x5f) {
        if (op == 0x4f && !readable_reg(s2))
            return false;
        if (op == 0x5f && (!readable_reg(s2) || !readable_reg(s1)))
            return false;
        unsigned pops = (op == 0x4f && is_fifo_field(s2)) ? 1 : (op == 0x5f ? is_fifo_field(s2) + is_fifo_field(s1) : 0);
        if (pops)
            *fifo_exit = emit_fifo_guard(e, pops);
        int32_t off = op != 0x5f ? spr_direct_offset(s1) : -1;
        if (off >= 0) {
            if (op == 0x4e)
                core_imm(e, (uint32_t) off, (ir >> 8) & 0xffff);
            else {
                read_reg(e, AX, s2);
                core_mem(e, 0x89, AX, (uint32_t) off, false);
            }
            return true;
        }
        emit_call(e, (const void *) risc_pixel_spr_insn, false, ir);
        return true;
    }
    if (op <= 0x3f) {
        bool pair      = op >= 0x20;
        bool immediate = op < 0x10;
        if (pair && sub == 15)
            return plain_reg(d) && plain_reg(s2) && plain_reg(s1);
        if (sub == 15) {
            /* 0x0f / 0x1f: f2i */
            if (!readable_reg(s2) || (!immediate && !readable_reg(s1)) || !writable_reg(d))
                return false;
            unsigned pops = is_fifo_field(s2) + (!immediate && is_fifo_field(s1));
            if (pops)
                *fifo_exit = emit_fifo_guard(e, pops);
            read_reg(e, AX, s2);
            if (immediate)
                imm(e, CX, s1);
            else
                read_reg(e, CX, s1);
            emit_call2_ret(e, (const void *) risc_f2i);
            write_reg(e, d, AX);
            return true;
        }
        if (pair) {
            if (!plain_reg(s2) || !plain_reg(s1) || (sub != 14 && (!plain_reg(d) || !writable_reg(d))))
                return false;
        } else {
            if (!readable_reg(s2) || (!immediate && !readable_reg(s1)) || (sub != 14 && !writable_reg(d)))
                return false;
            unsigned pops = is_fifo_field(s2) + (!immediate && is_fifo_field(s1));
            if (pops)
                *fifo_exit = emit_fifo_guard(e, pops);
            if (immediate && d == s2 && d >= 64 && (sub == 0 || sub == 1)) {
                if (s1) {
                    uint32_t disp = reg_address(e, d);
                    core_mem(e, s1 < 128 ? 0x83 : 0x81, sub == 0 ? 0 : 5, disp, segmented(d));
                    if (s1 < 128)
                        emit_byte(e, s1);
                    else
                        dword(e, s1);
                }
                return true;
            }
        }
        if (pair && sub < 8 && d >= 64 && s2 >= 64 && (op >= 0x30 || s1 >= 64)) {
            emit_packed_alu(e, sub, d, s2, s1, op < 0x30);
            return true;
        }
        if (pair) {
            emit_alu(e, sub, s2 & ~1U, op < 0x30 ? s1 & ~1U : s1, false);
            if (sub != 14 && d >= 64)
                rr(e, 0x89, AX, R8, false);
            emit_alu(e, sub, s2 | 1, op < 0x30 ? s1 | 1 : s1, false);
            if (sub != 14) {
                write_reg(e, d & ~1U, R8);
                write_reg(e, d | 1, AX);
            }
        } else {
            emit_alu(e, sub, s2, s1, immediate);
            if (sub != 14)
                write_reg(e, d, AX);
        }
        return true;
    }

    if (!((op >= 0x40 && op <= 0x4d) || (op >= 0x52 && op <= 0x5e && op != 0x5b)))
        return false;
    if (!writable_reg(d) || (op != 0x52 && !readable_reg(s2)) || (op >= 0x50 && !readable_reg(s1)))
        return false;
    unsigned pops = (op != 0x52 && is_fifo_field(s2)) + (op >= 0x50 && is_fifo_field(s1));
    if (pops)
        *fifo_exit = emit_fifo_guard(e, pops);

    if (op == 0x42 || op == 0x53) {
        read_reg(e, R8, s2);
        emit_seed_op(e, save_dx);
        if (op == 0x42)
            imm(e, CX, s1);
        else
            read_reg(e, CX, s1);
        /* AX = seed; if (scale >= b) seed >> (scale - b) else seed << (b - scale). */
        core_mem(e, 0x8b, R8, CORE_FIELD(scale), false);
        core_mem(e, 0x8b, AX, CORE_FIELD(seed), false);
        rr(e, 0x39, CX, R8, false); /* cmp r8d, ecx */
        opcode(e, 0x0f82);
        uint8_t *shl_path = e->pos;
        dword(e, 0);
        rr(e, 0x29, CX, R8, false);
        rr(e, 0x89, R8, CX, false);
        alu_imm(e, 7, CX, 32);
        opcode(e, 0x0f82);
        uint8_t *shr_ok = e->pos;
        dword(e, 0);
        rr(e, 0x31, AX, AX, false);
        emit_byte(e, 0xe9);
        uint8_t *done0 = e->pos;
        dword(e, 0);
        rel32(shr_ok, e->pos);
        rr(e, 0xd3, 5, AX, false); /* shr eax, cl */
        emit_byte(e, 0xe9);
        uint8_t *done1 = e->pos;
        dword(e, 0);

        rel32(shl_path, e->pos);
        rr(e, 0x29, R8, CX, false); /* sub ecx, r8d */
        alu_imm(e, 7, CX, 32);
        opcode(e, 0x0f82);
        uint8_t *shl_ok = e->pos;
        dword(e, 0);
        rr(e, 0x31, AX, AX, false);
        emit_byte(e, 0xe9);
        uint8_t *done2 = e->pos;
        dword(e, 0);
        rel32(shl_ok, e->pos);
        rr(e, 0xd3, 4, AX, false); /* shl eax, cl */
        rel32(done0, e->pos);
        rel32(done1, e->pos);
        rel32(done2, e->pos);
        write_reg(e, d, AX);
        return true;
    }

    read_reg(e, AX, op == 0x52 ? s1 : s2);
    switch (op) {
        case 0x40:
        case 0x41:
        case 0x4b:
            alu_imm(e, op == 0x41 ? 5 : 0, AX, s1 << (op == 0x4b ? 8 : 16));
            break;
        case 0x43:
            imm(e, CX, s1 << 16);
            rr(e, 0x29, AX, CX, false);
            rr(e, 0x89, CX, AX, false);
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
                static const uint8_t groups[] = { 1, 4, 7, 5 };
                unsigned             group_id = groups[op & 3];
                if (op < 0x50)
                    shift(e, group_id, AX, s1 & 31, false);
                else {
                    read_reg(e, CX, s1);
                    rr(e, 0xd3, group_id, AX, false);
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
                static const uint8_t cc[] = { 0x9c, 0x92, 0x94 };
                if (op < 0x50)
                    alu_imm(e, 7, AX, s1);
                else {
                    read_reg(e, CX, s1);
                    rr(e, 0x39, CX, AX, false);
                }
                rr(e, 0x0f00 | cc[(op & 15) - 8], 0, AX, false);
                rr(e, 0x0fb6, AX, AX, false);
                break;
            }
        case 0x4c:
        case 0x4d:
        case 0x5c:
        case 0x5d:
            if (op < 0x50)
                imm(e, CX, s1);
            else
                read_reg(e, CX, s1);
            rr(e, 0x39, CX, AX, false);
            rr(e, (op & 1) ? 0x0f4c : 0x0f4f, AX, CX, false);
            break;
        case 0x52:
            rr(e, 0x89, AX, CX, false);
            rr(e, 0xf7, 3, AX, false);
            rr(e, 0x0f4c, AX, CX, false);
            break;
        case 0x5e:
            read_reg(e, CX, s1);
            rr(e, 0x01, CX, AX, false);
            alu_imm(e, 4, AX, 0xffff0000);
            break;
    }
    write_reg(e, d, AX);
    return true;
}

static bool
emit_branch(emitter *e, uint32_t ir, uint32_t pc, uint8_t **fifo_exit)
{
    unsigned op   = ir >> 24;
    unsigned d    = (uint8_t) (ir >> 16);
    unsigned s1   = (uint8_t) ir;
    uint32_t slot = risc_next(pc);
    uint32_t link = risc_next(slot);

    if (op >= 0x60 && op <= 0x65 && readable_reg(s1)) {
        static const uint8_t cc[] = { 0x44, 0x45, 0x49, 0x48, 0x4f, 0x4e };
        if (is_fifo_field(s1))
            *fifo_exit = emit_fifo_guard(e, 1);
        read_reg(e, AX, s1);
        imm(e, DX, link);
        imm(e, CX, risc_relative(slot, (int16_t) (ir >> 8)));
        rr(e, 0x85, AX, AX, false);
        rr(e, 0x0f00 | cc[op - 0x60], DX, CX, false);
    } else if (op == 0x6f && readable_reg(s1) && writable_reg(d)) {
        if (is_fifo_field(s1))
            *fifo_exit = emit_fifo_guard(e, 1);
        read_reg(e, DX, s1);
        imm(e, AX, link);
        write_reg(e, d, AX);
    } else if (op == 0x68 || op == 0x6a || op == 0x6c || op == 0x6e) {
        if (op == 0x6a || op == 0x6e)
            core_imm(e, GPR(RISC_RA), link);
        imm(e, DX, op < 0x6c ? risc_relative(slot, (int32_t) (ir << 8) >> 8) : risc_absolute(ir));
    } else {
        return false;
    }
    return true;
}

/* Memory load/store. Returns slow-path exit stub on non-local address (or
   FIFO empty). Sets *selfmod_exit when a local store may overwrite
   [code_start, code_last]. */
static uint8_t *
emit_memory(emitter *e, uint32_t ir, bool save_dx, uint8_t mem_mode, uint32_t code_row,
            bool prev_mem, bool check_selfmod, uint32_t code_start, uint32_t code_last,
            uint8_t **selfmod_exit, uint8_t **fifo_exit)
{
    unsigned op    = ir >> 24;
    unsigned d     = (uint8_t) (ir >> 16);
    unsigned s2    = (uint8_t) (ir >> 8);
    unsigned s1    = (uint8_t) ir;
    bool     write = (op >= 0x78 && op <= 0x7a) || op == 0x7c;
    bool     load  = (op >= 0x70 && op <= 0x72) || op == 0x74 || op == 0x75;
    bool     pair  = op == 0x74 || op == 0x75 || op == 0x7c;
    if ((!write && !load) || !mem_mode)
        return NULL;
    if (load) {
        if (!readable_reg(s1))
            return NULL;
        if (pair ? (d < 64) : !writable_reg(d))
            return NULL;
    } else {
        if (!readable_nonfifo(s1))
            return NULL;
        if (pair ? (s2 < 64) : !readable_nonfifo(s2))
            return NULL;
    }

    unsigned size   = pair ? 8 : 1U << (op & 3);
    unsigned offset = size * (write ? d : s2);
    if (load && is_fifo_field(s1))
        *fifo_exit = emit_fifo_guard(e, 1);

    read_reg(e, AX, s1);
    if (offset)
        alu_imm(e, 0, AX, offset);
    alu_imm(e, 7, AX, LOCAL_END);
    opcode(e, 0x0f83); /* jae slow_path */
    uint8_t *exit = e->pos;
    dword(e, 0);

    if (save_dx)
        emit_byte(e, 0x50 | DX);
    /* eax = masked address; r8 = jit_vram */
    core_mem(e, 0x23, AX, CORE_FIELD(jit_mask), false);
    core_mem64(e, 0x8b, R8, CORE_FIELD(jit_vram));

    if (mem_mode == 2) {
        if (prev_mem) {
            core_mem(e, 0x8b, CX, CORE_FIELD(dram_row), false);
            alu_imm(e, 7, CX, code_row);
            opcode(e, 0x0f84);
            uint8_t *row_ok = e->pos;
            dword(e, 0);
            core_imm(e, CORE_FIELD(dram_row), code_row);
            core_imm(e, CORE_FIELD(dram_wait), 1);
            rr(e, 0xff, 0, MEM, false);
            emit_byte(e, 0xeb);
            uint8_t *fetch_done = e->pos++;
            rel32(row_ok, e->pos);
            core_imm(e, CORE_FIELD(dram_wait), 0);
            *fetch_done = (uint8_t) (e->pos - (fetch_done + 1));
        }
        rr(e, 0x89, AX, CX, false);
        shift(e, 5, CX, 12, false);
        core_mem(e, 0x3b, CX, CORE_FIELD(dram_row), false);
        opcode(e, 0x0f84);
        uint8_t *same_row = e->pos;
        dword(e, 0);
        core_mem(e, 0x89, CX, CORE_FIELD(dram_row), false);
        rr(e, 0xff, 0, MEM, false);
        core_mem(e, 0xff, 0, CORE_FIELD(dram_wait), false);
        rel32(same_row, e->pos);
    }

    if (write) {
        /* Bump page_gen[masked_addr >> 12 & 1023] and save masked_addr in DX. */
        rr(e, 0x89, AX, CX, false);
        shift(e, 5, CX, 12, false);
        alu_imm(e, 4, CX, 1023);
        rex(e, false, 0, CX, CORE);
        opcode(e, 0xff);
        emit_byte(e, 0x84);
        emit_byte(e, (2 << 6) | ((CX & 7) << 3) | (CORE & 7));
        dword(e, CORE_FIELD(page_gen));
        if (check_selfmod)
            rr(e, 0x89, AX, INDEX, false);
    }

    if (size != 1)
        alu_imm(e, 4, AX, ~(size - 1));
    if (size < 4)
        alu_imm(e, 6, AX, 4 - size);

    if (pair) {
        if (write)
            packed_reg(e, 0, s2, false);
        emit_byte(e, write ? 0x66 : 0xf3);
    } else if (write) {
        read_reg(e, CX, s2);
        if (size == 2)
            emit_byte(e, 0x66);
    }
    unsigned data = pair ? 0 : CX;
    rex(e, false, data, AX, R8);
    if (pair) {
        opcode(e, write ? 0x0fd6 : 0x0f7e);
    } else if (load && size != 4) {
        emit_byte(e, 0x0f);
        emit_byte(e, size == 1 ? 0xb6 : 0xb7);
    } else {
        emit_byte(e, write ? (size == 1 ? 0x88 : 0x89) : 0x8b);
    }
    emit_byte(e, ((data & 7) << 3) | 4);
    emit_byte(e, ((AX & 7) << 3) | (R8 & 7));

    if (write) {
        core_mem(e, 0xc6, 0, CORE_FIELD(drew), false);
        emit_byte(e, 1);
        if (check_selfmod) {
            uint32_t lo_guard = (size == 8) ? (code_start & ~7U) : code_start;
            uint32_t guard_len = code_last - lo_guard + 1;
            if (size != 8)
                alu_imm(e, 4, INDEX, ~(size - 1));
            else
                alu_imm(e, 4, INDEX, ~7U);
            if (lo_guard)
                alu_imm(e, 5, INDEX, lo_guard);
            alu_imm(e, 7, INDEX, guard_len);
            opcode(e, 0x0f83); /* jae ok */
            uint8_t *ok = e->pos;
            dword(e, 0);
            if (save_dx)
                emit_byte(e, 0x58 | DX);
            emit_byte(e, 0xe9);
            *selfmod_exit = e->pos;
            dword(e, 0);
            rel32(ok, e->pos);
        }
    } else if (pair) {
        packed_reg(e, 0, d, true);
    } else {
        write_reg(e, d, CX);
    }
    if (save_dx)
        emit_byte(e, 0x58 | DX);
    return exit;
}

static void
emit_finish(emitter *e, uint32_t pc, unsigned count, bool branch, bool delay_in_block, bool loop, bool slow_mem, bool last_mem)
{
    if (!count) {
        rr(e, 0x31, AX, AX, false);
        emit_byte(e, 0xc3);
        return;
    }
    if (branch && delay_in_block) {
        core_mem(e, 0x89, DX, CORE_FIELD(pc), false);
        rr(e, 0x89, DX, AX, false);
        alu_imm(e, 0, DX, 4);
        alu_imm(e, 4, AX, 0xff000000);
        alu_imm(e, 4, DX, 0x00ffffff);
        rr(e, 0x09, AX, DX, false);
    } else {
        core_imm(e, CORE_FIELD(pc), pc);
    }
    if (branch)
        core_mem(e, 0x89, DX, CORE_FIELD(npc), false);
    else
        core_imm(e, CORE_FIELD(npc), risc_next(pc));

    if (loop) {
        core_mem(e, 0x8b, AX, CORE_FIELD(jit_budget), false);
        rr(e, 0x29, MEM, AX, false);
        core_mem(e, 0x01, AX, CORE_FIELD(timer), false);
    } else if (slow_mem) {
        if (!last_mem)
            core_imm(e, CORE_FIELD(dram_wait), 0);
        imm(e, AX, count);
        rr(e, 0x01, MEM, AX, false); /* eax = count + extra_wait */
        core_mem(e, 0x01, AX, CORE_FIELD(timer), false);
    } else {
        core_mem(e, 0x83, 0, CORE_FIELD(timer), false);
        emit_byte(e, count);
        imm(e, AX, count);
    }
    emit_byte(e, 0xc3);
}

static void
emit_side_finish(emitter *e, uint32_t pc, unsigned count, bool loop, bool branch_delay, bool slow_mem, bool last_mem)
{
    if (!loop) {
        emit_finish(e, pc, count, branch_delay, false, false, slow_mem, last_mem);
        return;
    }
    core_imm(e, CORE_FIELD(pc), pc);
    if (branch_delay)
        core_mem(e, 0x89, DX, CORE_FIELD(npc), false);
    else
        core_imm(e, CORE_FIELD(npc), risc_next(pc));
    core_mem(e, 0x8b, AX, CORE_FIELD(jit_budget), false);
    rr(e, 0x29, MEM, AX, false);
    if (count)
        alu_imm(e, 0, AX, count);
    core_mem(e, 0x01, AX, CORE_FIELD(timer), false);
    emit_byte(e, 0xc3);
}

static uint8_t *
alloc_code(void)
{
#    ifdef _WIN32
    return VirtualAlloc(NULL, ARENA_SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
#    else
    int flags = MAP_PRIVATE | MAP_ANON;
#        if defined(__APPLE__) && defined(MAP_JIT)
    flags |= MAP_JIT;
#        endif
    void *code = mmap(NULL, ARENA_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC, flags, -1, 0);
    return code == MAP_FAILED ? NULL : code;
#    endif
}

void
risc_jit_free(verite_core *c)
{
    pixel_jit_free(c);
    if (c->jit) {
#    ifdef _WIN32
        VirtualFree(c->jit->arena, 0, MEM_RELEASE);
#    else
        munmap(c->jit->arena, ARENA_SIZE);
#    endif
        free(c->jit);
        c->jit = NULL;
    }
}

static struct verite_jit *
jit_create(void)
{
    struct verite_jit *jit = calloc(1, sizeof(*jit));
    if (jit && !(jit->arena = alloc_code())) {
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

#    if defined(_MSC_VER)
__declspec(noinline)
#    else
__attribute__((noinline))
#    endif
static jit_block *
compile_block(verite_core *c, verite_memory mem, unsigned limit, uint8_t mem_mode)
{
    struct verite_jit *jit = c->jit;
    uint8_t            code[MAX_BLOCK_CODE];
    emitter            e = { code };
    uint32_t           words[BLOCK_WORDS];
    uint8_t           *exits[BLOCK_WORDS]       = { 0 };
    uint8_t           *fifo_exits[BLOCK_WORDS]  = { 0 };
    uint8_t           *after[BLOCK_WORDS]       = { 0 };
    bool               exit_branch[BLOCK_WORDS] = { 0 };
    bool               mem_at[BLOCK_WORDS]      = { 0 };
    uint32_t           start                    = c->pc;
    bool               local                    = start < LOCAL_END;
    uint32_t           code_start               = (uint32_t) (start & mem.mask);
    uint32_t           code_last                = code_start + limit * 4 - 1;
    uint32_t           code_row                 = code_start >> 12;
    uint32_t           pc                       = start;
    unsigned           count                    = 0;
    bool               branch = false, delay_in_block = false;
    bool               uses_memory = false, prev_mem = false;
    unsigned           exit_count = 0, mem_ops = 0;
    bool               backedge = false, conditional = false;

    emit_byte(&e, 0xf3);
    emit_byte(&e, 0x0f);
    emit_byte(&e, 0x1e);
    emit_byte(&e, 0xfa);
    rr(&e, 0x89, ARG_CORE, CORE, true);
    uint8_t *prologue = e.pos;
    e.pos += 16;
    uint8_t *body = e.pos;
    while (count < limit) {
        if ((size_t) (e.pos - code) + MAX_INSN_CODE + (exit_count + 2) * FINISH_CODE > MAX_BLOCK_CODE)
            break;
        uint32_t ir  = risc_fetch(c, pc, mem);
        words[count] = ir;
        uint8_t *before_row_check = e.pos;
        if (mem_mode == 2 && prev_mem) {
            /* Ensure instruction fetch returns dram_row to code_row before a non-memory insn. */
            uint8_t op = ir >> 24;
            bool    is_mem = (op >= 0x70 && op <= 0x72) || op == 0x74 || op == 0x75 || (op >= 0x78 && op <= 0x7a) || op == 0x7c;
            if (!is_mem) {
                core_mem(&e, 0x8b, CX, CORE_FIELD(dram_row), false);
                alu_imm(&e, 7, CX, code_row);
                opcode(&e, 0x0f84);
                uint8_t *row_ok = e.pos;
                dword(&e, 0);
                core_imm(&e, CORE_FIELD(dram_row), code_row);
                rr(&e, 0xff, 0, MEM, false);
                rel32(row_ok, e.pos);
                prev_mem = false;
            }
        }
        uint8_t *f_exit = NULL;
        if (emit_integer(&e, ir, pc, branch, &f_exit) || emit_step(&e, ir, branch, &f_exit)) {
            if (f_exit) {
                fifo_exits[count]  = f_exit;
                exit_branch[count] = branch;
                exit_count++;
            }
            pc = risc_next(pc);
            count++;
            if (branch) {
                delay_in_block = true;
                break;
            }
        } else if (draw_site(c, ir)) {
            verite_draw_site *site = &jit->sites[jit->sites_used - 1];
            after[count]           = emit_site(&e, site, local && !branch, code_start, code_last);
            exit_count += after[count] != NULL;
            pc = risc_next(pc);
            count++;
            if (branch) {
                delay_in_block = true;
                break;
            }
        } else if (emit_draw(&e, ir)) {
            pc = risc_next(pc);
            count++;
            delay_in_block = branch;
            break;
        } else if (!branch && emit_branch(&e, ir, pc, &f_exit)) {
            if (f_exit) {
                fifo_exits[count] = f_exit;
                exit_count++;
            }
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
            uint8_t *selfmod = NULL;
            uint8_t *m_exit  = emit_memory(&e, ir, branch, mem_mode, code_row, prev_mem,
                                           local && (!branch || backedge), code_start, code_last,
                                           &selfmod, &f_exit);
            if (!m_exit) {
                e.pos = before_row_check;
                break;
            }
            exits[count]       = m_exit;
            fifo_exits[count]  = f_exit;
            after[count]       = selfmod;
            exit_branch[count] = branch;
            mem_at[count]      = true;
            uses_memory        = true;
            prev_mem           = true;
            mem_ops++;
            exit_count += 1 + (selfmod != NULL) + (f_exit != NULL);
            pc = risc_next(pc);
            count++;
            if (branch) {
                delay_in_block = true;
                break;
            }
        }
    }

    jit_block *block  = jit_lookup(jit, start);
    block->pc         = start;
    block->fn         = NULL;
    block->count      = count < 2 ? 1 : count;
    block->mem_mode   = mem_mode;
    block->max_wait   = mem_mode == 2 ? (uint8_t) (mem_ops * 2) : 0;
    block->page_gen   = c->page_gen[(code_start >> 12) & 1023U];
    block->global_gen = c->global_gen;
    memcpy(block->words, words, block->count * sizeof(*words));
    if (count < 2)
        return block;

    bool    loop     = backedge && delay_in_block && !(mem_mode == 2 && uses_memory);
    bool    slow_mem = mem_mode == 2 && uses_memory;
    emitter setup    = { prologue };
    if (loop) {
        rr(&setup, 0x89, ARG_BUDGET, MEM, false);
        core_mem(&setup, 0x89, MEM, CORE_FIELD(jit_budget), false);

        alu_imm(&e, 5, MEM, count);
        alu_imm(&e, 7, MEM, count);
        uint8_t *skip_loop = NULL;
        if (conditional) {
            opcode(&e, 0x0f82);
            skip_loop = e.pos;
            dword(&e, 0);
            rr(&e, 0x81, 7, DX, false);
            dword(&e, start);
        }
        opcode(&e, conditional ? 0x0f84 : 0x0f83);
        dword(&e, (uint32_t) (body - (e.pos + 4)));
        if (skip_loop)
            rel32(skip_loop, e.pos);
    } else if (slow_mem) {
        rr(&setup, 0x31, MEM, MEM, false);
        core_imm(&setup, CORE_FIELD(jit_budget), 0);
    }

    bool last_mem = mem_at[count - 1];
    emit_finish(&e, pc, count, branch, delay_in_block, loop, slow_mem, last_mem);
    for (unsigned i = 0; i < count; i++) {
        if (exits[i]) {
            rel32(exits[i], e.pos);
            emit_side_finish(&e, risc_relative(start, i), i, loop, exit_branch[i], slow_mem, i > 0 && mem_at[i - 1]);
        }
        if (fifo_exits[i]) {
            rel32(fifo_exits[i], e.pos);
            emit_side_finish(&e, risc_relative(start, i), i, loop, exit_branch[i], slow_mem, i > 0 && mem_at[i - 1]);
        }
        if (after[i]) {
            rel32(after[i], e.pos);
            emit_side_finish(&e, risc_relative(start, i + 1), i + 1, loop, exit_branch[i], slow_mem, mem_at[i]);
        }
    }

    size_t   prefix = setup.pos - code;
    size_t   size   = prefix + (e.pos - body);
    uint8_t *dest   = jit->arena + jit->used;
    memcpy(dest, code, prefix);
    memcpy(dest + prefix, body, e.pos - body);
#    ifdef _WIN32
    FlushInstructionCache(GetCurrentProcess(), dest, size);
#    endif
    jit->used += (size + 15) & ~(size_t) 15;
    block->fn = (block_fn) (void *) dest;
    return block;
}

static unsigned
risc_jit_exec_one(verite_core *c, verite_memory mem, unsigned budget)
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

    uint8_t mem_mode = (!canonical_memory || mem.mask < 7) ? 0 : (c->slow_precharge ? 2 : 1);
    bool    match    = block->count && block->pc == pc && block->count <= limit && block->mem_mode == mem_mode;
    if (match && local) {
        uint32_t p = (addr >> 12) & 1023U;
        if (c->page_tracking && mem_mode) {
            if (atomic_load_explicit(&c->page_dirty[p], memory_order_acquire)) {
                atomic_store_explicit(&c->page_dirty[p], 0, memory_order_relaxed);
                c->page_gen[p]++;
            }
            if (block->page_gen != c->page_gen[p] || block->global_gen != c->global_gen) {
                if (memcmp(mem.data + addr, block->words, block->count * sizeof(uint32_t)))
                    match = false;
                else {
                    block->page_gen   = c->page_gen[p];
                    block->global_gen = c->global_gen;
                }
            }
        } else {
            match = !memcmp(mem.data + addr, block->words, block->count * sizeof(uint32_t));
        }
    }
    if (!match) {
        if (jit->used > ARENA_SIZE - MAX_BLOCK_CODE - 15 || jit->sites_used > SITE_COUNT - BLOCK_WORDS) {
            memset(jit->blocks, 0, sizeof(jit->blocks));
            jit->used       = 0;
            jit->sites_used = 0;
        }
        block = compile_block(c, mem, limit, mem_mode);
    }

    unsigned wait   = local && c->slow_precharge && (addr >> 12) != c->dram_row;
    unsigned cycles = block->count + wait + block->max_wait;
    if (!block->fn || cycles > budget)
        return 0;

    uint32_t old_row = c->dram_row;
    if (wait)
        c->dram_row = addr >> 12;
    c->dram_wait = 0;

    jit_memory memory = { mem.data, (uint32_t) mem.mask };
    c->jit_mem        = &memory;
    c->jit_vram       = mem.data;
    c->jit_mask       = (uint32_t) mem.mask;
    unsigned executed = block->fn(c, &memory, budget - wait);
    c->jit_mem        = NULL;
    if (!executed) {
        c->dram_row = old_row;
        return 0;
    }
    if (wait) {
        c->timer++;
        if (executed == 1 && !c->dram_wait)
            c->dram_wait = 1;
        else if (mem_mode == 2 && executed <= 2 && c->dram_wait)
            c->dram_wait += wait;
    }
    if (mem_mode != 2)
        c->dram_wait = executed == 1 ? wait : 0;
    return executed + wait;
}

unsigned
risc_jit_run(verite_core *c, verite_memory mem, unsigned budget)
{
    return risc_jit_exec_one(c, mem, budget);
}

unsigned
risc_jit_run_chain(verite_core *c, verite_memory mem, unsigned budget)
{
    unsigned total = 0;
    while (total + 2 <= budget) {
        unsigned ran = risc_jit_exec_one(c, mem, budget - total);
        if (!ran)
            break;
        total += ran;
        if (c->halted || c->stalled || (c->regs[VIO_DEBUG] & (VDEBUG_SOFT_RESET | VDEBUG_HOLD)))
            break;
        if (c->dma.busy && c->in_count < IN_FIFO)
            break;
        if (c->polls >= 64)
            break;
    }
    return total;
}

#endif
