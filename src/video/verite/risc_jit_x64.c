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
    MAX_BLOCK_CODE = BLOCK_WORDS * 200 + 64,
    /* Room kept free for the largest single instruction (a paired step with
       two U/V lanes and a Q call) and for each epilogue. */
    MAX_INSN_CODE = 640,
    FINISH_CODE   = 64,
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

typedef struct {
    uint8_t *pos;
} emitter;

/* R10 holds the core. R11 indexes segmented operands. EDX is reserved for
   the next PC across a branch's delay slot. All are caller-saved in both ABIs. */
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

static void
read_reg(emitter *e, unsigned host, unsigned field)
{
    if (field < 32) {
        uint32_t value = risc_constant(field);
        if (value)
            imm(e, host, value);
        else
            rr(e, 0x31, host, host, false);
    } else {
        uint32_t disp = reg_address(e, field);
        core_mem(e, 0x8b, host, disp, segmented(field));
    }
}

/* Destinations in the constant bank discard writes. */
static void
write_reg(emitter *e, unsigned field, unsigned host)
{
    if (field >= 64) {
        uint32_t disp = reg_address(e, field);
        core_mem(e, 0x89, host, disp, segmented(field));
    }
}

static bool
plain_reg(unsigned field)
{
    return field < 32 || field >= 64;
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
emit_alu(emitter *e, unsigned op, unsigned s2, unsigned s1, bool immediate)
{
    static const uint8_t binary[]        = { 0x01, 0x29, 0x21, 0x29, 0x21, 0x09, 0x09, 0x31 };
    static const uint8_t group[]         = { 0, 5, 4, 0, 4, 1, 1, 6 };
    static const uint8_t product_shift[] = { 0, 8, 16, 24, 32, 31 };

    read_reg(e, AX, s2);
    if (op < 8) {
        if (immediate && op != 3)
            alu_imm(e, group[op], AX, op == 2 ? ~s1 : s1);
        else {
            if (immediate)
                imm(e, CX, s1);
            else
                read_reg(e, CX, s1);
            if (op == 2)
                rr(e, 0xf7, 2, CX, false); /* not ecx */
            rr(e, binary[op], op == 3 ? AX : CX, op == 3 ? CX : AX, false);
            if (op == 3)
                rr(e, 0x89, CX, AX, false);
        }
        if (op == 6)
            rr(e, 0xf7, 2, AX, false);
        return;
    }

    if (immediate)
        imm(e, CX, s1);
    else
        read_reg(e, CX, s1);
    if (op == 14) {
        emit_clip(e);
        return;
    }
    bool wide = op != 8;
    if (wide) {
        rr(e, 0x63, AX, AX, true); /* movsxd rax, eax */
        rr(e, 0x63, CX, CX, true);
    }
    rr(e, 0x0faf, AX, CX, wide);
    shift(e, 7, AX, product_shift[op - 8], wide);
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
        emit_byte(e, 0x66); /* movd xmm1, eax */
        rr(e, 0x0f6e, 1, AX, false);
        emit_byte(e, 0x66); /* pshufd xmm1, xmm1, 0 */
        rr(e, 0x0f70, 1, 1, false);
        emit_byte(e, 0);
    }
    unsigned result = sub == 2 || sub == 3;
    emit_byte(e, 0x66);
    rr(e, ops[sub], result, result ^ 1, false);
    if (sub == 6) {
        emit_byte(e, 0x66); /* pcmpeqd xmm1, xmm1 */
        rr(e, 0x0f76, 1, 1, false);
        emit_byte(e, 0x66); /* pxor xmm0, xmm1 */
        rr(e, 0x0fef, 0, 1, false);
    }
    packed_reg(e, result, d, true);
}

/* Call fn(core, arg1, memory). EAX supplies arg1 when from_ax is set. The block
   entry leaves RSP at 8 mod 16, so three pushes and 32 bytes of shadow space
   align the call for both ABIs. RDX keeps the next PC in a delay slot. */
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
    rr(e, 0x89, MEM, ARG_BUDGET, true);
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
    emit_byte(e, 0x58 | DX); /* pop rdx */
    emit_byte(e, 0x41);      /* pop r9 */
    emit_byte(e, 0x58 | (MEM & 7));
    emit_byte(e, 0x41); /* pop r10 */
    emit_byte(e, 0x58 | (CORE & 7));
}

/* EAX = clamp(EAX >> 16, 0, 255). Clobbers ECX. */
static void
emit_sat(emitter *e)
{
    shift(e, 7, AX, 16, false); /* sar eax, 16 */
    imm(e, CX, 255);
    rr(e, 0x39, CX, AX, false);      /* cmp eax, ecx */
    rr(e, 0x0f4f, AX, CX, false);    /* cmovg eax, ecx */
    rr(e, 0x31, CX, CX, false);      /* xor ecx, ecx */
    rr(e, 0x85, AX, AX, false);      /* test eax, eax */
    rr(e, 0x0f4c, AX, CX, false);    /* cmovl eax, ecx */
}

/* fg = (fg & ~mask) | (EAX & mask). Clobbers ECX. */
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

/* Mirrors step_param() and pixel_set() with the value in EAX and the delta in
   ECX. Clobbers EAX and ECX, and every caller-saved register for Q. */
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
            rr(e, 0x85, CX, CX, false);
            core_mem(e, 0x0f98, 0, PE(x_dir), false); /* sets */
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
            emit_byte(e, 0xa9); /* test eax, imm32 */
            dword(e, 0xf8000000);
            rr(e, 0x0f95, 0, CX, false); /* setnz cl */
            core_mem(e, 0xf7, 0, PE(src_mode), false);
            dword(e, SRC_Z_SCISSOR);
            rr(e, 0x0f95, 0, AX, false); /* setnz al */
            rr(e, 0x20, AX, CX, false);  /* and cl, al */
            core_mem(e, 0x88, CX, PE(z_ovf), false);
            break;
        case P_U:
        case P_V:
            {
                rr(e, 0x85, CX, CX, false);
                core_mem(e, 0x0f98, 0, param == P_U ? PE(u_dir) : PE(v_dir), false);
                core_mem(e, 0xf7, 0, PE(src_mode), false);
                dword(e, SRC_SWAP_UV);
                opcode(e, 0x0f85); /* jnz swapped */
                uint8_t *swapped = e->pos;
                dword(e, 0);
                emit_uv_store(e, param == P_U);
                emit_byte(e, 0xe9); /* jmp done */
                uint8_t *done = e->pos;
                dword(e, 0);
                rel32(swapped, e->pos);
                emit_uv_store(e, param != P_U);
                rel32(done, e->pos);
                break;
            }
        case P_Q:
            emit_call(e, (const void *) risc_seed_q, true, 0);
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
            rr(e, 0x69, AX, AX, false); /* imul eax, eax, 0x010101 */
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
    rr(e, 0x89, AX, R8, false);
    emit_alu(e, 0, s2 | 1, s1 | 1, false);
    write_reg(e, d & ~1U, R8);
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
    emit_byte(e, 0x49); /* mov r11, site */
    emit_byte(e, 0xbb);
    memcpy(e->pos, &address, 8);
    e->pos += 8;
    for (unsigned i = 0; i < 6; i++) {
        core_mem(e, 0x8b, AX, keys[i][0], false);
        emit_byte(e, 0x41); /* cmp eax, [r11 + key] */
        emit_byte(e, 0x3b);
        emit_byte(e, 0x43);
        emit_byte(e, keys[i][1]);
        opcode(e, 0x0f85);
        miss[i] = e->pos;
        dword(e, 0);
    }
    emit_byte(e, 0x49); /* mov rax, [r11 + fn] */
    emit_byte(e, 0x8b);
    emit_byte(e, 0x43);
    emit_byte(e, offsetof(verite_draw_site, fn));
    emit_byte(e, 0xeb); /* jmp call */
    uint8_t *hit = e->pos++;
    for (unsigned i = 0; i < 6; i++)
        rel32(miss[i], e->pos);
    emit_byte(e, 0x48); /* mov rax, pixel_jit_site_miss */
    emit_byte(e, 0xb8);
    uint64_t handler = (uint64_t) (uintptr_t) pixel_jit_site_miss;
    memcpy(e->pos, &handler, 8);
    e->pos += 8;
    *hit = (uint8_t) (e->pos - (hit + 1));

    emit_byte(e, 0x41); /* push r10 */
    emit_byte(e, 0x50 | (CORE & 7));
    emit_byte(e, 0x41); /* push r9 */
    emit_byte(e, 0x50 | (MEM & 7));
    emit_byte(e, 0x50 | DX); /* push rdx */
    rr(e, 0x83, 5, SP, true); /* sub rsp, 32 */
    emit_byte(e, 32);
    rr(e, 0x89, CORE, ARG_CORE, true);
    rr(e, 0x89, INDEX, ARG_MEMORY, true); /* second argument: the site */
    rr(e, 0xff, 2, AX, false);            /* call rax */
    rr(e, 0x83, 0, SP, true);             /* add rsp, 32 */
    emit_byte(e, 32);
    emit_byte(e, 0x58 | DX);
    emit_byte(e, 0x41);
    emit_byte(e, 0x58 | (MEM & 7));
    emit_byte(e, 0x41);
    emit_byte(e, 0x58 | (CORE & 7));
    if (!check)
        return NULL;

    emit_byte(e, 0x49); /* mov r11, site */
    emit_byte(e, 0xbb);
    memcpy(e->pos, &address, 8);
    e->pos += 8;
    emit_byte(e, 0x41); /* mov eax, [r11 + lo] */
    emit_byte(e, 0x8b);
    emit_byte(e, 0x43);
    emit_byte(e, offsetof(verite_draw_site, lo));
    alu_imm(e, 7, AX, code_last);
    opcode(e, 0x0f87); /* ja ok */
    uint8_t *after = e->pos;
    dword(e, 0);
    emit_byte(e, 0x41); /* mov eax, [r11 + hi] */
    emit_byte(e, 0x8b);
    emit_byte(e, 0x43);
    emit_byte(e, offsetof(verite_draw_site, hi));
    alu_imm(e, 7, AX, code_start);
    opcode(e, 0x0f82); /* jb ok */
    uint8_t *below = e->pos;
    dword(e, 0);
    emit_byte(e, 0xe9); /* jmp exit */
    uint8_t *exit = e->pos;
    dword(e, 0);
    rel32(after, e->pos);
    rel32(below, e->pos);
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
            imm(e, AX, risc_relative(pc, 2));
            write_reg(e, d, AX);
        }
        return true;
    }
    if (op == 0x76 || op == 0x77) {
        if (!plain_reg(d))
            return false;
        if (d >= 64) {
            imm(e, AX, op == 0x76 ? ir & 0xffff : ir << 16);
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
                uint32_t disp = reg_address(e, d);
                core_mem(e, s1 < 128 ? 0x83 : 0x81, sub == 0 ? 0 : 5, disp, segmented(d));
                if (s1 < 128)
                    emit_byte(e, s1);
                else
                    dword(e, s1);
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

    if (!((op >= 0x40 && op <= 0x4d && op != 0x42) || (op >= 0x52 && op <= 0x5e && op != 0x53 && op != 0x5b)) || !plain_reg(d) || !plain_reg(s2) || (op >= 0x50 && !plain_reg(s1)))
        return false;

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
                static const uint8_t groups[] = { 1, 4, 7, 5 }; /* ror, shl, sar, shr */
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
                static const uint8_t cc[] = { 0x9c, 0x92, 0x94 }; /* l, b, e */
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
            rr(e, 0xf7, 3, AX, false);    /* neg eax */
            rr(e, 0x0f4c, AX, CX, false); /* keep original if -a is negative */
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
emit_branch(emitter *e, uint32_t ir, uint32_t pc)
{
    unsigned op   = ir >> 24;
    unsigned d    = (uint8_t) (ir >> 16);
    unsigned s1   = (uint8_t) ir;
    uint32_t slot = risc_next(pc);
    uint32_t link = risc_next(slot);

    if (op >= 0x60 && op <= 0x65 && plain_reg(s1)) {
        static const uint8_t cc[] = { 0x44, 0x45, 0x49, 0x48, 0x4f, 0x4e };
        read_reg(e, AX, s1);
        imm(e, DX, link);
        imm(e, CX, risc_relative(slot, (int16_t) (ir >> 8)));
        rr(e, 0x85, AX, AX, false);
        rr(e, 0x0f00 | cc[op - 0x60], DX, CX, false);
    } else if (op == 0x6f && plain_reg(s1) && plain_reg(d)) {
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
        alu_imm(e, 0, AX, offset);
    alu_imm(e, 7, AX, LOCAL_END);
    emit_byte(e, 0x0f);
    emit_byte(e, 0x83); /* jae slow_path */
    uint8_t *exit = e->pos;
    dword(e, 0);

    /* and eax, [r9 + mask]; mov r8, [r9 + vram] */
    rex(e, false, AX, 0, MEM);
    emit_byte(e, 0x23);
    emit_byte(e, 0x40 | (MEM & 7));
    emit_byte(e, offsetof(jit_memory, mask));
    rex(e, true, R8, 0, MEM);
    emit_byte(e, 0x8b);
    emit_byte(e, ((R8 & 7) << 3) | (MEM & 7));
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
        /* A store may overwrite the next RISC instruction. End the block. */
        *store = true;
    } else if (pair) {
        packed_reg(e, 0, d, true);
    } else {
        write_reg(e, d, CX);
    }
    return exit;
}

static void
emit_finish(emitter *e, uint32_t pc, unsigned count, bool branch, bool delay_in_block, bool loop)
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
    } else {
        core_mem(e, 0x83, 0, CORE_FIELD(timer), false);
        emit_byte(e, count);
        imm(e, AX, count);
    }
    emit_byte(e, 0xc3);
}

/* Leave after instruction count - 1 of a possibly looping block. A loop has
   already run jit_budget - R9D cycles in earlier iterations. */
static void
emit_side_finish(emitter *e, uint32_t pc, unsigned count, bool loop)
{
    if (!loop) {
        emit_finish(e, pc, count, false, false, false);
        return;
    }
    core_imm(e, CORE_FIELD(pc), pc);
    core_imm(e, CORE_FIELD(npc), risc_next(pc));
    core_mem(e, 0x8b, AX, CORE_FIELD(jit_budget), false);
    rr(e, 0x29, MEM, AX, false);
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
    uint8_t            code[MAX_BLOCK_CODE];
    emitter            e = { code };
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

    emit_byte(&e, 0xf3); /* endbr64 (NOP on older hosts) */
    emit_byte(&e, 0x0f);
    emit_byte(&e, 0x1e);
    emit_byte(&e, 0xfa);
    rr(&e, 0x89, ARG_CORE, CORE, true);
    /* Reserve space for the memory argument or loop budget, filled in
       once we know the shape of the block. */
    uint8_t *prologue = e.pos;
    e.pos += 16;
    uint8_t *body = e.pos;
    while (count < limit) {
        if ((size_t) (e.pos - code) + MAX_INSN_CODE + (exit_count + 2) * FINISH_CODE > MAX_BLOCK_CODE)
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
        rr(&setup, 0x89, ARG_MEMORY, MEM, true);
    } else if (loop) {
        rr(&setup, 0x89, ARG_BUDGET, MEM, false);
        core_mem(&setup, 0x89, MEM, CORE_FIELD(jit_budget), false);

        /* R9D holds remaining cycles across iterations. Draw sites leave the
           loop when they write into this block, so its snapshot stays valid. */
        alu_imm(&e, 5, MEM, count);
        alu_imm(&e, 7, MEM, count);
        uint8_t *skip_loop = NULL;
        if (conditional) {
            opcode(&e, 0x0f82); /* jb over cmp/je */
            skip_loop = e.pos;
            dword(&e, 0);
            rr(&e, 0x81, 7, DX, false);
            dword(&e, start);
        }
        opcode(&e, conditional ? 0x0f84 : 0x0f83);
        dword(&e, (uint32_t) (body - (e.pos + 4)));
        if (skip_loop)
            rel32(skip_loop, e.pos);
    }
    emit_finish(&e, pc, count, branch, delay_in_block, loop);
    for (unsigned i = 0; i < count; i++) {
        if (exits[i]) {
            rel32(exits[i], e.pos);
            emit_finish(&e, risc_relative(start, i), i, false, false, false);
        }
        if (after[i]) {
            rel32(after[i], e.pos);
            emit_side_finish(&e, risc_relative(start, i + 1), i + 1, loop);
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
