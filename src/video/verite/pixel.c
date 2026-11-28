/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Vérité pixel engine, ported from rust-dos. */

#include "internal.h"

typedef struct {
    pixel_param even, odd;
} param_pair;
typedef struct {
    param_pair pair;
    bool       shadow;
} first_pair;
typedef struct {
    int32_t channel[4];
} color;

static const first_pair pair1[] = {
    { { P_X, P_NONE },    false },
    { { P_Q, P_NONE },    true  },
    { { P_X, P_Y },       false },
    { { P_Q, P_Y },       true  },
    { { P_X, P_U },       false },
    { { P_Q, P_U },       true  },
    { { P_X, P_R },       false },
    { { P_Q, P_R },       true  },
    { { P_X, P_A },       false },
    { { P_Q, P_A },       true  },
    { { P_Z, P_X },       false },
    { { P_NONE, P_NONE }, true  },
    { { P_X, P_F },       false },
    { { P_Q, P_F },       true  },
    { { P_Y, P_U },       true  },
    { { P_Q, P_U },       true  },
};

static const param_pair pair2[] = {
    { P_X,    P_V    },
    { P_Y,    P_V    },
    { P_R,    P_V    },
    { P_NONE, P_V    },
    { P_I,    P_V    },
    { P_Z,    P_V    },
    { P_A,    P_V    },
    { P_Z,    P_U    },
    { P_G,    P_B    },
    { P_Z,    P_A    },
    { P_Z,    P_NONE },
    { P_NONE, P_F    },
    { P_Z,    P_F    },
    { P_F,    P_V    },
    { P_R,    P_F    },
    { P_R,    P_A    },
    { P_R,    P_NONE },
    { P_Z,    P_R    },
    { P_NONE, P_A    },
    { P_F,    P_A    },
    { P_I,    P_NONE },
    { P_Z,    P_I    },
    { P_I,    P_A    },
    { P_I,    P_F    },
};

static const param_pair pair3[] = {
    { P_G,    P_B    },
    { P_Z,    P_A    },
    { P_Z,    P_NONE },
    { P_NONE, P_F    },
    { P_Z,    P_F    },
    { P_NONE, P_NONE },
    { P_R,    P_F    },
    { P_R,    P_A    },
    { P_R,    P_NONE },
    { P_Z,    P_R    },
    { P_NONE, P_A    },
    { P_F,    P_A    },
    { P_I,    P_NONE },
    { P_Z,    P_I    },
    { P_I,    P_A    },
    { P_I,    P_F    },
};

enum {
    DST_ZBUF_WRITE    = 1 << 16,
    DST_YUV2RGB       = 1 << 18,
    DST_BLEND         = 1 << 19,
    DST_DITHER        = 1 << 20,
    DST_FOG           = 1 << 21,
    DST_NO_PAD        = 1 << 22,
    DST_READ_DISABLE  = 1 << 23,
    DST_BGR           = 1 << 24,
    DST_TRANSP_REJECT = 1 << 25,
};

void
pixel_reset(pixel_engine *pe)
{
    *pe = (pixel_engine) {
        .src_mask     = 0x07FF07FF,
        .dst_mode     = 0xC,
        .dst_fmt      = 4,
        .pmask        = UINT32_MAX,
        .scissor_x    = 0x800,
        .scissor_y    = 0x800,
        .fg           = 0xFF000000,
        .yuv_contrast = 41,
        .u_mask       = 0x7FF,
        .v_mask       = 0x7FF,
        .q            = 1 << 24,
        .a            = 255 << 16,
        .f            = 255 << 16,
    };
}

static int32_t
clamp(int32_t value, int32_t max)
{
    return value < 0 ? 0 : value > max ? max
                                       : value;
}

static int32_t
sat(int32_t value)
{
    return clamp(value >> 16, 255);
}

static unsigned
fmt_bytes(unsigned fmt)
{
    if (fmt >= 1 && fmt <= 3)
        return 1;
    if (fmt >= 8 && fmt <= 10)
        return 0;
    return fmt == 12 || fmt == 13 ? 4 : 2;
}

static uint32_t
line_stride(uint32_t code)
{
    return verite_line_stride(code);
}

static uint32_t
read_pixel(verite_memory mem, uint32_t addr, unsigned bytes)
{
    if (bytes == 1)
        return mem_read8(mem, addr);
    if (bytes == 2)
        return mem_read16(mem, addr);
    return mem_read32(mem, addr);
}

static void
write_pixel(verite_memory mem, uint32_t addr, unsigned bytes, uint32_t v)
{
    if (bytes == 1)
        mem_write8(mem, addr, v);
    else if (bytes == 2)
        mem_write16(mem, addr, v);
    else
        mem_write32(mem, addr, v);
}

static inline uint32_t
rop(uint32_t code, uint32_t s, uint32_t d)
{
    return (~s & ~d & (0U - (code & 1))) | (~s & d & (0U - ((code >> 1) & 1))) | (s & ~d & (0U - ((code >> 2) & 1))) | (s & d & (0U - ((code >> 3) & 1)));
}

static bool
rop_reads(uint32_t code)
{
    return ((code >> 1) ^ code) & 5;
}

static inline int32_t
widen(uint32_t value, unsigned bits, bool no_pad)
{
    static const uint8_t widen3[8]  = { 0, 36, 73, 109, 146, 182, 219, 255 };
    static const uint8_t widen5[32] = {
        0, 8, 16, 25, 33, 41, 49, 58, 66, 74, 82, 90, 99, 107, 115, 123,
        132, 140, 148, 156, 165, 173, 181, 189, 197, 206, 214, 222, 230, 239, 247, 255
    };
    static const uint8_t widen6[64] = {
        0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 45, 49, 53, 57, 61,
        65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125,
        130, 134, 138, 142, 146, 150, 154, 158, 162, 166, 170, 174, 178, 182, 186, 190,
        194, 198, 202, 206, 210, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255
    };
    unsigned max = (1U << bits) - 1;
    uint32_t v   = value & max;
    if (no_pad)
        return (int32_t) (v << (8 - bits));
    switch (bits) {
        case 2:
            return (int32_t) (v * 85U);
        case 3:
            return widen3[v];
        case 4:
            return (int32_t) (v * 17U);
        case 5:
            return widen5[v];
        case 6:
            return widen6[v];
        default:
            return (int32_t) ((v * 255U + max / 2) / max);
    }
}

static inline int32_t
div255(int32_t x)
{
    return (int32_t) (((uint32_t) x * 0x8081U) >> 23);
}

static color
argb(uint32_t word)
{
    return (color) {
        { (word >> 16) & 255, (word >> 8) & 255, word & 255, word >> 24 }
    };
}

static color
decode(unsigned fmt, uint32_t raw, bool bgr, bool no_pad)
{
    color c;
    switch (fmt) {
        case 1:
            c = (color) {
                { widen(raw >> 5, 3, no_pad), widen(raw >> 2, 3, no_pad), widen(raw, 2, no_pad), 255 }
            };
            break;
        case 2:
            {
                int32_t i = raw & 255;
                c         = (color) {
                            { i, i, i, 255 }
                };
                break;
            }
        case 3:
            c = (color) {
                { 255, 255, 255, raw & 255 }
            };
            break;
        case 5:
            c = (color) {
                { widen(raw >> 8, 4, no_pad), widen(raw >> 4, 4, no_pad), widen(raw, 4, no_pad), widen(raw >> 12, 4, no_pad) }
            };
            break;
        case 6:
            c = (color) {
                { widen(raw >> 10, 5, no_pad), widen(raw >> 5, 5, no_pad), widen(raw, 5, no_pad), raw & 0x8000 ? 255 : 0 }
            };
            break;
        case 12:
            c = argb(raw);
            break;
        default:
            c = (color) {
                { widen(raw >> 11, 5, no_pad), widen(raw >> 5, 6, no_pad), widen(raw, 5, no_pad), 255 }
            };
            break;
    }
    if (bgr) {
        int32_t r    = c.channel[0];
        c.channel[0] = c.channel[2];
        c.channel[2] = r;
    }
    return c;
}

static uint32_t
quantize(int32_t value, unsigned n, unsigned dither)
{
    unsigned v = clamp(value, 255) + ((dither << 4) >> n);
    return (v < 255 ? v : 255) >> (8 - n);
}

static inline uint32_t
encode(unsigned fmt, color c, unsigned dither, bool bgr)
{
    int32_t r = c.channel[bgr ? 2 : 0], g = c.channel[1], b = c.channel[bgr ? 0 : 2], a = c.channel[3];
    switch (fmt) {
        case 1:
            return (quantize(r, 3, dither) << 5) | (quantize(g, 3, dither) << 2) | quantize(b, 2, dither);
        case 2:
            return quantize(r, 8, dither);
        case 3:
            return quantize(a, 8, dither);
        case 5:
            return (quantize(a, 4, dither) << 12) | (quantize(r, 4, dither) << 8) | (quantize(g, 4, dither) << 4) | quantize(b, 4, dither);
        case 6:
            return (quantize(a, 1, dither) << 15) | (quantize(r, 5, dither) << 10) | (quantize(g, 5, dither) << 5) | quantize(b, 5, dither);
        case 12:
            return (quantize(a, 8, dither) << 24) | (quantize(r, 8, dither) << 16) | (quantize(g, 8, dither) << 8) | quantize(b, 8, dither);
        default:
            return (quantize(r, 5, dither) << 11) | (quantize(g, 6, dither) << 5) | quantize(b, 5, dither);
    }
}

static void
field_set(uint32_t *reg, uint32_t mask, uint32_t value)
{
    *reg = (*reg & ~mask) | (value & mask);
}

void
pixel_set(pixel_engine *pe, pixel_param param, uint32_t value)
{
    int32_t v = (int32_t) value;
    switch (param) {
        case P_NONE:
            break;
        case P_XSHADOW:
            pe->x_shadow = v;
            break;
        case P_X:
            pe->x = v;
            break;
        case P_Y:
            pe->y = v;
            break;
        case P_Z:
            pe->z     = v;
            pe->z_ovf = (value & 0xf8000000) && (pe->src_mode & SRC_Z_SCISSOR);
            break;
        case P_U:
        case P_V:
            if ((param == P_U) != !!(pe->src_mode & SRC_SWAP_UV)) {
                pe->u      = v;
                pe->u_mask = pe->src_mask & 0x7ff;
            } else {
                pe->v      = v;
                pe->v_mask = (pe->src_mask >> 16) & 0x7ff;
            }
            break;
        case P_Q:
            pe->q = value;
            break;
        case P_R:
            pe->r = v;
            field_set(&pe->fg, 0xff0000, (uint32_t) sat(v) << 16);
            break;
        case P_G:
            pe->g = v;
            field_set(&pe->fg, 0xff00, (uint32_t) sat(v) << 8);
            break;
        case P_B:
            pe->b = v;
            field_set(&pe->fg, 0xff, sat(v));
            break;
        case P_A:
            pe->a = v;
            field_set(&pe->fg, 0xff000000, (uint32_t) sat(v) << 24);
            break;
        case P_F:
            pe->f = v;
            break;
        case P_I:
            {
                uint32_t gray = sat(v);
                pe->r = pe->g = pe->b = v;
                field_set(&pe->fg, 0xffffff, gray * 0x010101);
                break;
            }
        case P_CNT:
            pe->cnt = value >> 16;
            break;
    }
}

static void
set_fg(pixel_engine *pe, uint32_t value, bool alpha)
{
    field_set(&pe->fg, alpha ? UINT32_MAX : 0xffffff, value);
    color c = argb(pe->fg);
    pe->r   = c.channel[0] << 16;
    pe->g   = c.channel[1] << 16;
    pe->b   = c.channel[2] << 16;
    pe->a   = c.channel[3] << 16;
}

uint32_t
pixel_state(const pixel_engine *pe, uint8_t index)
{
    switch (index) {
        case 0:
            return pe->src_base;
        case 1:
            return pe->src_mode;
        case 2:
            return pe->src_mask;
        case 3:
            return pe->stride;
        case 4:
            return pe->dst_base;
        case 5:
            return pe->dst_mode;
        case 6:
            return pe->dst_fmt;
        case 7:
            return pe->pmask;
        case 8:
            return pe->pattern;
        case 12:
            return pe->pat_mode;
        case 13:
            return pe->pat_offset;
        case 14:
            return pe->scissor_x;
        case 15:
            return pe->scissor_y;
        case 16:
            return pe->z_base;
        case 17:
            return pe->dither_offset;
        case 19:
            return pe->fg;
        case 20:
            return pe->bg;
        case 21:
            return pe->fog_color;
        case 22:
            return pe->dst_color;
        case 23:
            return pe->chroma_color;
        case 24:
            return pe->yuv_contrast;
        case 26:
            return pe->chroma_mask;
        case 28:
            return ((uint32_t) pe->pick_valid << 16) | ((uint32_t) pe->pick_hit << 17);
        case 29:
            return pe->alpha_thres;
        case 30:
            return pe->spec_color;
        case 112:
            return pe->x;
        case 113:
            return pe->y;
        case 114:
            return ((uint32_t) pe->z_ovf << 27) | ((uint32_t) pe->z & 0x07ffffff);
        case 117:
            return (uint32_t) sat(pe->f) << 16;
        case 119:
            return (uint32_t) pe->u & 0x07ffff00;
        case 121:
            return (uint32_t) pe->v & 0x07ffff00;
        case 123:
            return (uint32_t) pe->u_dir << 31;
        case 125:
            return (uint32_t) pe->v_dir << 31;
        case 131:
            return pe->drawctl;
        case 132:
            return ((uint32_t) pe->x_shadow & 0xffff0000) | pe->x_dir;
        case 133:
            return pe->cnt << 16;
        default:
            return index >= 32 && index <= 39 ? pe->tex4pal[index - 32] : 0;
    }
}

static bool
inside(const pixel_engine *pe, int32_t x, int32_t y)
{
    return x >= (int32_t) ((pe->scissor_x >> 16) & 0xfff) && x < (int32_t) (pe->scissor_x & 0xfff) && x < 2048 && y >= (int32_t) ((pe->scissor_y >> 16) & 0xfff) && y < (int32_t) (pe->scissor_y & 0xfff) && y < 2048;
}

/* Returns false for a transparent pattern pixel, advancing PatOffset either way. */
static bool
pattern_pixel(pixel_engine *pe, bool *background)
{
    *background = false;
    if (!(pe->pat_mode & 0x10000))
        return true;
    *background    = !(pe->pattern & (UINT32_C(1) << (31 - (pe->pat_offset & 31))));
    pe->pat_offset = (pe->pat_offset & 31) != (pe->pat_mode & 31) ? (pe->pat_offset + 1) & 31 : 0;
    return !*background || (pe->pat_mode & 0x20000);
}

static unsigned
dst_bytes(const pixel_engine *pe)
{
    unsigned bytes = fmt_bytes(pe->dst_fmt & 15);
    return bytes ? bytes : 1;
}

static uint32_t
dst_at(const pixel_engine *pe, int32_t x, int32_t y)
{
    return pe->dst_base + line_stride(pe->stride >> 8) * (uint32_t) y + (uint32_t) x * dst_bytes(pe);
}

static unsigned
src_fmt(const pixel_engine *pe)
{
    unsigned fmt = pe->src_mode & 15;
    return fmt ? fmt : pe->dst_fmt & 15;
}

typedef struct {
    uint32_t src_base;
    uint32_t src_stride;
    uint32_t chroma_color;
    uint32_t chroma_mask;
    unsigned fmt;
    unsigned bytes;
    bool     filter;
    bool     u_clamp;
    bool     v_clamp;
    bool     bgr;
    bool     no_pad;
    bool     chroma;
    bool     chroma_black_yuv;
} tex_sampler;

static tex_sampler
tex_sampler_init(const pixel_engine *pe)
{
    uint32_t mode = pe->src_mode;
    unsigned fmt  = src_fmt(pe);
    return (tex_sampler) {
        .src_base         = pe->src_base,
        .src_stride       = line_stride(pe->stride),
        .chroma_color     = pe->chroma_color,
        .chroma_mask      = pe->chroma_mask,
        .fmt              = fmt,
        .bytes            = fmt_bytes(fmt),
        .filter           = !!(mode & SRC_FILTER),
        .u_clamp          = !!(mode & SRC_U_CLAMP),
        .v_clamp          = !!(mode & SRC_V_CLAMP),
        .bgr              = !!(mode & SRC_BGR),
        .no_pad           = !!(mode & SRC_NO_PAD),
        .chroma           = !!(mode & SRC_CHROMA),
        .chroma_black_yuv = !!(mode & SRC_CHROMA_BLACK_YUV),
    };
}

static bool
keyed_with(const tex_sampler *s, uint32_t raw, unsigned bytes)
{
    if (!s->chroma)
        return false;
    if (bytes == 1)
        raw *= 0x01010101;
    else if (bytes == 2)
        raw *= 0x00010001;
    return ((raw ^ s->chroma_color) & s->chroma_mask) == 0;
}

static color
key_color_with(bool chroma_black_yuv)
{
    return chroma_black_yuv ? (color) {
        { 128, 16, 128, 0 }
    }
                            : (color) { 0 };
}

static bool
keyed(const pixel_engine *pe, uint32_t raw, unsigned bytes)
{
    if (!(pe->src_mode & SRC_CHROMA))
        return false;
    if (bytes == 1)
        raw *= 0x01010101;
    else if (bytes == 2)
        raw *= 0x00010001;
    return ((raw ^ pe->chroma_color) & pe->chroma_mask) == 0;
}

static uint32_t
texture_coord(int32_t coord, uint32_t mask, bool clamped)
{
    return clamped ? (uint32_t) clamp(coord, mask) : (uint32_t) coord & mask;
}

static color
texel_with(const pixel_engine *pe, const tex_sampler *s, uint32_t row, uint32_t tx, verite_memory mem)
{
    if (s->fmt >= 8 && s->fmt <= 10) {
        uint32_t byte  = mem_read8(mem, row + tx / 2);
        unsigned index = tx & 1 ? byte & 15 : byte >> 4;
        uint32_t entry = (pe->tex4pal[index / 2] >> (index & 1 ? 0 : 16)) & 0xffff;
        return decode(s->fmt - 4, entry, s->bgr, s->no_pad);
    }
    if (s->fmt == 13) {
        uint32_t word = mem_read32(mem, row + tx / 2 * 4);
        if (keyed_with(s, word, 4))
            return key_color_with(s->chroma_black_yuv);
        return (color) {
            { (word >> (s->bgr ? 0 : 16)) & 255, (word >> (tx & 1 ? 8 : 24)) & 255,
             (word >> (s->bgr ? 16 : 0)) & 255, 255 }
        };
    }
    uint32_t raw = read_pixel(mem, row + tx * s->bytes, s->bytes);
    return keyed_with(s, raw, s->bytes) ? key_color_with(s->chroma_black_yuv) : decode(s->fmt, raw, s->bgr, s->no_pad);
}

static color
sample_bilinear_565(const uint32_t taps[4], const int32_t weights[4], bool no_pad)
{
    color sum = { { 0, 0, 0, 255 } };
    if (!no_pad) {
        /* Expand each padded texel before filtering to preserve its rounding. */
        for (unsigned i = 0; i < 4; i++) {
            sum.channel[0] += widen(taps[i] >> 11, 5, false) * weights[i];
            sum.channel[1] += widen(taps[i] >> 5, 6, false) * weights[i];
            sum.channel[2] += widen(taps[i], 5, false) * weights[i];
        }
        for (unsigned i = 0; i < 3; i++)
            sum.channel[i] = (sum.channel[i] + 32768) >> 16;
        return sum;
    }

    for (unsigned i = 0; i < 4; i++) {
        sum.channel[0] += (taps[i] >> 11) * weights[i];
        sum.channel[1] += ((taps[i] >> 5) & 63) * weights[i];
        sum.channel[2] += (taps[i] & 31) * weights[i];
    }
    sum.channel[0] = (sum.channel[0] + 4096) >> 13;
    sum.channel[1] = (sum.channel[1] + 8192) >> 14;
    sum.channel[2] = (sum.channel[2] + 4096) >> 13;
    return sum;
}

static inline color
sample_with(const pixel_engine *pe, const tex_sampler *s, int32_t u, int32_t v, verite_memory mem)
{
    int32_t  tx   = u >> 16, ty = v >> 16;
    uint32_t x0   = texture_coord(tx, pe->u_mask, s->u_clamp);
    uint32_t row0 = s->src_base + s->src_stride * texture_coord(ty, pe->v_mask, s->v_clamp);
    if (!s->filter)
        return texel_with(pe, s, row0, x0, mem);

    /* Map each neighbor independently so a tap can wrap or clamp at an edge. */
    uint32_t x1        = texture_coord(tx + 1, pe->u_mask, s->u_clamp);
    uint32_t row1      = s->src_base + s->src_stride * texture_coord(ty + 1, pe->v_mask, s->v_clamp);
    int32_t  fu        = (u >> 8) & 255, fv = (v >> 8) & 255;
    int32_t  weights[] = { (256 - fu) * (256 - fv), fu * (256 - fv), (256 - fu) * fv, fu * fv };
    if (s->fmt == 4 && !s->bgr && !s->chroma) {
        uint32_t taps[] = { mem_read16(mem, row0 + 2 * x0), mem_read16(mem, row0 + 2 * x1),
                            mem_read16(mem, row1 + 2 * x0), mem_read16(mem, row1 + 2 * x1) };
        return sample_bilinear_565(taps, weights, s->no_pad);
    }
    if (s->fmt == 2 && !s->chroma) {
        int32_t intensity = mem_read8(mem, row0 + x0) * weights[0] + mem_read8(mem, row0 + x1) * weights[1]
                          + mem_read8(mem, row1 + x0) * weights[2] + mem_read8(mem, row1 + x1) * weights[3];
        intensity = (intensity + 32768) >> 16;
        return (color) { { intensity, intensity, intensity, 255 } };
    }

    const uint32_t rows[] = { row0, row1 };
    const uint32_t xs[]   = { x0, x1 };
    color          sum    = { 0 };
    for (unsigned tap = 0; tap < 4; tap++) {
        color c = texel_with(pe, s, rows[tap >> 1], xs[tap & 1], mem);
        for (unsigned k = 0; k < 4; k++)
            sum.channel[k] += c.channel[k] * weights[tap];
    }
    for (unsigned k = 0; k < 4; k++)
        sum.channel[k] = (sum.channel[k] + 32768) >> 16;
    return sum;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((unused))
#endif
static inline color
sample(const pixel_engine *pe, int32_t u, int32_t v, verite_memory mem)
{
    tex_sampler s = tex_sampler_init(pe);
    return sample_with(pe, &s, u, v, mem);
}

static int32_t
factor(uint32_t func, bool source, unsigned k, color s, color d)
{
    int32_t saturation = k == 3 ? 255 : clamp(s.channel[3], 255 - d.channel[3]);
    switch (func) {
        case 0:
            return source ? d.channel[k] : s.channel[k];
        case 1:
            return 255 - (source ? d.channel[k] : s.channel[k]);
        case 2:
            return s.channel[3];
        case 3:
            return 255 - s.channel[3];
        case 4:
            return d.channel[3];
        case 5:
            return 255 - d.channel[3];
        case 6:
            return 0;
        case 8:
            return saturation;
        case 9:
            return 255 - saturation;
        case 10:
            return source ? s.channel[k] : d.channel[k];
        case 11:
            return 255 - (source ? s.channel[k] : d.channel[k]);
        default:
            return 255;
    }
}

static inline color
blend_with(unsigned sf, unsigned df, color s, color d)
{
    color out;
    for (unsigned k = 0; k < 4; k++) {
        int32_t a      = sf == 15 ? s.channel[k] - 128 : div255(s.channel[k] * factor(sf, true, k, s, d) + 127);
        int32_t b      = df == 15 ? d.channel[k] - 128 : div255(d.channel[k] * factor(df, false, k, s, d) + 127);
        out.channel[k] = clamp(a + b, 255);
    }
    return out;
}

static color
yuv_to_rgb_with(int32_t yuv_black, int32_t yuv_scale, color c)
{
    int32_t cr = c.channel[0] - 128, cb = c.channel[2] - 128;
    int32_t sy = yuv_scale * (c.channel[1] - yuv_black) + 128;
    /* Euclidean and truncating division agree after clamping negative channels. */
    return (color) {
        { clamp((sy + 407 * cr) / 255, 255),
         clamp((sy - 207 * cr - 100 * cb) / 255, 255),
         clamp((sy + 515 * cb) / 255, 255), c.channel[3] }
    };
}

void
pixel_spr(pixel_engine *pe, uint32_t index, uint32_t value, verite_memory mem)
{
    static const uint8_t source_bits[] = { 7, 9, 10, 11, 12, 13 };
    switch (index) {
        case 0:
            pe->src_base = value;
            break;
        case 1:
            pe->src_mode = value;
            break;
        case 2:
            pe->src_mask = value;
            break;
        case 3:
            pe->stride = value;
            break;
        case 4:
            pe->dst_base = value;
            break;
        case 5:
            pe->dst_mode = value;
            break;
        case 6:
            pe->dst_fmt = value;
            break;
        case 7:
            pe->pmask = value;
            break;
        case 8:
            pe->pattern    = value;
            pe->pat_offset = 0;
            break;
        case 10:
            pe->pattern = 0;
            for (unsigned byte = 0; byte < 4; byte++)
                pe->pattern = (pe->pattern << 8) | mem_read8(mem, value + byte);
            pe->pat_offset = 0;
            break;
        case 12:
            pe->pat_mode = value;
            break;
        case 13:
            pe->pat_offset = value & 31;
            break;
        case 14:
            pe->scissor_x = value;
            break;
        case 15:
            pe->scissor_y = value;
            break;
        case 16:
            pe->z_base = value;
            break;
        case 17:
            pe->dither_offset = value;
            break;
        case 19:
            set_fg(pe, value, true);
            break;
        case 20:
            pe->bg = value;
            break;
        case 21:
            pe->fog_color = value;
            break;
        case 22:
            pe->dst_color = value;
            break;
        case 23:
            pe->chroma_color = value;
            break;
        case 24:
            pe->yuv_contrast = value;
            break;
        case 26:
            pe->chroma_mask = value;
            break;
        case 27:
            set_fg(pe, value, false);
            break;
        case 28:
            pe->pick_valid = !!(value & 0x10000);
            pe->pick_hit   = !!(value & 0x20000);
            break;
        case 29:
            pe->alpha_thres = value;
            break;
        case 30:
            pe->spec_color = value;
            break;
        case 48:
            field_set(&pe->src_mode, 0xf, value);
            break;
        case 49:
            field_set(&pe->src_mode, 0x70, value);
            break;
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 55:
            field_set(&pe->src_mode, 1U << source_bits[index - 50], value);
            break;
        case 56:
            field_set(&pe->src_mask, 0x7ff, value);
            break;
        case 57:
            field_set(&pe->src_mask, 0x7ff0000, value);
            break;
        case 58:
        case 59:
        case 60:
            field_set(&pe->stride, 0x7fU << (8 * (index - 58)), value);
            break;
        case 61:
            field_set(&pe->src_mode, SRC_CHROMA_BLACK_YUV, value);
            break;
        case 62:
            /* The single clamp bit covers both texture axes. */
            field_set(&pe->src_mode, SRC_U_CLAMP | SRC_V_CLAMP,
                      (value & SRC_U_CLAMP) ? SRC_U_CLAMP | SRC_V_CLAMP : 0);
            break;
        case 63:
            field_set(&pe->src_mode, 1U << 17, value);
            break;
        case 64:
            field_set(&pe->dst_mode, 0xf, value);
            break;
        case 65:
            field_set(&pe->dst_mode, 0xf0, value);
            break;
        case 66:
            field_set(&pe->dst_mode, 0xf00, value);
            break;
        case 67:
            field_set(&pe->dst_mode, 0x7000, value);
            break;
        case 68:
            field_set(&pe->dst_mode, DST_ZBUF_WRITE, value);
            break;
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
        case 74:
        case 75:
        case 76:
            field_set(&pe->dst_mode, 1U << (index - 69 + 18), value);
            break;
        case 80:
            field_set(&pe->pat_mode, 0x1f, value);
            break;
        case 81:
            field_set(&pe->pat_mode, 1U << 16, value);
            break;
        case 82:
            field_set(&pe->pat_mode, 1U << 17, value);
            break;
        case 83:
            field_set(&pe->src_mode, 1U << 17, value);
            break;
        default:
            if (index >= 32 && index <= 39)
                pe->tex4pal[index - 32] = value;
            break;
    }
}

static uint32_t
pixel_mask(unsigned bytes)
{
    return bytes == 4 ? UINT32_MAX : (1U << (8 * bytes)) - 1;
}

static void
pixel_parallel(verite_core *c, int32_t x, int32_t y, int32_t sx, int32_t sy, verite_memory mem)
{
    pixel_engine *pe = &c->pe;
    bool          background;
    if (!pattern_pixel(pe, &background) || !inside(pe, x, y))
        return;
    unsigned bytes = dst_bytes(pe);
    uint32_t addr  = dst_at(pe, x, y);
    uint32_t mask  = pixel_mask(bytes);
    uint32_t source;
    if (!((pe->src_mode >> 4) & 7) || background) {
        /* A colour word packs one pixel for each position within the word. */
        uint32_t word  = background ? pe->bg : pe->fg;
        unsigned shift = 8 * (4 - bytes - (addr & 3 & ~(bytes - 1)));
        source         = (word >> shift) & mask;
    } else {
        unsigned n = fmt_bytes(src_fmt(pe));
        if (!n)
            n = 1;
        uint32_t row = pe->src_base + line_stride(pe->stride) * ((uint32_t) sy & pe->v_mask);
        source       = read_pixel(mem, row + ((uint32_t) sx & pe->u_mask) * n, n);
        if (keyed(pe, source, n))
            return;
    }
    unsigned code   = pe->dst_mode & 15;
    uint32_t pmask  = pe->pmask & mask;
    uint32_t old    = rop_reads(code) || pmask != mask ? read_pixel(mem, addr, bytes) : 0;
    uint32_t result = (rop(code, source, old) & pmask) | (old & ~pmask);
    write_pixel(mem, addr, bytes, result & mask);
    verite_dirty_vram_page(c, (addr & mem.mask) >> 12);
    pe->pick_hit = c->drew = true;
}

static inline void
dirty_span_bytes(verite_core *c, verite_memory mem, uint32_t start, uint32_t len)
{
    if (!len)
        return;
    uint32_t lo = start & (uint32_t) mem.mask;
    uint32_t hi = (start + len - 1) & (uint32_t) mem.mask;
    if (lo <= hi)
        verite_dirty_vram_range(c, lo, hi);
    else {
        verite_dirty_vram_range(c, lo, (uint32_t) mem.mask);
        verite_dirty_vram_range(c, 0, hi);
    }
}

/* Copies and fills with no pattern, chroma key, raster op or plane mask: the
   same pixels as pixel_parallel(), with the per-span state hoisted. */
static bool
pixel_parallel_copy(verite_core *c, unsigned n, uint32_t step, verite_memory mem)
{
    pixel_engine *pe     = &c->pe;
    unsigned      bytes  = dst_bytes(pe);
    uint32_t      mask   = pixel_mask(bytes);
    bool          filled = !((pe->src_mode >> 4) & 7);
    if ((pe->pat_mode & 0x10000) || (pe->dst_mode & 15) != 12 || (pe->pmask & mask) != mask
        || (!filled && (pe->src_mode & SRC_CHROMA)))
        return false;

    uint32_t final_x_shadow = (uint32_t) pe->x_shadow + step * n;
    uint32_t final_u        = (uint32_t) pe->u + step * n;
    int32_t  y              = pe->y >> 16;
    bool     row_inside     = y >= (int32_t) ((pe->scissor_y >> 16) & 0xfff) && y < (int32_t) (pe->scissor_y & 0xfff) && y < 2048;
    int32_t  x_min          = (int32_t) ((pe->scissor_x >> 16) & 0xfff);
    int32_t  x_end          = (int32_t) (pe->scissor_x & 0xfff);
    if (x_end > 2048)
        x_end = 2048;
    if (n == 0 || !row_inside || x_min >= x_end) {
        pe->x_shadow = (int32_t) final_x_shadow;
        pe->u        = (int32_t) final_u;
        return true;
    }

    uint32_t line  = pe->dst_base + line_stride(pe->stride >> 8) * (uint32_t) y;
    unsigned n_src = fmt_bytes(src_fmt(pe));
    if (!n_src)
        n_src = 1;
    uint32_t row       = pe->src_base + line_stride(pe->stride) * ((uint32_t) (pe->v >> 16) & pe->v_mask);
    int32_t  x0        = pe->x_shadow >> 16;
    bool     forward   = step == 0x10000;
    bool     backward  = step == (0U - 0x10000);
    bool     monotonic = (forward && (int64_t) x0 + (int64_t) n <= 32768)
                      || (backward && (int64_t) x0 - (int64_t) n >= -32769);
    if (!monotonic) {
        uint32_t x_shadow = (uint32_t) pe->x_shadow, u = (uint32_t) pe->u;
        bool     drew = false;
        for (unsigned i = 0; i < n; i++, x_shadow += step, u += step) {
            int32_t x = (int32_t) x_shadow >> 16;
            if (x < x_min || x >= x_end)
                continue;
            uint32_t addr = line + (uint32_t) x * bytes;
            uint32_t source;
            if (filled)
                source = pe->fg >> (8 * (4 - bytes - (addr & 3 & ~(bytes - 1))));
            else
                source = read_pixel(mem, row + (((int32_t) u >> 16) & pe->u_mask) * n_src, n_src);
            write_pixel(mem, addr, bytes, source & mask);
            verite_dirty_vram_page(c, (addr & mem.mask) >> 12);
            drew = true;
        }
        pe->x_shadow = (int32_t) final_x_shadow;
        pe->u        = (int32_t) final_u;
        if (drew)
            pe->pick_hit = c->drew = true;
        return true;
    }

    int32_t  x_lo;
    unsigned count;
    uint32_t u_cur;
    if (forward) {
        int32_t start_x = x0 > x_min ? x0 : x_min;
        int32_t stop_x  = (x0 + (int32_t) n) < x_end ? (x0 + (int32_t) n) : x_end;
        if (start_x >= stop_x) {
            pe->x_shadow = (int32_t) final_x_shadow;
            pe->u        = (int32_t) final_u;
            return true;
        }
        unsigned skip = (unsigned) (start_x - x0);
        count         = (unsigned) (stop_x - start_x);
        u_cur         = (uint32_t) pe->u + step * skip;
        x_lo          = start_x;
    } else {
        int32_t hi_x = x0 < x_end - 1 ? x0 : x_end - 1;
        int32_t lo_x = (x0 - (int32_t) n + 1) > x_min ? (x0 - (int32_t) n + 1) : x_min;
        if (lo_x > hi_x) {
            pe->x_shadow = (int32_t) final_x_shadow;
            pe->u        = (int32_t) final_u;
            return true;
        }
        unsigned skip = (unsigned) (x0 - hi_x);
        count         = (unsigned) (hi_x - lo_x + 1);
        u_cur         = (uint32_t) pe->u + step * skip;
        x_lo          = lo_x;
    }

    if (filled) {
        uint32_t addr = line + (uint32_t) x_lo * bytes;
        unsigned rem  = count;
        if ((mem.mask & 3) == 3) {
            while (rem && (addr & 3 & ~(bytes - 1))) {
                uint32_t source = pe->fg >> (8 * (4 - bytes - (addr & 3 & ~(bytes - 1))));
                write_pixel(mem, addr, bytes, source & mask);
                addr += bytes;
                rem--;
            }
            unsigned ppb   = 4 / bytes;
            unsigned words = rem / ppb;
            if (words) {
                uint32_t base = addr & ~3U;
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
                if ((size_t) (base & mem.mask) + (size_t) words * 4 <= mem.mask + 1) {
                    uint8_t *dst = mem.data + (base & mem.mask);
                    if (words >= 2) {
                        uint64_t fg64  = (uint64_t) pe->fg | ((uint64_t) pe->fg << 32);
                        unsigned pairs = words / 2;
                        for (unsigned p = 0; p < pairs; p++, dst += 8)
                            memcpy(dst, &fg64, sizeof(fg64));
                    }
                    if (words & 1)
                        memcpy(dst, &pe->fg, sizeof(pe->fg));
                } else
#endif
                {
                    for (unsigned w = 0; w < words; w++)
                        mem_write32(mem, base + 4 * w, pe->fg);
                }
                addr += words * 4;
                rem -= words * ppb;
            }
        }
        while (rem--) {
            uint32_t source = pe->fg >> (8 * (4 - bytes - (addr & 3 & ~(bytes - 1))));
            write_pixel(mem, addr, bytes, source & mask);
            addr += bytes;
        }
    } else {
        uint32_t u_at_lo   = forward ? u_cur : u_cur + step * (count - 1);
        uint32_t u_start   = (uint32_t) ((int32_t) u_at_lo >> 16) & pe->u_mask;
        bool     fast_blit = n_src == bytes
                      && (mem.mask & 3) == 3 && (mem.mask & (mem.mask + 1)) == 0
                      && (pe->u_mask & (pe->u_mask + 1)) == 0
                      && (uint64_t) u_start + count <= (uint64_t) pe->u_mask + 1;
        if (fast_blit) {
            uint32_t dst_lo     = line + (uint32_t) x_lo * bytes;
            uint32_t src_lo     = row + u_start * bytes;
            uint32_t phase_mask = bytes == 4 ? 0 : (bytes == 2 ? 2 : 3);
            if (((dst_lo ^ src_lo) & phase_mask) == 0) {
                size_t dst_off      = dst_lo & mem.mask;
                size_t src_off      = src_lo & mem.mask;
                size_t dst_last_off = (dst_lo + (count - 1) * bytes) & mem.mask;
                size_t src_last_off = (src_lo + (count - 1) * bytes) & mem.mask;
                size_t dst_first    = dst_off & ~(size_t) 3;
                size_t src_first    = src_off & ~(size_t) 3;
                size_t dst_phys_hi  = (dst_last_off & ~(size_t) 3) + 4;
                size_t src_phys_hi  = (src_last_off & ~(size_t) 3) + 4;
                bool   no_vram_wrap = dst_off + (size_t) count * bytes <= mem.mask + 1
                                   && src_off + (size_t) count * bytes <= mem.mask + 1
                                   && dst_first <= dst_last_off && src_first <= src_last_off;
                if (no_vram_wrap) {
                    bool disjoint = (dst_phys_hi <= src_first) || (src_phys_hi <= dst_first);
                    if (disjoint || (forward && dst_first < src_first)) {
                        uint32_t d_addr = dst_lo;
                        uint32_t s_addr = src_lo;
                        unsigned rem    = count;
                        while (rem && (d_addr & phase_mask)) {
                            write_pixel(mem, d_addr, bytes, read_pixel(mem, s_addr, bytes) & mask);
                            d_addr += bytes;
                            s_addr += bytes;
                            rem--;
                        }
                        unsigned ppb   = 4 / bytes;
                        unsigned words = rem / ppb;
                        if (words) {
                            size_t d_phys = (d_addr & mem.mask) & ~(size_t) 3;
                            size_t s_phys = (s_addr & mem.mask) & ~(size_t) 3;
                            if (disjoint) {
                                memcpy(mem.data + d_phys, mem.data + s_phys, (size_t) words * 4);
                            } else {
                                for (unsigned w = 0; w < words; w++)
                                    mem_write32(mem, (uint32_t) (d_phys + 4 * w), mem_read32(mem, (uint32_t) (s_phys + 4 * w)));
                            }
                            d_addr += words * 4;
                            s_addr += words * 4;
                            rem -= words * ppb;
                        }
                        while (rem--) {
                            write_pixel(mem, d_addr, bytes, read_pixel(mem, s_addr, bytes) & mask);
                            d_addr += bytes;
                            s_addr += bytes;
                        }
                        pe->x_shadow = (int32_t) final_x_shadow;
                        pe->u        = (int32_t) final_u;
                        dirty_span_bytes(c, mem, dst_lo, count * bytes);
                        pe->pick_hit = c->drew = true;
                        return true;
                    }
                    if (backward && dst_first > src_first) {
                        uint32_t d_addr = dst_lo + (count - 1) * bytes;
                        uint32_t s_addr = src_lo + (count - 1) * bytes;
                        unsigned rem    = count;
                        while (rem && ((d_addr + bytes) & phase_mask)) {
                            write_pixel(mem, d_addr, bytes, read_pixel(mem, s_addr, bytes) & mask);
                            d_addr -= bytes;
                            s_addr -= bytes;
                            rem--;
                        }
                        unsigned ppb   = 4 / bytes;
                        unsigned words = rem / ppb;
                        if (words) {
                            size_t d_phys = ((d_addr & mem.mask) & ~(size_t) 3) + 4;
                            size_t s_phys = ((s_addr & mem.mask) & ~(size_t) 3) + 4;
                            for (unsigned w = 1; w <= words; w++)
                                mem_write32(mem, (uint32_t) (d_phys - 4 * w), mem_read32(mem, (uint32_t) (s_phys - 4 * w)));
                            d_addr -= words * 4;
                            s_addr -= words * 4;
                            rem -= words * ppb;
                        }
                        while (rem--) {
                            write_pixel(mem, d_addr, bytes, read_pixel(mem, s_addr, bytes) & mask);
                            d_addr -= bytes;
                            s_addr -= bytes;
                        }
                        pe->x_shadow = (int32_t) final_x_shadow;
                        pe->u        = (int32_t) final_u;
                        dirty_span_bytes(c, mem, dst_lo, count * bytes);
                        pe->pick_hit = c->drew = true;
                        return true;
                    }
                }
            }
        }
        uint32_t addr      = line + (uint32_t) (forward ? x_lo : x_lo + (int32_t) count - 1) * bytes;
        int32_t  addr_step = forward ? (int32_t) bytes : -(int32_t) bytes;
        for (unsigned i = 0; i < count; i++, addr += (uint32_t) addr_step, u_cur += step) {
            uint32_t source = read_pixel(mem, row + (((int32_t) u_cur >> 16) & pe->u_mask) * n_src, n_src);
            write_pixel(mem, addr, bytes, source & mask);
        }
    }
    pe->x_shadow = (int32_t) final_x_shadow;
    pe->u        = (int32_t) final_u;
    dirty_span_bytes(c, mem, line + (uint32_t) x_lo * bytes, count * bytes);
    pe->pick_hit = c->drew = true;
    return true;
}

/* drawp stops at a word boundary. rdrawp continues until Cnt is exhausted. */
void
pixel_draw_parallel(verite_core *c, uint8_t op, uint8_t d, uint8_t s2, uint8_t s1, verite_memory mem)
{
    uint32_t      e2        = risc_src(c, s2 & ~1);
    uint32_t      o2        = risc_src(c, s2 | 1);
    uint32_t      e1        = risc_src(c, s1 & ~1);
    uint32_t      o1        = risc_src(c, s1 | 1);
    pixel_engine *pe        = &c->pe;
    bool          resetting = !!(op & 4), repeated = !!(op & 1);
    if (resetting)
        pe->y = (int32_t) o2;
    else
        pixel_set(pe, P_U, o2);

    unsigned cnt  = pe->cnt & 0xffff;
    int32_t  x0   = pe->x_shadow;
    uint32_t step = pe->x_dir ? 0U - 0x10000 : 0x10000;
    unsigned n    = cnt;
    if (!repeated && !resetting) {
        uint32_t addr  = dst_at(pe, pe->x_shadow >> 16, pe->y >> 16);
        unsigned bytes = dst_bytes(pe);
        unsigned left  = pe->x_dir ? (addr & 3) / bytes + 1 : (4 - (addr & 3)) / bytes;
        if (left < n)
            n = left;
    }
    if (!pixel_parallel_copy(c, n, step, mem)) {
        for (unsigned i = 0; i < n; i++) {
            pixel_parallel(c, pe->x_shadow >> 16, pe->y >> 16, pe->u >> 16, pe->v >> 16, mem);
            pe->x_shadow = (int32_t) ((uint32_t) pe->x_shadow + step);
            pe->u        = (int32_t) ((uint32_t) pe->u + step);
        }
    }
    pe->cnt = resetting ? cnt : cnt - n;
    if (resetting)
        pe->x_shadow = x0;
    if (resetting || (!repeated && cnt)) {
        risc_dst(c, d & ~1, e2 + e1);
        risc_dst(c, d | 1, o2 + o1);
    }
}

static inline bool
depth_pass(unsigned mode, uint32_t z, verite_memory mem, uint32_t addr)
{
    if (!mode)
        return true;
    if (mode == 7)
        return false;
    uint32_t old = mem_read16(mem, addr);
    switch (mode) {
        case 1:
            return z < old;
        case 2:
            return z == old;
        case 3:
            return z <= old;
        case 4:
            return z > old;
        case 5:
            return z != old;
        default:
            return z >= old;
    }
}

typedef struct {
    int32_t     x_min, y_min;
    uint32_t    x_len, y_len;
    uint32_t    dst_base, z_base, dst_stride, z_stride, mask, pmask, dst_mode, src_mode;
    unsigned    format, bytes, rop_code, texture_func, depth_mode;
    int32_t     alpha_thres, yuv_black, yuv_scale;
    color       fog_color, dst_color;
    tex_sampler tex;
    bool        perspective, bgr, blend, read_disabled, read_dst, copy, pat_enabled;
} pixel_span;

static void
pixel_span_init(pixel_span *span, const pixel_engine *pe, bool perspective)
{
    int32_t x_min = (int32_t) ((pe->scissor_x >> 16) & 0xfff);
    int32_t x_end = (int32_t) (pe->scissor_x & 0xfff);
    int32_t y_min = (int32_t) ((pe->scissor_y >> 16) & 0xfff);
    int32_t y_end = (int32_t) (pe->scissor_y & 0xfff);
    if (x_end > 2048)
        x_end = 2048;
    if (y_end > 2048)
        y_end = 2048;

    uint32_t dst_mode      = pe->dst_mode;
    uint32_t src_mode      = pe->src_mode;
    unsigned bytes         = dst_bytes(pe);
    uint32_t mask          = pixel_mask(bytes);
    uint32_t pmask         = pe->pmask & mask;
    unsigned rop_code      = dst_mode & 15;
    unsigned texture_func  = (src_mode >> 4) & 7;
    unsigned depth_mode    = (dst_mode >> 12) & 7;
    bool     blend         = !!(dst_mode & DST_BLEND);
    bool     read_disabled = !!(dst_mode & DST_READ_DISABLE);

    span->x_min         = x_min;
    span->y_min         = y_min;
    span->x_len         = x_end > x_min ? (uint32_t) (x_end - x_min) : 0;
    span->y_len         = y_end > y_min ? (uint32_t) (y_end - y_min) : 0;
    span->dst_base      = pe->dst_base;
    span->dst_stride    = line_stride(pe->stride >> 8);
    span->mask          = mask;
    span->pmask         = pmask;
    span->dst_mode      = dst_mode;
    span->src_mode      = src_mode;
    span->format        = pe->dst_fmt & 15;
    span->bytes         = bytes;
    span->rop_code      = rop_code;
    span->texture_func  = texture_func;
    span->depth_mode    = depth_mode;
    span->perspective   = perspective;
    span->bgr           = !!(dst_mode & DST_BGR);
    span->blend         = blend;
    span->read_disabled = read_disabled;
    span->pat_enabled   = !!(pe->pat_mode & 0x10000);
    span->copy          = rop_code == 12 && pmask == mask;
    span->read_dst      = !read_disabled && (blend || rop_reads(rop_code) || pmask != mask);

    if (depth_mode || (dst_mode & DST_ZBUF_WRITE)) {
        span->z_base   = pe->z_base;
        span->z_stride = line_stride(pe->stride >> 16);
    }
    if (texture_func)
        span->tex = tex_sampler_init(pe);
    if (dst_mode & DST_TRANSP_REJECT)
        span->alpha_thres = (int32_t) (pe->alpha_thres & 255);
    if (dst_mode & DST_FOG)
        span->fog_color = argb(pe->fog_color);
    if (blend && read_disabled)
        span->dst_color = argb(pe->dst_color);
    else if (!blend && (dst_mode & DST_YUV2RGB)) {
        span->yuv_black = (int32_t) ((pe->fog_color >> 8) & 255);
        span->yuv_scale = (int32_t) (pe->yuv_contrast & 255) + 256;
    }
}

static int32_t
perspective_coord(int32_t coord, int64_t q, double scale)
{
    int64_t num = (int64_t) coord * (INT64_C(1) << 24);
    int64_t estimate = (int64_t) (coord * scale);
    int64_t remainder = num - estimate * q;
    if (num < 0)
        estimate += (remainder > 0) - (remainder <= -q);
    else
        estimate += (remainder >= q) - (remainder < 0);
    return (int32_t) estimate;
}

static void
perspective_uv(int32_t *u, int32_t *v, uint32_t q)
{
    /* For q >= 64 the quotient magnitude is at most 2^49. The two
       double-precision operations miss by less than one, so one
       remainder correction gives the exact truncating quotient. */
    if (q >= 64) {
        double scale = (double) (1 << 24) / q;
        *u = perspective_coord(*u, q, scale);
        *v = perspective_coord(*v, q, scale);
    } else {
        *u = (int32_t) (((int64_t) *u * (INT64_C(1) << 24)) / q);
        *v = (int32_t) (((int64_t) *v * (INT64_C(1) << 24)) / q);
    }
}

static void
pixel_full(verite_core *core, int32_t x, int32_t y, const pixel_span *span, verite_memory mem)
{
    static const unsigned dither_matrix[4][4] = {
        { 0,  12, 3,  15 },
        { 7,  11, 4,  8  },
        { 13, 1,  14, 2  },
        { 10, 6,  9,  5  }
    };
    pixel_engine *pe         = &core->pe;
    bool          background = false;
    bool          rejected   = span->pat_enabled && !pattern_pixel(pe, &background);
    bool          outside    = ((uint32_t) x - (uint32_t) span->x_min >= span->x_len)
                             | ((uint32_t) y - (uint32_t) span->y_min >= span->y_len)
                             | (pe->z_ovf && (span->src_mode & SRC_Z_SCISSOR));
    if (rejected | outside)
        return;

    uint32_t dst_mode = span->dst_mode;
    uint32_t z_addr = 0, z = 0;
    if (span->depth_mode || (dst_mode & DST_ZBUF_WRITE)) {
        z_addr = span->z_base + span->z_stride * (uint32_t) y + 2 * (uint32_t) x;
        z      = (uint32_t) (pe->z >> 11) & 0xffff;
    }
    /* A rejected pixel cannot modify either buffer. Test depth before sampling. */
    if (span->depth_mode && !depth_pass(span->depth_mode, z, mem, z_addr))
        return;

    /* The parameter setters already keep the saturated channels in FG. */
    unsigned func = span->texture_func;
    color    c;
    if (func) {
        int32_t u = pe->u, v = pe->v;
        if (span->perspective && pe->q)
            perspective_uv(&u, &v, pe->q);
        color t = sample_with(pe, &span->tex, u, v, mem);
        if (func == 2) {
            c = argb(background ? pe->bg : pe->fg);
            for (unsigned k = 0; k < 3; k++)
                c.channel[k] = div255(c.channel[k] * (255 - t.channel[3]) + t.channel[k] * t.channel[3] + 127);
        } else if (func == 3) {
            c = argb(background ? pe->bg : pe->fg);
            for (unsigned k = 0; k < 4; k++)
                c.channel[k] = div255(c.channel[k] * t.channel[k] + 127);
        } else
            c = t;
    } else {
        c = argb(background ? pe->bg : pe->fg);
    }
    if ((dst_mode & DST_TRANSP_REJECT) && c.channel[3] <= span->alpha_thres)
        return;
    if (dst_mode & DST_FOG) {
        int32_t f = sat(pe->f);
        for (unsigned k = 0; k < 3; k++)
            c.channel[k] = f * (c.channel[k] - span->fog_color.channel[k]) / 255 + span->fog_color.channel[k];
    }
    uint32_t addr = span->dst_base + span->dst_stride * (uint32_t) y + (uint32_t) x * span->bytes;
    uint32_t old  = span->read_dst ? read_pixel(mem, addr, span->bytes) : 0;
    if (span->blend) {
        color d = span->read_disabled ? span->dst_color : decode(span->format, old, span->bgr, !!(dst_mode & DST_NO_PAD));
        c       = blend_with((dst_mode >> 4) & 15, (dst_mode >> 8) & 15, c, d);
    } else if (dst_mode & DST_YUV2RGB)
        c = yuv_to_rgb_with(span->yuv_black, span->yuv_scale, c);

    unsigned dither = 0;
    if (dst_mode & DST_DITHER) {
        unsigned dx = ((uint32_t) x + (pe->dither_offset & 3)) & 3;
        unsigned dy = ((uint32_t) y + ((pe->dither_offset >> 4) & 3)) & 3;
        dither      = dither_matrix[dy][dx];
    }
    uint32_t encoded = encode(span->format, c, dither, span->bgr);
    uint32_t out     = span->copy ? encoded : (rop(span->rop_code, encoded, old) & span->pmask) | (old & ~span->pmask);
    write_pixel(mem, addr, span->bytes, out & span->mask);
    verite_dirty_vram_page(core, (addr & mem.mask) >> 12);
    if (dst_mode & DST_ZBUF_WRITE) {
        mem_write16(mem, z_addr, z);
        verite_dirty_vram_page(core, (z_addr & mem.mask) >> 12);
    }
    pe->pick_hit = core->drew = true;
}

/* Repeated Q/U, Z/V draws step shadow X and latch Q after each pixel. */
static void
pixel_draw_perspective_span(verite_core *c, const pixel_span *span, const uint32_t values[4],
                            const uint32_t deltas[4], int count, uint32_t x_step, verite_memory mem)
{
    pixel_engine *pe        = &c->pe;
    bool          swap      = !!(pe->src_mode & SRC_SWAP_UV);
    bool          z_scissor = !!(pe->src_mode & SRC_Z_SCISSOR);
    unsigned      u_index   = swap ? 3 : 1;
    unsigned      v_index   = swap ? 1 : 3;
    uint32_t      q         = values[0], dq = deltas[0];
    uint32_t      u         = values[u_index], du = deltas[u_index];
    uint32_t      z         = values[2], dz = deltas[2];
    uint32_t      v         = values[v_index], dv = deltas[v_index];
    pe->u_mask = pe->src_mask & 0x7ff;
    pe->v_mask = (pe->src_mask >> 16) & 0x7ff;

    for (int i = 0; i < count; i++) {
        pe->u     = (int32_t) u;
        pe->v     = (int32_t) v;
        pe->z     = (int32_t) z;
        pe->z_ovf = z_scissor && (z & 0xf8000000);
        pixel_full(c, pe->x_shadow >> 16, pe->y >> 16, span, mem);
        pe->q        = q;
        pe->x_shadow = (int32_t) ((uint32_t) pe->x_shadow + x_step);
        q += dq;
        u += du;
        z += dz;
        v += dv;
    }
    pe->cnt = 0;
}

/* Unpadded RGB565 expands to multiples of its quantization step. The dither
   offset is smaller than that step, so replace-mode draws can copy the texel. */
static void
pixel_copy_565(verite_core *c, uint8_t d, uint8_t s2, uint8_t s1, verite_memory mem)
{
    uint32_t x  = risc_src(c, s2 & ~1);
    uint32_t u  = risc_src(c, s2 | 1);
    uint32_t dx = risc_src(c, s1 & ~1);
    uint32_t du = risc_src(c, s1 | 1);
    pixel_engine *pe = &c->pe;
    int cnt = (int16_t) pe->cnt;
    if (cnt <= 0)
        return;

    pe->x = (int32_t) x;
    pe->u = (int32_t) u;
    pe->u_mask = pe->src_mask & 0x7ff;
    int32_t px = pe->x >> 16, py = pe->y >> 16;
    if (inside(pe, px, py)) {
        uint32_t tx = (u >> 16) & pe->u_mask;
        uint32_t ty = (uint32_t) (pe->v >> 16) & pe->v_mask;
        uint32_t texel = mem_read16(mem, pe->src_base + line_stride(pe->stride) * ty + 2 * tx);
        bool transparent = keyed(pe, texel, 2);
        if (!(pe->dst_mode & DST_TRANSP_REJECT) || (!transparent && (pe->alpha_thres & 255) < 255)) {
            uint32_t addr = pe->dst_base + line_stride(pe->stride >> 8) * (uint32_t) py + 2 * (uint32_t) px;
            mem_write16(mem, addr, transparent ? 0 : texel);
            verite_dirty_vram_page(c, (addr & mem.mask) >> 12);
            pe->pick_hit = c->drew = true;
        }
    }
    pe->cnt = cnt - 1;
    risc_dst(c, d & ~1, x + dx);
    risc_dst(c, d | 1, u + du);
}

/* Parameterized draws take the group of values before the group of deltas. */
void
pixel_draw_param(verite_core *c, uint8_t op, uint8_t d, uint8_t s2, uint8_t s1, verite_memory mem)
{
    pixel_engine *pe  = &c->pe;
    uint32_t      ctl = pe->drawctl;
    unsigned      count;
    if (op == 0xf0 && (ctl & 15) == 4 && !(pe->pat_mode & 0x10000)
        && (pe->dst_fmt & 15) == 4 && src_fmt(pe) == 4
        && (pe->src_mode & ~(15U | SRC_CHROMA)) == (0x10 | SRC_NO_PAD)
        && (pe->dst_mode & ~(0xff0U | DST_DITHER | DST_NO_PAD | DST_READ_DISABLE | DST_TRANSP_REJECT)) == 12
        && (pe->pmask & 0xffff) == 0xffff) {
        pixel_copy_565(c, d, s2, s1, mem);
        return;
    }
    switch (op & 0xfe) {
        case 0xf0:
            count = 2;
            break;
        case 0xf2:
            count = 4;
            break;
        case 0xf8:
            count = 6;
            break;
        case 0xfa:
            count = 8;
            break;
        default:
            count = 10;
            break;
    }
    first_pair  p1 = pair1[ctl & 15];
    unsigned    i2 = (ctl >> 4) & 31;
    param_pair  p2 = i2 <= 7 ? pair3[i2] : pair2[i2 - 8];
    param_pair  p3 = pair3[(ctl >> 9) & 15], p4 = pair3[(ctl >> 13) & 7];
    pixel_param params[] = { p1.pair.even, p1.pair.odd, p2.even, p2.odd, p3.even, p3.odd, p4.even, p4.odd, P_G, P_B };
    unsigned    align    = 2;
    while (align < count)
        align *= 2;
    d &= ~(align - 1);
    s2 &= ~(align - 1);
    s1 &= ~(align - 1);
    uint32_t values[10], deltas[10];
    for (unsigned i = 0; i < count; i++)
        values[i] = risc_src(c, s2 + i);
    for (unsigned i = 0; i < count; i++)
        deltas[i] = risc_src(c, s1 + i);
    int cnt = (int16_t) pe->cnt;
    if (cnt <= 0)
        return;
    /* A native span for this render state does the whole draw. */
    if (pixel_jit_span(c, op, values, deltas, mem)) {
        if (!(op & 1))
            for (unsigned k = 0; k < count; k++)
                risc_dst(c, d + k, values[k] + deltas[k]);
        return;
    }

    bool     shadow = p1.shadow;
    int      q_index = -1;
    unsigned active[10], active_count = 0;
    for (unsigned i = 0; i < count; i++) {
        if (params[i] == P_Q)
            q_index = (int) i;
        else if (params[i] != P_NONE)
            active[active_count++] = i;
        if (params[i] == P_X)
            shadow = false;
    }

    pixel_span span;
    pixel_span_init(&span, pe, q_index >= 0);

    uint32_t step     = pe->x_dir ? 0U - 0x10000 : 0x10000;
    if (op == 0xf3 && params[0] == P_Q && params[1] == P_U && params[2] == P_Z && params[3] == P_V) {
        pixel_draw_perspective_span(c, &span, values, deltas, cnt, step, mem);
        return;
    }
    bool     repeated = !!(op & 1);
    int      n        = repeated ? cnt : 1;
    for (int i = 0; i < n; i++) {
        for (unsigned k = 0; k < active_count; k++) {
            unsigned param = active[k];
            pixel_set(pe, params[param], values[param]);
        }
        int32_t x = (shadow ? pe->x_shadow : pe->x) >> 16;
        pixel_full(c, x, pe->y >> 16, &span, mem);
        /* Perspective uses the previous Q. Commit the last Q operand after drawing. */
        if (q_index >= 0)
            pe->q = values[q_index];
        if (shadow)
            pe->x_shadow = (int32_t) ((uint32_t) pe->x_shadow + step);
        if (repeated && i + 1 < n)
            for (unsigned k = 0; k < count; k++)
                values[k] += deltas[k];
    }
    pe->cnt = repeated ? 0 : cnt - 1;
    if (!repeated)
        for (unsigned k = 0; k < count; k++)
            risc_dst(c, d + k, values[k] + deltas[k]);
}
