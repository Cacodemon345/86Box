/* SPDX-License-Identifier: GPL-2.0-or-later */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdatomic.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#    include <immintrin.h>
#    define verite_pause() _mm_pause()
#else
#    define verite_pause() ((void) 0)
#endif

#include <86box/86box.h>
#include <86box/device.h>
#include <86box/dma.h>
#include <86box/i2c.h>
#include <86box/io.h>
#include <86box/mem.h>
#include <86box/pci.h>
#include <86box/rom.h>
#include <86box/thread.h>
#include <86box/timer.h>
#include <86box/video.h>
#include <86box/vid_ddc.h>
#include <86box/vid_svga.h>
#include <86box/vid_vga.h>

#include "verite/core.h"

#define VERITE_ROM "roms/video/verite/phx49.rom"
#define ROM_SIZE   0x8000
#define VRAM_SIZE  0x400000
#define BAR_SIZE   0x1000000
#define TICK_US    50
#define MODE_VESA            0x01
#define MODE_VGA             0x02
#define CRTC_CTL_DDC_DRIVE   0x20
#define CRTC_CTL_DDC_SCL     0x40
#define CRTC_CTL_DDC_SDA     0x80
#define CRTC_CTL_VSYNC       0x400
#define CRTC_CTL_VIDEO       0x1000
#define CRTC_CTL_STEREO      0x2000
#define CRTC_CTL_DISPLAYED   0x4000
#define CRTC_CTL_BGR         0x8000
#define CRTC_CTL_EVEN_FRAME  0x10000
#define CRTC_CTL_LINE_DOUBLE 0x20000
#define CRTC_CTL_SWITCHED    0x40000
#define CRTC_CTL_FRAME_FLAGS (CRTC_CTL_DISPLAYED | CRTC_CTL_EVEN_FRAME | CRTC_CTL_SWITCHED)
#define CRTC_TEST_VBLANK     0x40000
#define CRTC_TEST_NOT_VBLANK 0x10000
#define DRAM_SLOW_PRECHARGE  0x0010
#define DRAM_REFRESH_VBLANK  0x2000
#define DRAM_REFRESH_DISABLE 0x8000
#define DRAM_REFRESH_COUNT   0xff0000
/* Approximate EDO refresh costs in the 25 MHz RISC clock domain. */
#define DRAM_REFRESH_UNIT_NS 320
#define DRAM_REFRESH_CYCLES  2
#define PLL_VGA_CLOCK        0x40000
#define PLL_WORD_MASK        0xfffff
#define PLL_REF_HZ           14318180.0

enum {
    CRTC_TEST   = 0x80,
    CRTC_CTL    = 0x84,
    CRTC_HORZ   = 0x88,
    CRTC_VERT   = 0x8c,
    FRAME_BASE_B = 0x90,
    FRAME_BASE_A = 0x94,
    CRTC_OFFSET = 0x98,
    CRTC_STATUS = 0x9c,
    DRAM_CTL    = 0xa0,
    RAMDAC      = 0xb0,
    PLL_DEVICE  = 0xc0
};

/* Threaded RISC. The core belongs to whichever thread holds owner. The CPU
   thread queues data-FIFO writes and retrace events for the worker, answers
   status reads from the registers the worker last published, and takes the
   core itself for other reads and register writes. The worker posts its board
   writes to the CPU thread and asks it for board state the CPU thread owns. */
#define QUEUE_SIZE  8192
#define POSTED_SIZE 1024
#define RUN_AHEAD   25000 /* RISC cycles the worker may run past guest time */
#define SYNC_AHEAD  2000  /* the same for a catch-up to a queued entry */
#define SLICE       4000  /* RISC cycles the worker runs per hold of the core */
#define SYNC_FREE   64    /* queue space left before FIFO space reads sync */
#define HIST_SIZE   16384 /* DMA register states kept for the guest's reads */
#define HIST_NS     1000  /* DMA progress closer than this shares an entry */
#define IDLE_SPIN_US 100  /* host time the worker waits for work before sleeping */
#define VIO_MEM_ENDIAN_REG 0x43 /* its top bit reads the DMA busy */
#define DMA_REGS           0x50 /* DMA command pointer, address and length */
#define DMA_REGS_END       0x5c

enum { OWN_NONE, OWN_WORKER, OWN_CPU };
enum { REQ_IDLE, REQ_READ, REQ_FLUSH, REQ_DMA, REQ_DONE };
enum { Q_WRITE, Q_RETRACE };

typedef struct {
    uint64_t t;
    uint32_t value;
    uint8_t  kind, reg, len;
} verite_entry;

typedef struct {
    thread_t    *thread;
    event_t     *wake, *done;
    atomic_int   owner, cpu_want, stop, sleeping;
    atomic_int   irq, drew, dma_parked;
    atomic_uint  head, tail;
    atomic_ullong now_ns, dma_end_ns;
    atomic_uint  refresh;
    /* The board request. */
    atomic_int   req;
    uint8_t      req_reg, req_len;
    uint32_t     req_value, req_count;
    uint32_t    *req_words;
    /* The worker's own. */
    uint64_t     slept_ns;
    atomic_int   wait_input;
    /* Board writes from the RISC, applied in order on the CPU thread. */
    struct {
        uint32_t value;
        uint8_t  reg, len;
    } posted[POSTED_SIZE];
    atomic_uint  posted_head, posted_tail;
    /* Host registers 40h-5Fh as of the last time the core changed hands. */
    atomic_uint  mirror_seq;
    uint8_t      mirror[32];
    unsigned     mirror_head; /* the queue head the mirror reflects */
    uint64_t     mirror_ns;   /* the RISC's time when it was published */
    /* The DMA registers 50h-5Bh as they change, in the guest's time, for
       reads of a DMA the worker has run past. Written by the core's owner
       and read by the CPU thread, which alone moves the tail back. */
    struct {
        uint64_t ns;
        uint32_t regs[3];
    } hist[HIST_SIZE];
    atomic_uint  hist_head, hist_tail;
    verite_entry queue[QUEUE_SIZE];
} verite_worker;

typedef struct {
    vga_t         vga;
    rom_t         bios;
    mem_mapping_t linear;
    mem_mapping_t banked;
    pc_timer_t    timer;
    verite_core  *core;
    void         *ramdac;
    void         *ddc_bus;
    void         *ddc_monitor;
    uint8_t       ddc1_edid[128];
    unsigned      ddc1_bit;
    int           ddc2_active;
    uint8_t       regs[256];
    uint8_t       pci_slot, irq_state, irq_line, irq_asserted, command;
    uint8_t       bar_hi, io_hi, io_installed, endian, pci_control, mode, dac_command0;
    uint32_t      rom_bar, bank, frame_base_b, frame[2], pitch[2];
    unsigned      scanout_offset;
    int           native_bpp;
    uint64_t      time_ns;
    uint64_t      dram_last_refresh_ns;
    unsigned      dram_refresh_pending;
    uint32_t      frame_flags;
    uint32_t      pll_shift;
    double        pixel_clock_hz;
    uint32_t     *word_colors; /* 16-bit scanout word to output colour */
    uint32_t      word_colors_key;
    int           word_colors_valid;
    verite_worker *w;
} verite_t;

static const uint16_t dac_ports[] = { 0x3c8, 0x3c9, 0x3c6, 0x3c7 };

static const video_timings_t verite_timing = { VIDEO_PCI, 2, 2, 1, 20, 20, 21 };

static uint32_t
verite_reg(const verite_t *v, uint8_t reg)
{
    return v->regs[reg] | ((uint32_t) v->regs[reg + 1] << 8) | ((uint32_t) v->regs[reg + 2] << 16) | ((uint32_t) v->regs[reg + 3] << 24);
}

/* Builds the map from a 16-bit scanout word to the RAMDAC's output colour.
   Palette and format changes mark it stale. */
static const uint32_t *
verite_word_colors(verite_t *v, uint32_t ctl)
{
    svga_t  *svga       = &v->vga.svga;
    int      rgb565     = bt48x_rgb565(v->ramdac);
    int      contiguous = bt48x_clut_contiguous(v->ramdac);
    int      bgr        = !!(ctl & CRTC_CTL_BGR);
    uint32_t key        = v->native_bpp | (rgb565 << 8) | (contiguous << 9) | (svga->lut_map << 10) | (bgr << 11)
                   | (svga->ramdac_type << 12);

    if (v->word_colors_valid && v->word_colors_key == key)
        return v->word_colors;

    /* The CRTC's 1555 format holds 5:5:5 colour with bit 15 unused, whatever
       layout the RAMDAC is in. The 16-bit format passes words through in the
       RAMDAC's RGB layout. */
    int             src565 = v->native_bpp == 16 && rgb565;
    const uint32_t *colors = src565 ? video_16to32 : video_15to32;
    /* The Bt485 routes the 5/6-bit components to either end of the palette
       address, filling the unused bits with zero. 1555 pixels reach a 5:6:5
       RAMDAC with their 5-bit green in the low bits of its 6-bit field;
       Scorched Planet's identity palette puts the green ramp there. */
    unsigned green_shift      = src565 ? 2 : 3;
    unsigned rb_address_shift = contiguous ? 0 : 3;
    unsigned g_address_shift  = contiguous ? 0 : (rgb565 ? 2 : 3);

    for (uint32_t c = 0; c < 65536; c++) {
        uint32_t p = colors[c];
        if (bgr)
            p = makecol32(getcolb(p), getcolg(p), getcolr(p));
        if (svga->lut_map) {
            unsigned r = (getcolr(p) >> 3) << rb_address_shift;
            unsigned g = (getcolg(p) >> green_shift) << g_address_shift;
            unsigned b = (getcolb(p) >> 3) << rb_address_shift;
            p          = svga_lookup_lut_ram(svga, makecol32(r, g, b));
        }
        v->word_colors[c] = p;
    }
    v->word_colors_key   = key;
    v->word_colors_valid = 1;
    return v->word_colors;
}

static void
verite_render(svga_t *svga)
{
    verite_t *v = svga->priv;
    int       y = svga->displine + svga->y_add;
    if (y < 0 || y >= 2048 || !svga->monitor->target_buffer)
        return;

    uint32_t       ctl   = verite_reg(v, CRTC_CTL);
    /* Stereoscopic scanout uses A on even frames and B on odd frames. */
    unsigned       eye   = (ctl & CRTC_CTL_STEREO) && !(v->frame_flags & CRTC_CTL_EVEN_FRAME);
    unsigned       row   = svga->displine >> !!(ctl & CRTC_CTL_LINE_DOUBLE);
    uint32_t      *pixel = &svga->monitor->target_buffer->line[y][svga->x_add];
    uint32_t       addr  = (v->frame[eye] + v->pitch[eye] * row) & (VRAM_SIZE - 1);
    const uint8_t *ram   = svga->vram;
    int            width = svga->hdisp / svga->dots_per_clock;

    if (svga->firstline_draw == 2000)
        svga->firstline_draw = svga->displine;
    svga->lastline_draw = svga->displine;

    if (v->native_bpp == 15 || v->native_bpp == 16) {
        const uint32_t *colors = verite_word_colors(v, ctl);
        if (!(addr & 3) && !(width & 1) && 2U * width <= VRAM_SIZE - addr) {
            /* VRAM stores the two 16-bit pixels in each dword in reverse order. */
            for (int x = 0; x < width; x += 2) {
                const uint8_t *pair = ram + addr + 2 * x;
                pixel[x]     = colors[pair[2] | (pair[3] << 8)];
                pixel[x + 1] = colors[pair[0] | (pair[1] << 8)];
            }
        } else {
            for (int x = 0; x < width; x++, addr += 2) {
                uint32_t a = addr & (VRAM_SIZE - 1);
                pixel[x]   = colors[ram[a ^ 2] | (ram[((a + 1) & (VRAM_SIZE - 1)) ^ 2] << 8)];
            }
        }
    } else if (v->native_bpp == 8) {
        for (int x = 0; x < width; x++, addr++)
            pixel[x] = svga->map8[ram[(addr & (VRAM_SIZE - 1)) ^ 3] & svga->dac_mask];
    } else {
        for (int x = 0; x < width; x++, addr += 4) {
            uint32_t a = addr & (VRAM_SIZE - 1);
            uint32_t p = ((uint32_t) ram[(a + 2) & (VRAM_SIZE - 1)] << 16) | ((uint32_t) ram[(a + 1) & (VRAM_SIZE - 1)] << 8) | ram[a];
            if (ctl & CRTC_CTL_BGR)
                p = makecol32(getcolb(p), getcolg(p), getcolr(p));
            pixel[x] = svga->lut_map ? svga_lookup_lut_ram(svga, p) : p;
        }
    }

    if (svga->dots_per_clock == 2) {
        /* Expand from right to left so the source pixels are still available. */
        for (int x = width - 1; x >= 0; x--)
            pixel[2 * x] = pixel[2 * x + 1] = pixel[x];
    }
}

static void
verite_update_pitch(verite_t *v, unsigned width, unsigned bpp, uint32_t ctl)
{
    unsigned bytes = width * ((bpp + 7) / 8);
    unsigned fifo  = (ctl & 0x10) ? 128 : 64;

    for (unsigned eye = 0; eye < 2; eye++) {
        unsigned pitch = (bytes - !(v->frame[eye] & 7)) / fifo * fifo + v->scanout_offset;
        v->pitch[eye]  = MAX(pitch, bytes);
    }
}

static void
verite_recalctimings(svga_t *svga)
{
    verite_t *v   = svga->priv;
    uint32_t  ctl = verite_reg(v, CRTC_CTL);
    int       native = (ctl & CRTC_CTL_VIDEO) && !(v->mode & MODE_VGA);

    svga_set_ramdac_type(svga, native && (v->dac_command0 & 2) ? RAMDAC_8BIT : RAMDAC_6BIT);
    svga->dac_hwcursor_draw = native ? bt48x_hwcursor_draw : NULL;
    svga->lut_map           = 0;
    svga->hoverride         = 0;
    if (!native)
        return;

    int bpp;
    switch (ctl & 15) {
        case 1:
        case 2:
            bpp = 8;
            break;
        case 4:
            bpp = 16;
            break;
        case 6:
            bpp = 15;
            break;
        case 12:
            bpp = 32;
            break;
        default:
            return;
    }

    uint32_t horz  = verite_reg(v, CRTC_HORZ);
    uint32_t vert  = verite_reg(v, CRTC_VERT);
    int      width = ((horz & 0xff) + 1) * 8;
    int      total = width + 8 * (3 + ((horz >> 9) & 63) + ((horz >> 16) & 31) + ((horz >> 21) & 7));
    int      lines = (vert & 0x7ff) + 1;
    int      front = ((vert >> 20) & 63) + 1;
    int      sync  = ((vert >> 17) & 7) + 1;
    int      back  = ((vert >> 11) & 63) + 1;

    verite_update_pitch(v, width, bpp, ctl);

    v->native_bpp                   = bpp;
    svga->lut_map                  = bpp > 8 && !bt48x_clut_bypass(v->ramdac);
    svga->render                   = verite_render;
    svga->hoverride                = 1;
    /* Low-resolution scanout repeats rows and needs matching horizontal
       expansion in the host buffer. Keep CRTC timing and pitch in source units. */
    svga->dots_per_clock           = (ctl & CRTC_CTL_LINE_DOUBLE) && width <= 1024 ? 2 : 1;
    svga->hdisp                   = width * svga->dots_per_clock;
    svga->hdisp_time              = width;
    svga->htotal                   = total;
    svga->hblankstart              = width - 1;
    svga->hblank_end_val           = total - 1;
    svga->hblank_end_mask          = 0xfff;
    svga->char_width               = 1;
    svga->dispend = svga->vblankstart = lines;
    svga->vsyncstart                  = lines + front;
    svga->vtotal                      = lines + front + sync + back;
    /* Without a programmed PLL clock, use the VGA clock selected above. */
    if (v->pixel_clock_hz)
        svga->clock = (cpuclock * (double) (1ULL << 32)) / v->pixel_clock_hz;
    if (bt48x_clock_doubled(v->ramdac))
        svga->clock *= 0.5;
    /* Cancel the VGA sequencer's dot clock divide when using native timings. */
    if (svga->seqregs[1] & 8)
        svga->clock *= 0.5;
    /* Native vertical timing counts field lines. The SVGA poller weaves them. */
    svga->interlace                = bt48x_interlaced(v->ramdac);
    svga->linedbl = svga->rowcount = 0;
    svga->split                   = 0x1000;
    svga->monitor->mon_overscan_x = svga->monitor->mon_overscan_y = 0;
}

static void
verite_redraw(void *priv)
{
    verite_t *v            = priv;
    v->vga.svga.fullchange = changeframecount;
}

static uint32_t
verite_crtc_status(verite_t *v)
{
    svga_t *svga   = &v->vga.svga;
    int     native = svga->render == verite_render;
    int     line   = svga->vc;
    int     phase, end;

    if (line < svga->dispend) {
        phase = 0;
        end   = svga->dispend;
    } else if (line < svga->vsyncstart) {
        phase = 1;
        end   = svga->vsyncstart;
    } else {
        int sync_lines = native ? ((verite_reg(v, CRTC_VERT) >> 17) & 7) + 1
                                : ((svga->crtc[0x11] - svga->vsyncstart - 1) & 15) + 1;
        end   = svga->vsyncstart + sync_lines;
        phase = 3;
        if (line >= end) {
            phase = 2;
            end   = svga->vtotal;
        }
    }

    unsigned horizontal = !!svga->linepos;
    uint64_t remaining  = timer_get_remaining_u64(&svga->timer);
    uint64_t duration   = horizontal ? svga->dispofftime : svga->dispontime;
    unsigned clocks     = horizontal ? MAX(0, svga->htotal - svga->hdisp_time) : svga->hdisp_time;
    if (native)
        clocks /= 8;
    if (duration) {
        remaining = MIN(remaining, duration);
        clocks = (unsigned) (((uint128_t) remaining * clocks + duration - 1) / duration);
    }
    if (horizontal && native) {
        uint32_t horz      = verite_reg(v, CRTC_HORZ);
        unsigned back      = ((horz >> 9) & 63) + 1;
        unsigned sync      = ((horz >> 16) & 31) + 1;
        if (clocks <= back)
            horizontal = 2;
        else if (clocks <= back + sync)
            horizontal = 3;
        if (horizontal == 1)
            clocks -= back + sync;
        else if (horizontal == 3)
            clocks -= back;
    }

    /* Phase and clocks remaining in each direction. */
    unsigned lines_left = end > line ? end - line : 0;
    return ((uint32_t) phase << 22) | (MIN(lines_left, 0x7ffU) << 11)
           | (horizontal << 9) | MIN(clocks, 0xffU);
}

static int
verite_ddc_data(verite_t *v)
{
    uint8_t ctl = v->regs[CRTC_CTL];
    if (!(ctl & CRTC_CTL_DDC_DRIVE))
        return i2c_gpio_get_sda(v->ddc_bus);
    if (v->ddc2_active)
        return !!(ctl & CRTC_CTL_DDC_SDA);

    /* DDC1 sends the base EDID MSB first, with a zero after each byte. */
    unsigned bit = v->ddc1_bit % 9;
    return bit < 8 && (v->ddc1_edid[v->ddc1_bit / 9] & (0x80 >> bit));
}

static void
verite_ddc_drive(verite_t *v, uint8_t previous)
{
    uint8_t ctl = v->regs[CRTC_CTL];
    if ((previous ^ ctl) & CRTC_CTL_DDC_SCL)
        v->ddc2_active = 1;
    /* The direction bit releases SDA only. The BIOS still clocks SCL
       while reading data from the monitor. */
    i2c_gpio_set(v->ddc_bus, !!(ctl & CRTC_CTL_DDC_SCL),
                 !(ctl & CRTC_CTL_DDC_DRIVE) || !!(ctl & CRTC_CTL_DDC_SDA));
}

static void
verite_latch_pixel_clock(verite_t *v)
{
    uint32_t word = v->pll_shift;
    unsigned m    = ((word >> 10) & 0xff) + 2;
    unsigned n    = (word & 0xff) + 2;
    unsigned p    = (word >> 8) & 3;
    double   hz   = (word & PLL_VGA_CLOCK) ? 0.0 : 2.0 * PLL_REF_HZ * m / (n << p);

    if (hz != v->pixel_clock_hz) {
        v->pixel_clock_hz = hz;
        svga_recalctimings(&v->vga.svga);
    }
}

static uint32_t
verite_ext_read(void *priv, uint8_t reg, uint8_t len)
{
    verite_t *v      = priv;
    /* Reading the PLL port transfers the serial word to the clock generator. */
    if (reg <= PLL_DEVICE && reg + len > PLL_DEVICE)
        verite_latch_pixel_clock(v);
    svga_t   *svga   = &v->vga.svga;
    uint32_t  status = (reg < CRTC_STATUS + 4 && reg + len > CRTC_STATUS) ? verite_crtc_status(v) : 0;
    uint32_t  test   = (verite_reg(v, CRTC_TEST) & ~(CRTC_TEST_VBLANK | CRTC_TEST_NOT_VBLANK))
                       | (svga->vc >= svga->vblankstart ? CRTC_TEST_VBLANK : CRTC_TEST_NOT_VBLANK);

    uint32_t value = 0;
    for (unsigned i = 0; i < len; i++) {
        uint8_t r    = reg + i;
        uint8_t byte = v->regs[r];
        if (r >= RAMDAC && r < RAMDAC + 16)
            byte = bt48x_ramdac_in(dac_ports[r & 3], r & 4, r & 8, v->ramdac, svga);
        else if (r >= CRTC_STATUS && r < CRTC_STATUS + 4)
            byte = status >> (8 * (r - CRTC_STATUS));
        else if (r >= CRTC_TEST && r < CRTC_TEST + 4)
            byte = test >> (8 * (r - CRTC_TEST));
        else if (r >= FRAME_BASE_B && r < FRAME_BASE_A)
            byte = v->frame_base_b >> (8 * (r - FRAME_BASE_B));
        else if (r == CRTC_CTL)
            byte = (byte & ~CRTC_CTL_DDC_SDA) | (verite_ddc_data(v) ? CRTC_CTL_DDC_SDA : 0);
        else if (r > CRTC_CTL && r < CRTC_CTL + 4) {
            unsigned shift = 8 * (r - CRTC_CTL);
            byte = (byte & ~(CRTC_CTL_FRAME_FLAGS >> shift)) | (v->frame_flags >> shift);
        }
        value |= (uint32_t) byte << (8 * i);
    }
    return value;
}

static void
verite_dac_write(verite_t *v, uint16_t port, int rs2, int rs3, uint8_t value)
{
    svga_t *svga     = &v->vga.svga;
    uint8_t gdc_mode = svga->gdcreg[5];
    int     dac_type = svga->ramdac_type;
    int     command0 = port == 0x3c6 && rs2 && !rs3;

    v->word_colors_valid = 0;

    if (command0)
        v->dac_command0 = value;
    bt48x_ramdac_out(port, rs2, rs3, value, v->ramdac, svga);
    if (command0)
        svga->ramdac_type = dac_type;

    /* Native pixel formats come from CRTC_CTL. Keep the VGA mode registers
       independent of changes to the RAMDAC's native pixel format. */
    if (command0 || svga->bpp != 8 || svga->gdcreg[5] != gdc_mode) {
        svga->bpp       = 8;
        svga->gdcreg[5] = gdc_mode;
        svga_recalctimings(svga);
    }
}

static void
verite_ext_write(void *priv, uint8_t reg, uint32_t value, uint8_t len)
{
    verite_t *v = priv;
    uint8_t   previous_ctl = v->regs[CRTC_CTL];
    uint32_t  previous_dram = verite_reg(v, DRAM_CTL);
    for (unsigned i = 0; i < len; i++) {
        uint8_t r    = reg + i;
        uint8_t byte = value >> (8 * i);
        if (r >= RAMDAC && r < RAMDAC + 16) {
            verite_dac_write(v, dac_ports[r & 3], r & 4, r & 8, byte);
        } else {
            v->regs[r] = byte;
            /* The guest clocks one bit per write, most significant bit first. */
            if (r == PLL_DEVICE)
                v->pll_shift = ((v->pll_shift << 1) | (byte & 1)) & PLL_WORD_MASK;
        }
    }
    if (reg <= CRTC_CTL && reg + len > CRTC_CTL)
        verite_ddc_drive(v, previous_ctl);
    if (reg < DRAM_CTL + 4 && reg + len > DRAM_CTL) {
        uint32_t dram = verite_reg(v, DRAM_CTL);
        if ((dram ^ previous_dram) & (DRAM_REFRESH_DISABLE | DRAM_REFRESH_COUNT)) {
            v->dram_last_refresh_ns = v->time_ns;
            v->dram_refresh_pending = 0;
        }
        verite_core_set_precharge(v->core, !!(dram & DRAM_SLOW_PRECHARGE));
    }
    if (reg < FRAME_BASE_A + 4 && reg + len > FRAME_BASE_A) {
        /* Writing A commits the pending B address as part of the same pair. */
        v->frame_base_b = verite_reg(v, FRAME_BASE_B);
        v->frame_flags &= ~(CRTC_CTL_DISPLAYED | CRTC_CTL_SWITCHED);
    }
    if (reg < CRTC_STATUS && reg + len > CRTC_CTL) {
        svga_recalctimings(&v->vga.svga);
        verite_redraw(v);
    }
}

static uint32_t
verite_dma_read(void *priv, uint32_t address)
{
    const verite_t *v = priv;
    uint8_t         data[4];
    if (!(v->command & 4))
        return UINT32_MAX;
    dma_bm_read(address, data, sizeof(data), 4);
    return data[0] | ((uint32_t) data[1] << 8) | ((uint32_t) data[2] << 16) | ((uint32_t) data[3] << 24);
}

static void
verite_dma_read_block(void *priv, uint32_t address, uint32_t *words, unsigned count)
{
    const verite_t *v = priv;
    uint8_t         data[4 * 32];
    if (!(v->command & 4) || count > 32) {
        for (unsigned i = 0; i < count; i++)
            words[i] = (v->command & 4) ? verite_dma_read(priv, address + 4 * i) : UINT32_MAX;
        return;
    }
    dma_bm_read(address, data, 4 * count, 4);
    for (unsigned i = 0; i < count; i++)
        words[i] = data[4 * i] | ((uint32_t) data[4 * i + 1] << 8) | ((uint32_t) data[4 * i + 2] << 16)
                   | ((uint32_t) data[4 * i + 3] << 24);
}

static void verite_dma_moved(void *priv, uint64_t ns, const uint32_t regs[3]);

static const verite_board board = { verite_ext_read, verite_ext_write, verite_dma_read, verite_dma_read_block,
                                    verite_dma_moved };

static void
verite_update_bank(verite_t *v)
{
    uint32_t bank = verite_core_bank(v->core) & 0xffe000;
    int      vga  = v->mode & MODE_VGA;
    v->bank       = bank;
    if ((v->command & 2) && vga)
        mem_mapping_enable(&v->vga.svga.mapping);
    else
        mem_mapping_disable(&v->vga.svga.mapping);
    if ((v->command & 2) && !vga && (v->mode & MODE_VESA))
        mem_mapping_enable(&v->banked);
    else
        mem_mapping_disable(&v->banked);
}

static void
verite_settle(verite_t *v)
{
    static const uint8_t endian_xor[] = { 0, 3, 1, 2 };
    uint32_t             status       = verite_core_status(v->core);
    v->endian                         = endian_xor[(status >> 8) & 3];
    uint8_t mode                      = verite_core_mode(v->core);
    if ((v->mode ^ mode) & (MODE_VESA | MODE_VGA)) {
        v->mode = mode;
        verite_update_bank(v);
        svga_recalctimings(&v->vga.svga);
        verite_redraw(v);
    }
    if (status & 2)
        verite_redraw(v);
    uint8_t asserted = status & 1;
    if (asserted != v->irq_asserted) {
        v->irq_asserted = asserted;
        pci_irq(v->pci_slot, PCI_INTA, 0, asserted, &v->irq_state);
    }
}

static void
verite_run(verite_t *v)
{
    verite_core_run(v->core, v->time_ns, v->vga.svga.vram, VRAM_SIZE, &board, v);
    verite_settle(v);
}

/* The worker thread's view of the board. CRTC and DRAM register reads only
   look at state, so the worker reads them directly once its posted writes
   have landed; a torn read of the beam position costs one poll. RAMDAC and
   PLL reads, and DMA outside plain RAM, run on the CPU thread while the
   worker waits. */
static uint32_t
verite_request(verite_t *v, int kind)
{
    verite_worker *w = v->w;
    atomic_store_explicit(&w->req, kind, memory_order_release);
    for (unsigned spin = 0; atomic_load_explicit(&w->req, memory_order_acquire) != REQ_DONE; spin++) {
        if (atomic_load_explicit(&w->stop, memory_order_relaxed))
            return 0;
        if (spin < 256)
            verite_pause();
        else {
            thread_reset_event(w->done);
            if (atomic_load_explicit(&w->req, memory_order_acquire) != REQ_DONE)
                thread_wait_event(w->done, 1);
        }
    }
    uint32_t value = w->req_value;
    atomic_store_explicit(&w->req, REQ_IDLE, memory_order_relaxed);
    return value;
}

static uint32_t
worker_board_read(void *priv, uint8_t reg, uint8_t len)
{
    verite_t *v = priv;
    if (atomic_load_explicit(&v->w->posted_head, memory_order_relaxed)
        != atomic_load_explicit(&v->w->posted_tail, memory_order_relaxed))
        verite_request(v, REQ_FLUSH);
    if (reg + len <= RAMDAC)
        return verite_ext_read(v, reg, len);
    v->w->req_reg = reg;
    v->w->req_len = len;
    return verite_request(v, REQ_READ);
}

static void
worker_board_write(void *priv, uint8_t reg, uint32_t value, uint8_t len)
{
    verite_t      *v    = priv;
    verite_worker *w    = v->w;
    unsigned       tail = atomic_load_explicit(&w->posted_tail, memory_order_relaxed);
    if (tail - atomic_load_explicit(&w->posted_head, memory_order_acquire) >= POSTED_SIZE)
        verite_request(v, REQ_FLUSH);
    w->posted[tail % POSTED_SIZE].reg   = reg;
    w->posted[tail % POSTED_SIZE].len   = len;
    w->posted[tail % POSTED_SIZE].value = value;
    atomic_store_explicit(&w->posted_tail, tail + 1, memory_order_release);
}

static void
worker_dma_read_block(void *priv, uint32_t address, uint32_t *words, unsigned count)
{
    verite_t *v     = priv;
    /* Plain RAM above the first megabyte needs no mapping lookup. This
       assumes nothing remaps it, as SMRAM or a memory hole could. */
    size_t top = mem_ram_size();
    if (top > (1U << 30))
        top = 1U << 30;
    if ((v->command & 4) && address >= 0x100000 && (size_t) address + 4 * count <= top) {
        memcpy(words, ram + address, 4 * count);
        return;
    }
    v->w->req_value = address;
    v->w->req_words = words;
    v->w->req_count = count;
    verite_request(v, REQ_DMA);
}

static uint32_t
worker_dma_read(void *priv, uint32_t address)
{
    uint32_t word;
    worker_dma_read_block(priv, address, &word, 1);
    return word;
}

static const verite_board worker_board = { worker_board_read, worker_board_write, worker_dma_read,
                                           worker_dma_read_block, verite_dma_moved };

/* Records a DMA register change. The entries are in order of time: one the
   guest made, at its own time, replaces those of the RISC's run past it.
   Entries before the guest's time but the last are no longer needed. */
static void
verite_dma_moved(void *priv, uint64_t ns, const uint32_t regs[3])
{
    verite_t *v = priv;
    if (!v->w)
        return;
    verite_worker *w    = v->w;
    unsigned       head = atomic_load_explicit(&w->hist_head, memory_order_relaxed);
    unsigned       tail = atomic_load_explicit(&w->hist_tail, memory_order_relaxed);
    while (tail != head && w->hist[(tail - 1) % HIST_SIZE].ns > ns)
        tail--;
    if (tail != head) {
        uint64_t now = atomic_load_explicit(&w->now_ns, memory_order_relaxed);
        while (tail - head > 1 && w->hist[(head + 1) % HIST_SIZE].ns <= now)
            head++;
        atomic_store_explicit(&w->hist_head, head, memory_order_release);
        /* Progress within the same list entry needs no entry of its own. */
        unsigned last = (tail - 1) % HIST_SIZE;
        if (tail - head >= HIST_SIZE
            || (w->hist[last].regs[0] == regs[0] && ns - w->hist[last].ns < HIST_NS && ns >= w->hist[last].ns)) {
            memcpy(w->hist[last].regs, regs, sizeof(w->hist[last].regs));
            atomic_store_explicit(&w->hist_tail, tail, memory_order_release);
            return;
        }
    }
    w->hist[tail % HIST_SIZE].ns = ns;
    memcpy(w->hist[tail % HIST_SIZE].regs, regs, sizeof(w->hist[0].regs));
    atomic_store_explicit(&w->hist_tail, tail + 1, memory_order_release);
}

/* The DMA registers as of the guest's time ns: the last entry at or before
   it. Called on the CPU thread once the RISC has run to ns. */
static int
verite_dma_at(verite_worker *w, uint64_t ns, uint32_t regs[3])
{
    unsigned head = atomic_load_explicit(&w->hist_head, memory_order_acquire);
    unsigned tail = atomic_load_explicit(&w->hist_tail, memory_order_acquire);
    if (head == tail)
        return 0;
    /* Binary search for the first entry past ns. */
    unsigned lo = head + 1, hi = tail;
    while (lo < hi) {
        unsigned mid = lo + (hi - lo) / 2;
        if (w->hist[mid % HIST_SIZE].ns <= ns)
            lo = mid + 1;
        else
            hi = mid;
    }
    memcpy(regs, w->hist[(lo - 1) % HIST_SIZE].regs, 3 * sizeof(uint32_t));
    return 1;
}

/* Applies the worker's posted board writes and answers its request, on the
   CPU thread. */
static void
verite_serve(verite_t *v)
{
    verite_worker *w    = v->w;
    unsigned       head = atomic_load_explicit(&w->posted_head, memory_order_relaxed);
    unsigned       tail = atomic_load_explicit(&w->posted_tail, memory_order_acquire);
    for (; head != tail; head++)
        verite_ext_write(v, w->posted[head % POSTED_SIZE].reg, w->posted[head % POSTED_SIZE].value,
                         w->posted[head % POSTED_SIZE].len);
    atomic_store_explicit(&w->posted_head, head, memory_order_release);

    int kind = atomic_load_explicit(&w->req, memory_order_acquire);
    switch (kind) {
        case REQ_READ:
            w->req_value = verite_ext_read(v, w->req_reg, w->req_len);
            break;
        case REQ_FLUSH:
            break;
        case REQ_DMA:
            if (w->req_count == 1)
                w->req_words[0] = verite_dma_read(v, w->req_value);
            else
                verite_dma_read_block(v, w->req_value, w->req_words, w->req_count);
            break;
        default:
            return;
    }
    atomic_store_explicit(&w->req, REQ_DONE, memory_order_release);
    thread_set_event(w->done);
}

/* Copies the status registers for the CPU thread. Called by the core's owner. */
static void
verite_publish(verite_t *v)
{
    verite_worker *w = v->w;
    uint8_t        regs[32];
    for (unsigned r = 0; r < 32; r += 4) {
        uint32_t value = verite_core_read(v->core, 0x40 + r, 4, v->vga.svga.vram, VRAM_SIZE);
        memcpy(regs + r, &value, 4);
    }
    unsigned seq = atomic_load_explicit(&w->mirror_seq, memory_order_relaxed);
    atomic_store_explicit(&w->mirror_seq, seq + 1, memory_order_relaxed);
    atomic_thread_fence(memory_order_release);
    memcpy(w->mirror, regs, sizeof(regs));
    w->mirror_head = atomic_load_explicit(&w->head, memory_order_relaxed);
    w->mirror_ns   = verite_core_time_ns(v->core);
    atomic_store_explicit(&w->dma_end_ns, verite_core_dma_end_ns(v->core), memory_order_relaxed);
    atomic_store_explicit(&w->dma_parked, verite_core_dma_parked(v->core), memory_order_relaxed);
    atomic_store_explicit(&w->mirror_seq, seq + 2, memory_order_release);
}

/* Reads the published status registers. Data-FIFO writes still queued in
   front of the worker are in the FIFO as far as the guest can tell, so with
   any queued the FIFO is not empty and the RISC is not waiting for input. A
   guest that waits for an empty FIFO before reusing VRAM the RISC reads
   from, as Speedy3D does without DMA, would otherwise race the worker. */
static void
verite_mirror_snapshot(verite_worker *w, uint8_t *regs, unsigned *head, uint64_t *ns)
{
    unsigned seq;
    do {
        seq = atomic_load_explicit(&w->mirror_seq, memory_order_acquire);
        memcpy(regs, w->mirror, sizeof(w->mirror));
        *head = w->mirror_head;
        *ns   = w->mirror_ns;
        atomic_thread_fence(memory_order_acquire);
    } while ((seq & 1) || seq != atomic_load_explicit(&w->mirror_seq, memory_order_relaxed));
}

static int
verite_reads_dma(uint8_t reg, uint8_t len)
{
    return (reg <= VIO_MEM_ENDIAN_REG && reg + len > VIO_MEM_ENDIAN_REG) || (reg < DMA_REGS_END && reg + len > DMA_REGS);
}

static uint32_t
verite_mirror_read(verite_worker *w, uint8_t reg, uint8_t len, uint64_t now)
{
    uint8_t  regs[32];
    unsigned head;
    uint64_t ns;
    uint32_t dma[3];
    verite_mirror_snapshot(w, regs, &head, &ns);
    /* A running DMA's registers as of the guest's time, wherever the worker
       has got to since. */
    if (verite_reads_dma(reg, len) && verite_dma_at(w, now, dma)) {
        memcpy(regs + DMA_REGS - 0x40, dma, sizeof(dma));
        regs[VIO_MEM_ENDIAN_REG - 0x40] = (regs[VIO_MEM_ENDIAN_REG - 0x40] & 0x7f) | ((dma[0] & 1) << 7);
    }
    unsigned queued = atomic_load_explicit(&w->tail, memory_order_acquire) - head;
    if (queued) {
        /* The queue takes writes while it has room, though never all 32. */
        regs[0] = QUEUE_SIZE - queued >= SYNC_FREE ? 30 : 0;
        uint8_t *status = &regs[0x4a - 0x40];
        if (*status & 0x10) {
            *status &= ~0x10;
            if (!(*status & 0x23))
                *status &= ~0x80;
        }
    }
    uint32_t value = 0;
    for (unsigned i = 0; i < len; i++)
        value |= (uint32_t) regs[reg - 0x40 + i] << (8 * i);
    return value;
}

static void
verite_wake(verite_worker *w)
{
    if (atomic_load(&w->sleeping))
        thread_set_event(w->wake);
}

/* Applies queued entries and runs the RISC. The worker stops at the end of
   its slice, when the RISC polls the board or when it would get too far ahead
   of the guest. The CPU thread applies everything and catches the RISC up as
   the synchronous path does. Returns whether the RISC can make more progress
   without the guest's time advancing. */
static int
verite_drain(verite_t *v, const verite_board *ops, int worker)
{
    verite_worker *w      = v->w;
    verite_core   *c      = v->core;
    uint8_t       *vram   = v->vga.svga.vram;
    uint64_t       now    = atomic_load_explicit(&w->now_ns, memory_order_acquire);
    int64_t        budget = worker ? SLICE : INT64_MAX;
    int            more   = 0;

    unsigned refresh = atomic_exchange_explicit(&w->refresh, 0, memory_order_relaxed);
    if (refresh)
        verite_core_refresh(c, refresh);

    while (budget > 0) {
        if (worker && atomic_load_explicit(&w->cpu_want, memory_order_relaxed)) {
            more = 1;
            break;
        }
        unsigned head = atomic_load_explicit(&w->head, memory_order_relaxed);
        unsigned tail = atomic_load_explicit(&w->tail, memory_order_acquire);
        if (head == tail) {
            if (!worker)
                break;
            int64_t used = verite_core_advance(c, now, RUN_AHEAD, budget, vram, VRAM_SIZE, ops, v);
            budget -= used;
            more = used > 0 && budget <= 0;
            break;
        }

        const verite_entry *e    = &w->queue[head % QUEUE_SIZE];
        int64_t             used = verite_core_advance(c, e->t, SYNC_AHEAD, budget, vram, VRAM_SIZE, ops, v);
        budget -= used;
        if (budget <= 0) {
            more = 1;
            break;
        }
        if (worker && e->kind == Q_WRITE && e->reg < 0x10 && verite_core_input_full(c) && !verite_core_waiting(c)) {
            /* Let the RISC make room, as far as it may run ahead. */
            used = verite_core_advance(c, now, RUN_AHEAD, budget, vram, VRAM_SIZE, ops, v);
            budget -= used;
            more = used > 0 && !verite_core_polling(c);
            if (!more || verite_core_input_full(c))
                break;
        }
        if (e->kind == Q_WRITE)
            verite_core_write(c, e->reg, e->value, e->len, vram, VRAM_SIZE, ops, v);
        else
            verite_core_retrace(c);
        atomic_store_explicit(&w->head, head + 1, memory_order_release);
        more = 1;
    }
    if (!worker)
        verite_core_run(c, now, vram, VRAM_SIZE, ops, v);
    return more;
}

static int
verite_has_work(verite_worker *w)
{
    if (atomic_load(&w->owner) != OWN_NONE || atomic_load(&w->cpu_want))
        return 0;
    if (atomic_load(&w->head) != atomic_load(&w->tail))
        return 1;
    return !atomic_load(&w->wait_input) && atomic_load(&w->now_ns) != w->slept_ns;
}

/* Spins up to us microseconds of host time for work to come. */
static int
verite_await_work(verite_worker *w, long us)
{
    struct timespec start, now;
    timespec_get(&start, TIME_UTC);
    for (unsigned spin = 1;; spin++) {
        if (verite_has_work(w) || atomic_load_explicit(&w->stop, memory_order_relaxed))
            return 1;
        verite_pause();
        if (!(spin % 64)) {
            timespec_get(&now, TIME_UTC);
            long elapsed = (now.tv_sec - start.tv_sec) * 1000000L + (now.tv_nsec - start.tv_nsec) / 1000;
            if (elapsed >= us || elapsed < 0)
                return 0;
        }
    }
}

static void
verite_worker_thread(void *priv)
{
    verite_t      *v = priv;
    verite_worker *w = v->w;

    while (!atomic_load(&w->stop)) {
        int expected = OWN_NONE;
        if (!atomic_load(&w->cpu_want) && atomic_compare_exchange_strong(&w->owner, &expected, OWN_WORKER)) {
            uint64_t now  = atomic_load(&w->now_ns);
            int      more = verite_drain(v, &worker_board, 1);
            uint32_t st   = verite_core_status(v->core);
            verite_publish(v);
            atomic_store(&w->irq, st & 1);
            if (st & 2)
                atomic_store(&w->drew, 1);
            atomic_store(&w->wait_input, verite_core_waiting(v->core));
            w->slept_ns = now;
            atomic_store(&w->owner, OWN_NONE);
            if (more)
                continue;
        }
        /* While the guest runs, its ticks come microseconds apart in host
           time. Waking from an event takes longer, and a guest reading a
           running DMA waits for the worker to reach its time. */
        if (verite_await_work(w, IDLE_SPIN_US))
            continue;
        atomic_store(&w->sleeping, 1);
        thread_reset_event(w->wake);
        if (!verite_has_work(w) && !atomic_load(&w->stop))
            thread_wait_event(w->wake, 1);
        atomic_store(&w->sleeping, 0);
    }
}

/* Takes the core from the worker, answering its board requests meanwhile. */
static void
verite_take(verite_t *v)
{
    verite_worker *w = v->w;
    atomic_store(&w->cpu_want, 1);
    for (;;) {
        int expected = OWN_NONE;
        if (atomic_compare_exchange_weak(&w->owner, &expected, OWN_CPU))
            break;
        verite_serve(v);
        verite_pause();
    }
    atomic_store(&w->cpu_want, 0);
    /* The RISC's board writes from here on land directly, so the ones the
       worker posted before must land first. A flip or CRTC write applied
       after a newer one undoes it. */
    verite_serve(v);
}

static void
verite_give(verite_t *v)
{
    verite_worker *w = v->w;
    verite_publish(v);
    atomic_store(&w->irq, v->irq_asserted);
    atomic_store(&w->owner, OWN_NONE);
    verite_wake(w);
}

/* Takes the core and brings the RISC up to the guest's time. */
static void
verite_sync(verite_t *v)
{
    verite_take(v);
    atomic_store(&v->w->now_ns, v->time_ns);
    verite_drain(v, &board, 0);
    verite_core_finish_dma(v->core, v->time_ns, &board, v);
    verite_settle(v);
}

/* Ends a DMA the worker ran to the end of its list once the guest's time
   reaches the point the RISC got there, unless the guest has extended the
   list meanwhile. Ending it at the next tick instead would leave it busy
   for a guest that in step would find it done; such a guest may then append
   to it, and the end would be reread halfway through the guest's patch. */
static void
verite_finish_parked(verite_t *v)
{
    verite_worker *w = v->w;
    if (!atomic_load_explicit(&w->dma_parked, memory_order_acquire)
        || !verite_core_dma_end_due(atomic_load_explicit(&w->dma_end_ns, memory_order_relaxed), v->time_ns))
        return;
    verite_take(v);
    verite_core_finish_dma(v->core, v->time_ns, &board, v);
    verite_settle(v);
    verite_give(v);
}

/* A guest reading the DMA's state sees where a DMA running in step would be
   by now, and games decide from it whether to wait for a DMA, append to it
   or reuse what it has read. Rebel Moon refills whichever of its two lists
   the DMA pointer is not in, and appends to a list it finds still moving.
   Shown a pointer behind the guest's time, it overwrote a list before the
   DMA had read it; shown one ahead, it appended to a list the RISC had
   finished, whose end was then reread halfway through the patch. A worker
   that has run past the guest's time answers from the DMA states it
   recorded on the way; one behind it is waited for, answering its board
   requests meanwhile. A DMA published idle needs no wait, since only the
   guest starts one. */
static void
verite_dma_catch_up(verite_t *v, uint8_t reg, uint8_t len)
{
    if (!verite_reads_dma(reg, len))
        return;
    verite_worker *w = v->w;
    for (unsigned spin = 0;; spin++) {
        uint8_t  regs[32];
        unsigned head;
        uint64_t ns;
        verite_mirror_snapshot(w, regs, &head, &ns);
        if (!(regs[DMA_REGS - 0x40] & 1) || ns >= v->time_ns || atomic_load_explicit(&w->dma_parked, memory_order_acquire))
            return;
        if ((!spin && atomic_load(&w->sleeping)) || spin >= (1U << 24)) {
            /* An idle worker takes a while to wake, and one not getting
               there is no faster; run the RISC here. */
            verite_sync(v);
            verite_settle(v);
            verite_give(v);
            return;
        }
        if (!spin)
            verite_wake(w);
        verite_serve(v);
        verite_pause();
    }
}

static void
verite_settle_async(verite_t *v)
{
    verite_worker *w = v->w;
    if (atomic_exchange(&w->drew, 0))
        verite_redraw(v);
    uint8_t asserted = atomic_load(&w->irq);
    if (asserted != v->irq_asserted) {
        v->irq_asserted = asserted;
        pci_irq(v->pci_slot, PCI_INTA, 0, asserted, &v->irq_state);
    }
}

static void
verite_push(verite_t *v, uint8_t kind, uint8_t reg, uint32_t value, uint8_t len)
{
    verite_worker *w    = v->w;
    unsigned       tail = atomic_load_explicit(&w->tail, memory_order_relaxed);
    if (tail - atomic_load_explicit(&w->head, memory_order_acquire) >= QUEUE_SIZE) {
        verite_sync(v);
        verite_give(v);
    }
    verite_entry *e = &w->queue[tail % QUEUE_SIZE];
    e->t            = v->time_ns;
    e->kind         = kind;
    e->reg          = reg;
    e->value        = value;
    e->len          = len;
    atomic_store(&w->tail, tail + 1);
    verite_wake(w);
}

static void
verite_add_refresh(verite_t *v, unsigned cycles)
{
    if (v->w)
        atomic_fetch_add_explicit(&v->w->refresh, cycles, memory_order_relaxed);
    else
        verite_core_refresh(v->core, cycles);
}

static void
verite_dram_refresh(verite_t *v)
{
    uint32_t dram     = verite_reg(v, DRAM_CTL);
    unsigned interval = ((dram & DRAM_REFRESH_COUNT) >> 16) * DRAM_REFRESH_UNIT_NS;
    if (!interval || (dram & DRAM_REFRESH_DISABLE))
        return;

    unsigned refreshes = (v->time_ns - v->dram_last_refresh_ns) / interval;
    v->dram_last_refresh_ns += (uint64_t) refreshes * interval;
    v->dram_refresh_pending += refreshes;
    if ((dram & DRAM_REFRESH_VBLANK) && v->vga.svga.dispon)
        return;

    verite_add_refresh(v, v->dram_refresh_pending * (DRAM_REFRESH_CYCLES + !!(dram & DRAM_SLOW_PRECHARGE)));
    v->dram_refresh_pending = 0;
}

static void
verite_timer(void *priv)
{
    verite_t *v = priv;
    timer_advance_u64(&v->timer, TICK_US * TIMER_USEC);
    v->time_ns += TICK_US * 1000;
    verite_dram_refresh(v);
    if (v->w) {
        verite_worker *w = v->w;
        atomic_store(&w->now_ns, v->time_ns);
        if (!atomic_load_explicit(&w->wait_input, memory_order_relaxed))
            verite_wake(w);
        verite_serve(v);
        verite_finish_parked(v);
        verite_settle_async(v);
    } else
        verite_run(v);
}

static void
verite_vsync(svga_t *svga)
{
    verite_t *v   = svga->priv;
    uint32_t  ctl = verite_reg(v, CRTC_CTL);

    /* Address registers take effect at retrace, even with RISC retrace disabled. */
    v->frame[0]       = verite_reg(v, FRAME_BASE_A) & 0xffffff;
    v->frame[1]       = v->frame_base_b & 0xffffff;
    v->scanout_offset = verite_reg(v, CRTC_OFFSET) & 0xffff;
    if (svga->render == verite_render)
        verite_update_pitch(v, svga->hdisp / svga->dots_per_clock, v->native_bpp, ctl);

    v->frame_flags = (v->frame_flags ^ CRTC_CTL_EVEN_FRAME) | CRTC_CTL_DISPLAYED | CRTC_CTL_SWITCHED;
    if (ctl & CRTC_CTL_VSYNC) {
        v->ddc1_bit = (v->ddc1_bit + 1) % (sizeof(v->ddc1_edid) * 9);
        if (v->w) {
            verite_push(v, Q_RETRACE, 0, 0, 0);
            verite_settle_async(v);
        } else {
            verite_core_retrace(v->core);
            verite_settle(v);
        }
    }
}

static uint32_t
verite_port_read(uint16_t port, uint8_t len, verite_t *v)
{
    uint8_t reg = port & 0xff;
    if (v->w) {
        verite_worker *w = v->w;
        verite_serve(v);
        /* Status registers come from the last state the worker published,
           FIFO space and idle status counting the queue in front of the
           RISC; the host's own registers come from the core directly. */
        if (reg >= 0x40 && reg + len <= 0x60) {
            verite_finish_parked(v);
            verite_dma_catch_up(v, reg, len);
            verite_settle_async(v);
            return verite_mirror_read(w, reg, len, v->time_ns);
        }
        if (reg >= 0x60 && reg + len <= 0x80 && !(reg < 0x68 && reg + len > 0x64)) {
            verite_settle_async(v);
            return verite_core_read(v->core, reg, len, v->vga.svga.vram, VRAM_SIZE);
        }
        verite_sync(v);
        uint32_t value = reg >= 0x80 ? verite_ext_read(v, reg, len)
                                     : verite_core_read(v->core, reg, len, v->vga.svga.vram, VRAM_SIZE);
        verite_settle(v);
        verite_give(v);
        return value;
    }
    verite_run(v);
    if (reg >= 0x80)
        return verite_ext_read(v, reg, len);
    uint32_t value = verite_core_read(v->core, reg, len, v->vga.svga.vram, VRAM_SIZE);
    verite_settle(v);
    return value;
}

static void
verite_port_write(uint16_t port, uint32_t value, uint8_t len, verite_t *v)
{
    uint8_t reg = port & 0xff;
    if (v->w) {
        verite_serve(v);
        if (reg < 0x10) {
            verite_push(v, Q_WRITE, reg, value, len);
            return;
        }
        verite_sync(v);
        if (reg >= 0x80)
            verite_ext_write(v, reg, value, len);
        else {
            verite_core_write(v->core, reg, value, len, v->vga.svga.vram, VRAM_SIZE, &board, v);
            if ((reg <= 0x48 && reg + len > 0x48) || (reg < 0x78 && reg + len > 0x72))
                verite_update_bank(v);
        }
        verite_settle(v);
        verite_give(v);
        return;
    }
    verite_run(v);
    if (reg >= 0x80)
        verite_ext_write(v, reg, value, len);
    else {
        verite_core_write(v->core, reg, value, len, v->vga.svga.vram, VRAM_SIZE, &board, v);
        if (reg < 0x10)
            verite_run(v);
        if ((reg <= 0x48 && reg + len > 0x48) || (reg < 0x78 && reg + len > 0x72))
            verite_update_bank(v);
        verite_settle(v);
    }
}

static uint8_t
verite_in(uint16_t port, void *priv)
{
    return verite_port_read(port, 1, priv);
}
static uint16_t
verite_inw(uint16_t port, void *priv)
{
    return verite_port_read(port, 2, priv);
}
static uint32_t
verite_inl(uint16_t port, void *priv)
{
    return verite_port_read(port, 4, priv);
}
static void
verite_out(uint16_t port, uint8_t val, void *priv)
{
    verite_port_write(port, val, 1, priv);
}
static void
verite_outw(uint16_t port, uint16_t val, void *priv)
{
    verite_port_write(port, val, 2, priv);
}
static void
verite_outl(uint16_t port, uint32_t val, void *priv)
{
    verite_port_write(port, val, 4, priv);
}

static uint8_t
verite_vga_in(uint16_t port, void *priv)
{
    verite_t *v = priv;
    if (port >= 0x3c6 && port <= 0x3c9)
        return bt48x_ramdac_in(port, 0, 0, v->ramdac, &v->vga.svga);
    return vga_in(port, &v->vga);
}

static void
verite_vga_out(uint16_t port, uint8_t value, void *priv)
{
    verite_t *v = priv;
    if (port >= 0x3c6 && port <= 0x3c9)
        verite_dac_write(v, port, 0, 0, value);
    else
        vga_out(port, value, &v->vga);
    if (port == 0x3cf && v->vga.svga.gdcaddr == 6)
        verite_update_bank(v);
}

static uint8_t
verite_vram_read(verite_t *v, uint32_t addr)
{
    return v->vga.svga.vram[(addr ^ v->endian) & (VRAM_SIZE - 1)];
}

static void
verite_vram_write(verite_t *v, uint32_t addr, uint8_t value)
{
    addr                                = (addr ^ v->endian) & (VRAM_SIZE - 1);
    v->vga.svga.vram[addr]              = value;
    v->vga.svga.changedvram[addr >> 12] = changeframecount;
}

static uint8_t
linear_read(uint32_t addr, void *priv)
{
    return verite_vram_read(priv, addr & (BAR_SIZE - 1));
}
static void
linear_write(uint32_t addr, uint8_t val, void *priv)
{
    verite_vram_write(priv, addr & (BAR_SIZE - 1), val);
}

static uint8_t
banked_read(uint32_t addr, void *priv)
{
    verite_t *v = priv;
    return verite_vram_read(v, v->bank + (addr & 0xffff));
}

static void
banked_write(uint32_t addr, uint8_t val, void *priv)
{
    verite_t *v = priv;
    verite_vram_write(v, v->bank + (addr & 0xffff), val);
}

static void
verite_decode(verite_t *v, int enable)
{
    if (v->io_installed) {
        io_removehandler((uint16_t) v->io_installed << 8, 0x100,
                         verite_in, verite_inw, verite_inl, verite_out, verite_outw, verite_outl, v);
        v->io_installed = 0;
    }
    io_removehandler(0x3a0, 0x20, verite_vga_in, NULL, NULL, verite_vga_out, NULL, NULL, v);
    io_removehandler(0x3c0, 0x20, verite_vga_in, NULL, NULL, verite_vga_out, NULL, NULL, v);
    if (!enable)
        return;
    v->vga.svga.vga_enabled = !!(v->command & 1);
    if (v->command & 1) {
        io_sethandler(0x3c0, 0x20, verite_vga_in, NULL, NULL, verite_vga_out, NULL, NULL, v);
        if (!(v->vga.svga.miscout & 1))
            io_sethandler(0x3a0, 0x20, verite_vga_in, NULL, NULL, verite_vga_out, NULL, NULL, v);
        if (v->io_hi) {
            v->io_installed = v->io_hi;
            io_sethandler((uint16_t) v->io_hi << 8, 0x100,
                          verite_in, verite_inw, verite_inl, verite_out, verite_outw, verite_outl, v);
        }
    }
    if ((v->command & 2) && v->bar_hi)
        mem_mapping_set_addr(&v->linear, (uint32_t) v->bar_hi << 24, BAR_SIZE);
    else
        mem_mapping_disable(&v->linear);
    verite_update_bank(v);

    if (v->rom_bar & 1)
        mem_mapping_set_addr(&v->bios.mapping, v->rom_bar & ~(ROM_SIZE - 1), ROM_SIZE);
    else
        mem_mapping_disable(&v->bios.mapping);
}

static uint8_t
verite_pci_read(int func, int addr, int len, void *priv)
{
    const verite_t *v = priv;
    (void) len;
    if (func)
        return 0xff;
    switch (addr) {
        case 0x00:
            return 0x63;
        case 0x01:
            return 0x11;
        case 0x02:
            return 0x01;
        case 0x04:
            return v->command;
        case 0x07:
            return 2;
        case 0x08:
            return 2;
        case 0x0b:
            return 3;
        case 0x13:
            return v->bar_hi;
        case 0x14:
            return 1;
        case 0x15:
            return v->io_hi;
        case 0x30:
        case 0x31:
        case 0x32:
        case 0x33:
            return v->rom_bar >> ((addr - 0x30) * 8);
        case 0x3c:
            return v->irq_line;
        case 0x3d:
            return PCI_INTA;
        case 0x50:
            return v->pci_control;
        default:
            return 0;
    }
}

static void
verite_pci_write(int func, int addr, int len, uint8_t value, void *priv)
{
    verite_t *v = priv;
    (void) len;
    if (func)
        return;
    switch (addr) {
        case 0x04:
            v->command = value & 7;
            break;
        case 0x13:
            v->bar_hi = value;
            break;
        case 0x15:
            v->io_hi = value;
            break;
        case 0x30:
        case 0x31:
        case 0x32:
        case 0x33:
            {
                unsigned shift = (addr - 0x30) * 8;
                v->rom_bar     = ((v->rom_bar & ~(0xffU << shift)) | ((uint32_t) value << shift)) & 0xffff8001;
                break;
            }
        case 0x3c:
            v->irq_line = value;
            return;
        case 0x50:
            v->pci_control = value;
            return;
        default:
            return;
    }
    verite_decode(v, 1);
}

static int
verite_available(void)
{
    return rom_present(VERITE_ROM);
}

static void
verite_speed_changed(void *priv)
{
    verite_t *v = priv;
    svga_recalctimings(&v->vga.svga);
}

static void
verite_reset(void *priv)
{
    verite_t *v = priv;
    if (v->w) {
        verite_take(v);
        atomic_store(&v->w->head, atomic_load(&v->w->tail));
        atomic_store(&v->w->refresh, 0);
        atomic_store(&v->w->hist_head, atomic_load(&v->w->hist_tail));
    }
    verite_core_reset(v->core);
    memset(v->regs, 0, sizeof(v->regs));
    v->command = 3;
    v->bar_hi = v->io_hi = v->irq_line = v->pci_control = 0;
    v->rom_bar                                          = 0xc0001;
    v->frame_flags                                      = 0;
    v->frame_base_b                                     = 0;
    v->frame[0] = v->frame[1]                            = 0;
    v->pitch[0] = v->pitch[1]                            = 0;
    v->scanout_offset                                   = 0;
    v->pll_shift                                        = PLL_VGA_CLOCK;
    v->pixel_clock_hz                                   = 0.0;
    v->dram_last_refresh_ns                             = v->time_ns;
    v->dram_refresh_pending                             = 0;
    v->ddc1_bit                                         = 0;
    v->ddc2_active                                      = 0;
    /* Terminate any transaction carried over from before the reset. */
    i2c_gpio_set(v->ddc_bus, 0, 0);
    i2c_gpio_set(v->ddc_bus, 1, 0);
    i2c_gpio_set(v->ddc_bus, 1, 1);
    v->mode                                             = verite_core_mode(v->core);
    verite_decode(v, 1);
    verite_settle(v);
    if (v->w)
        verite_give(v);
    verite_speed_changed(v);
}

static void *
verite_init(const device_t *info)
{
    verite_t *v = calloc(1, sizeof(*v));
    if (!v)
        return NULL;
    v->word_colors = malloc(65536 * sizeof(*v->word_colors));
    v->core        = v->word_colors ? verite_core_new() : NULL;
    if (!v->core) {
        free(v->word_colors);
        free(v);
        return NULL;
    }
#ifdef VERITE_JIT_X64
    verite_core_set_jit(v->core, device_get_config_int("recompiler"));
#endif
    if (rom_init(&v->bios, VERITE_ROM, 0xc0000, ROM_SIZE, ROM_SIZE - 1, 0, MEM_MAPPING_EXTERNAL)) {
        verite_core_free(v->core);
        free(v->word_colors);
        free(v);
        return NULL;
    }

    v->ramdac = device_add(&bt485_ramdac_device);
    v->ddc_bus = i2c_gpio_init("Vérité DDC");
    v->ddc_monitor = ddc_init(i2c_gpio_get_bus(v->ddc_bus));
    const uint8_t *edid = ddc_get_edid(v->ddc_monitor);
    if (edid) {
        memcpy(v->ddc1_edid, edid, sizeof(v->ddc1_edid));
        /* DDC1 can only carry one block. Remove the extension count. */
        v->ddc1_edid[127] += v->ddc1_edid[126];
        v->ddc1_edid[126] = 0;
    }
    video_inform(VIDEO_FLAG_TYPE_SPECIAL, &verite_timing);
    svga_init(info, &v->vga.svga, v, VRAM_SIZE, verite_recalctimings,
              verite_vga_in, verite_vga_out, NULL, NULL);
    v->vga.svga.ramdac            = v->ramdac;
    v->vga.svga.vsync_callback    = verite_vsync;
    mem_mapping_add(&v->linear, 0, 0, linear_read, NULL, NULL, linear_write, NULL, NULL,
                    NULL, MEM_MAPPING_EXTERNAL, v);
    mem_mapping_add(&v->banked, 0xa0000, 0x10000, banked_read, NULL, NULL, banked_write, NULL, NULL,
                    NULL, MEM_MAPPING_EXTERNAL, v);
    pci_add_card(PCI_ADD_NORMAL, verite_pci_read, verite_pci_write, v, &v->pci_slot);
    verite_reset(v);
    if (device_get_config_int("risc_thread")) {
        verite_worker *w = calloc(1, sizeof(*w));
        if (w) {
            w->wake = thread_create_event();
            w->done = thread_create_event();
            atomic_store(&w->now_ns, v->time_ns);
            v->w      = w;
            verite_core_defer_dma_end(v->core, 1);
            w->thread = thread_create_named(verite_worker_thread, v, "Verite RISC");
        }
    }
    timer_add(&v->timer, verite_timer, v, 0);
    timer_set_delay_u64(&v->timer, TICK_US * TIMER_USEC);
    return v;
}

static void
verite_close(void *priv)
{
    verite_t *v = priv;
    if (v->w) {
        verite_worker *w = v->w;
        atomic_store(&w->stop, 1);
        thread_set_event(w->wake);
        thread_set_event(w->done);
        thread_wait(w->thread);
        thread_destroy_event(w->wake);
        thread_destroy_event(w->done);
        free(w);
        v->w = NULL;
    }
    timer_disable(&v->timer);
    timer_disable(&v->vga.svga.timer);
    mem_mapping_disable(&v->vga.svga.mapping);
    verite_decode(v, 0);
    mem_mapping_disable(&v->linear);
    mem_mapping_disable(&v->banked);
    mem_mapping_disable(&v->bios.mapping);
    svga_close(&v->vga.svga);
    ddc_close(v->ddc_monitor);
    i2c_gpio_close(v->ddc_bus);
    verite_core_free(v->core);
    free(v->bios.rom);
    free(v->word_colors);
    free(v);
}

static const device_config_t verite_config[] = {
#ifdef VERITE_JIT_X64
    {
        .name        = "recompiler",
        .description = "Dynamic recompiler",
        .type        = CONFIG_BINARY,
        .default_int = 1
    },
#endif
    {
        .name        = "risc_thread",
        .description = "Run the RISC on its own thread",
        .type        = CONFIG_BINARY,
        .default_int = 1
    },
    { .name = "", .description = "", .type = CONFIG_END }
};

const device_t verite_v1000_device = {
    .name          = "Rendition Vérité V1000",
    .internal_name = "verite_v1000",
    .flags         = DEVICE_PCI,
    .init          = verite_init,
    .close         = verite_close,
    .reset         = verite_reset,
    .available     = verite_available,
    .speed_changed = verite_speed_changed,
    .force_redraw  = verite_redraw,
    .config        = verite_config
};
