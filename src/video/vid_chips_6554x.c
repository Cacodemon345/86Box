/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          Chips & Technologies 65545/65548 flat panel/CRT VGA controllers.
 *
 *          From the 65540/545 data sheet (rev. 1.2, 1995-10) and its VGA BIOS OEM
 *          reference guide (rev. 4.1). The 65548 is the 65545 core with its own
 *          PCI device ID (00DCh), a ROM base register and chip version DCh.
 *          Emulated: the VGA core with the extension (XR) registers, banked and
 *          linear mapping, the clock synthesizer, 8/15/16/24 bpp, the BitBLT engine
 *          (screen and system sources, colour and mono, 8x8 patterns, all 256
 *          ROPs) and the hardware cursor / pop-up (DR registers, by I/O and, on
 *          PCI, memory mapped). The display is the CRT's: flat panel timing,
 *          compensation and SmartMap are registers only, as are the CGA/Hercules
 *          emulation modes and the PC Video interface.
 *
 * Authors: 86Box contributors.
 *
 *          Copyright 2026 86Box contributors.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <wchar.h>
#define HAVE_STDARG_H
#include <86box/86box.h>
#include <86box/io.h>
#include <86box/mem.h>
#include <86box/pci.h>
#include <86box/rom.h>
#include <86box/device.h>
#include <86box/timer.h>
#include <86box/video.h>
#include <86box/vid_svga.h>
#include <86box/vid_svga_render.h>
#include <86box/vid_chips.h>
#include <86box/plat_unused.h>
#include "cpu.h"

#define BIOS_65545_PATH "roms/video/chips/ct65545_prox1330.bin"
#define BIOS_65548_PATH "roms/video/chips/ct65548_op47.bin"

enum {
    CHIPS_65545 = 0,
    CHIPS_65548
};

enum {
    CHIPS_BUS_ISA = 0,
    CHIPS_BUS_VLB,
    CHIPS_BUS_PCI
};

/* On-board chips: bit 16; bit 17 when the system BIOS holds the video BIOS. */
#define CHIPS_LOCAL(chip, bus, onboard, sysbios) ((chip) | ((bus) << 8) | ((onboard) << 16) | ((sysbios) << 17))

/* DR04, BitBLT control. */
#define BLT_INC_Y     (1 << 8)
#define BLT_INC_X     (1 << 9)
#define BLT_SRC_FG    (1 << 10)
#define BLT_SRC_MONO  (1 << 11)
#define BLT_PAT_MONO  (1 << 12)
#define BLT_BG_TRANS  (1 << 13)
#define BLT_SRC_SYS   (1 << 14)
#define BLT_PAT_SOLID (1 << 19)
#define BLT_BUSY      (1 << 20)

typedef struct chips_6554x_t {
    svga_t svga;

    rom_t         bios_rom;
    mem_mapping_t linear_mapping;
    mem_mapping_t mmio_mapping;

    int      chip;
    int      bus;
    int      on_board;
    int      has_rom;
    uint32_t vram_size;

    uint8_t xr_index;
    uint8_t xr[0x80];
    uint8_t vclk[3]; /* XR30-XR32 as loaded into the VCLK VCO */
    uint8_t mclk[3];

    uint32_t dr[16];
    uint16_t dr_base; /* I/O address of DR00 (XR07) */
    int      dr_io_on;

    uint8_t setup_ctrl;    /* 46E8h, ISA */
    uint8_t global_enable; /* 102h, ISA */
    uint8_t vse;           /* 3C3h, local bus */

    uint32_t bank[2];
    int      ext_map;
    int      dual_map;

    /* PCI configuration. */
    uint8_t  pci_slot;
    uint8_t  pci_cmd;
    uint8_t  pci_cmd_hi;
    uint8_t  int_line;
    uint32_t mbase;
    uint32_t iobase;
    uint32_t rom_bar;

    /* BitBLT engine: a system-to-screen transfer waiting for the CPU's data. */
    struct {
        int      active;
        uint32_t dst;    /* current line's first destination byte */
        int      skip;   /* first byte of the line in its doubleword stream */
        int      pos;    /* bytes of the line's stream received */
        int      need;   /* source bytes the line takes */
        int      total;  /* bytes the line's stream takes (whole doublewords) */
        int      line;
        int      lines;
        int      bpl;
        int      pat_row;
    } blt;

    uint32_t cursor_col[4];
} chips_6554x_t;

static video_timings_t timing_chips_6554x_isa = { .type = VIDEO_ISA, .write_b = 6, .write_w = 8, .write_l = 16, .read_b = 6, .read_w = 8, .read_l = 16 };
static video_timings_t timing_chips_6554x_vlb = { .type = VIDEO_BUS, .write_b = 4, .write_w = 4, .write_l = 8, .read_b = 10, .read_w = 10, .read_l = 20 };
static video_timings_t timing_chips_6554x_pci = { .type = VIDEO_PCI, .write_b = 4, .write_w = 4, .write_l = 8, .read_b = 10, .read_w = 10, .read_l = 20 };

#ifdef ENABLE_CHIPS_6554X_LOG
int chips_6554x_do_log = ENABLE_CHIPS_6554X_LOG;

static void
chips_6554x_log(const char *fmt, ...)
{
    va_list ap;

    if (chips_6554x_do_log) {
        va_start(ap, fmt);
        pclog_ex(fmt, ap);
        va_end(ap);
    }
}
#else
#    define chips_6554x_log(fmt, ...)
#endif

static void chips_6554x_dr_io_update(chips_6554x_t *chips);
static void chips_6554x_blt_host_byte(chips_6554x_t *chips, uint8_t val);

/* ---- Enables and mapping ---------------------------------------------------- */

static int
chips_6554x_io_enabled(const chips_6554x_t *chips)
{
    switch (chips->bus) {
        case CHIPS_BUS_PCI:
            return !!(chips->pci_cmd & PCI_COMMAND_IO);
        case CHIPS_BUS_VLB:
            return !!(chips->vse & 0x01);
        default:
            return (chips->setup_ctrl & 0x08) && (chips->global_enable & 0x01);
    }
}

static int
chips_6554x_mem_enabled(const chips_6554x_t *chips)
{
    if (chips->bus == CHIPS_BUS_PCI)
        return !!(chips->pci_cmd & PCI_COMMAND_MEM);

    return chips_6554x_io_enabled(chips);
}

static void
chips_6554x_update_linear(chips_6554x_t *chips)
{
    mem_mapping_disable(&chips->linear_mapping);
    mem_mapping_disable(&chips->mmio_mapping);

    if (!chips_6554x_mem_enabled(chips))
        return;

    if (chips->bus == CHIPS_BUS_PCI) {
        /* MBASE: an 8 MB space, display memory in the lower 2 MB, the DR registers
           memory mapped from 2 MB on (at the offsets of their I/O addresses). */
        if (chips->mbase) {
            mem_mapping_set_addr(&chips->linear_mapping, chips->mbase, 0x200000);
            mem_mapping_set_addr(&chips->mmio_mapping, chips->mbase + 0x200000, 0x10000);
        }
    } else if (chips->xr[0x0b] & 0x10) {
        /* XR08: bits 7-0 against A27-A20 (VL), bits 3-0 against A23-A20 (ISA). */
        uint32_t base = (chips->bus == CHIPS_BUS_ISA) ? ((chips->xr[0x08] & 0x0f) << 20) : (chips->xr[0x08] << 20);

        if (base)
            mem_mapping_set_addr(&chips->linear_mapping, base, 0x100000);
    }
}

static void
chips_6554x_update_enable(chips_6554x_t *chips)
{
    if (chips_6554x_mem_enabled(chips))
        mem_mapping_enable(&chips->svga.mapping);
    else
        mem_mapping_disable(&chips->svga.mapping);

    chips_6554x_update_linear(chips);
    chips_6554x_dr_io_update(chips);
}

static void
chips_6554x_update_mapping(chips_6554x_t *chips)
{
    svga_t *svga   = &chips->svga;
    int     packed = !!(chips->xr[0x0b] & 0x04);

    chips->ext_map  = 0;
    chips->dual_map = 0;
    chips->bank[0] = chips->bank[1] = 0;

    /* Banks are kept in display memory (plane) address units, bits 17-10;
       chips_6554x_decode() scales them to CPU bytes in chain-4 mode. */
    if (chips->xr[0x0b] & 0x01) {
        chips->ext_map = 1;
        chips->bank[0] = chips->xr[0x10] << 10;
        chips->bank[1] = chips->xr[0x11] << 10;

        if ((chips->xr[0x0b] & 0x02) && (((svga->gdcreg[6] >> 2) & 3) <= 1))
            chips->dual_map = 1;
    }

    if (svga->packed_chain4 != packed) {
        svga->packed_chain4 = packed;
        svga->fast          = (svga->gdcreg[8] == 0xff && !(svga->gdcreg[3] & 0x18) && !svga->gdcreg[1]) &&
                              ((svga->chain4 && (svga->packed_chain4 || svga->force_old_addr)) || svga->fb_only);
        svga->fullchange    = changeframecount;
        svga_recalctimings(svga);
    }
}

static uint32_t
chips_6554x_decode(const chips_6554x_t *chips, uint32_t addr)
{
    const svga_t *svga = &chips->svga;
    uint32_t      half;
    const int     shift = svga->chain4 ? 2 : 0;

    addr &= 0x1ffff;

    switch ((svga->gdcreg[6] >> 2) & 3) {
        case 0:
            half = 0x10000;
            break;
        case 1:
            if (addr >= 0x10000)
                return 0xffffffff;
            half = 0x8000;
            break;
        case 2:
            addr -= 0x10000;
            if (addr >= 0x8000)
                return 0xffffffff;
            half = 0;
            break;
        default:
            addr -= 0x18000;
            if (addr >= 0x8000)
                return 0xffffffff;
            half = 0;
            break;
    }

    if (chips->dual_map && half && (addr >= half))
        addr = (addr - half) + (chips->bank[1] << shift);
    else
        addr += chips->bank[0] << shift;

    return addr & (svga->vram_mask >> (2 - shift));
}

/* ---- Memory ----------------------------------------------------------------- */

static void
chips_6554x_write(uint32_t addr, uint8_t val, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (chips->blt.active) {
        chips_6554x_blt_host_byte(chips, val);
        return;
    }

    if (!chips->ext_map) {
        svga_write(addr, val, &chips->svga);
        return;
    }

    addr = chips_6554x_decode(chips, addr);
    if (addr != 0xffffffff)
        svga_write_linear(addr, val, &chips->svga);
}

static void
chips_6554x_writew(uint32_t addr, uint16_t val, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (chips->blt.active) {
        chips_6554x_blt_host_byte(chips, val & 0xff);
        chips_6554x_blt_host_byte(chips, val >> 8);
        return;
    }

    if (!chips->ext_map) {
        svga_writew(addr, val, &chips->svga);
        return;
    }

    addr = chips_6554x_decode(chips, addr);
    if (addr != 0xffffffff)
        svga_writew_linear(addr, val, &chips->svga);
}

static void
chips_6554x_writel(uint32_t addr, uint32_t val, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (chips->blt.active) {
        for (int i = 0; i < 4; i++)
            chips_6554x_blt_host_byte(chips, (val >> (i << 3)) & 0xff);
        return;
    }

    if (!chips->ext_map) {
        svga_writel(addr, val, &chips->svga);
        return;
    }

    addr = chips_6554x_decode(chips, addr);
    if (addr != 0xffffffff)
        svga_writel_linear(addr, val, &chips->svga);
}

static uint8_t
chips_6554x_read(uint32_t addr, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (!chips->ext_map)
        return svga_read(addr, &chips->svga);

    addr = chips_6554x_decode(chips, addr);
    if (addr == 0xffffffff)
        return 0xff;

    return svga_read_linear(addr, &chips->svga);
}

static uint16_t
chips_6554x_readw(uint32_t addr, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (!chips->ext_map)
        return svga_readw(addr, &chips->svga);

    addr = chips_6554x_decode(chips, addr);
    if (addr == 0xffffffff)
        return 0xffff;

    return svga_readw_linear(addr, &chips->svga);
}

static uint32_t
chips_6554x_readl(uint32_t addr, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (!chips->ext_map)
        return svga_readl(addr, &chips->svga);

    addr = chips_6554x_decode(chips, addr);
    if (addr == 0xffffffff)
        return 0xffffffff;

    return svga_readl_linear(addr, &chips->svga);
}

/* The linear window: 1 MB (ISA/VL) or the lower 2 MB of MBASE (PCI), wrapping
   around the installed memory. */
static void
chips_6554x_lin_write(uint32_t addr, uint8_t val, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (chips->blt.active) {
        chips_6554x_blt_host_byte(chips, val);
        return;
    }

    svga_write_linear(addr & chips->svga.vram_mask, val, &chips->svga);
}

static void
chips_6554x_lin_writew(uint32_t addr, uint16_t val, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (chips->blt.active) {
        chips_6554x_blt_host_byte(chips, val & 0xff);
        chips_6554x_blt_host_byte(chips, val >> 8);
        return;
    }

    svga_writew_linear(addr & chips->svga.vram_mask, val, &chips->svga);
}

static void
chips_6554x_lin_writel(uint32_t addr, uint32_t val, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (chips->blt.active) {
        for (int i = 0; i < 4; i++)
            chips_6554x_blt_host_byte(chips, (val >> (i << 3)) & 0xff);
        return;
    }

    svga_writel_linear(addr & chips->svga.vram_mask, val, &chips->svga);
}

static uint8_t
chips_6554x_lin_read(uint32_t addr, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    return svga_read_linear(addr & chips->svga.vram_mask, &chips->svga);
}

static uint16_t
chips_6554x_lin_readw(uint32_t addr, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    return svga_readw_linear(addr & chips->svga.vram_mask, &chips->svga);
}

static uint32_t
chips_6554x_lin_readl(uint32_t addr, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    return svga_readl_linear(addr & chips->svga.vram_mask, &chips->svga);
}

/* ---- BitBLT engine ------------------------------------------------------------ */

static uint8_t
chips_6554x_rop3(uint8_t rop, uint8_t p, uint8_t s, uint8_t d)
{
    uint8_t out = 0x00;

    /* Bit n of a Windows ROP3 is the result for P = n.2, S = n.1, D = n.0. */
    for (int n = 0; n < 8; n++) {
        if (rop & (1 << n))
            out |= ((n & 4) ? p : (uint8_t) ~p) & ((n & 2) ? s : (uint8_t) ~s) & ((n & 1) ? d : (uint8_t) ~d);
    }

    return out;
}

static int
chips_6554x_blt_pixbytes(const chips_6554x_t *chips)
{
    /* XR40: 01 = 8 bpp, 10 = 16 bpp; 24 bpp is drawn as 8 bpp. */
    return ((chips->xr[0x40] & 0x03) == 0x02) ? 2 : 1;
}

static uint8_t
chips_6554x_blt_color(const chips_6554x_t *chips, int reg, int lane)
{
    return (chips->dr[reg] >> (lane << 3)) & 0xff;
}

/* The pattern byte at a destination byte; *mono gets the mono pattern's bit. */
static uint8_t
chips_6554x_blt_pattern(const chips_6554x_t *chips, uint32_t dst, int row, int *mono)
{
    const svga_t  *svga = &chips->svga;
    const uint32_t ctrl = chips->dr[4];
    const int      pixb = chips_6554x_blt_pixbytes(chips);
    const int      lane = dst % pixb;
    const int      px   = (dst / pixb) & 7;
    uint32_t       base;

    *mono = 1;

    /* A solid colour pattern repeats the pattern's first line: the Windows drivers
       store one line of the brush colour there (8 pixels) and leave DR02/DR03
       alone. A solid mono pattern is all background (DR02), as the drivers'
       fills with it expect. */
    if (ctrl & BLT_PAT_SOLID) {
        if (ctrl & BLT_PAT_MONO)
            return chips_6554x_blt_color(chips, 2, lane);
        row = 0;
    }

    if (ctrl & BLT_PAT_MONO) {
        base  = chips->dr[1] & ~7;
        *mono = (svga->vram[(base + row) & svga->vram_mask] >> (7 - px)) & 1;
        return chips_6554x_blt_color(chips, *mono ? 3 : 2, lane);
    }

    base = chips->dr[1] & ~((64 * pixb) - 1);
    return svga->vram[(base + (((row << 3) + px) * pixb) + lane) & svga->vram_mask];
}

/* Draws one destination byte from source byte s; mono is the source's (or the
   pattern's) mono bit for background transparency, 1 when there is none. */
static void
chips_6554x_blt_put(chips_6554x_t *chips, uint32_t dst, uint8_t s, int smono, int row)
{
    svga_t        *svga = &chips->svga;
    const uint32_t ctrl = chips->dr[4];
    int            pmono;
    uint8_t        p;

    dst &= svga->vram_mask;
    p = chips_6554x_blt_pattern(chips, dst, row, &pmono);

    if (ctrl & BLT_BG_TRANS) {
        if ((ctrl & BLT_SRC_MONO) ? !smono : ((ctrl & BLT_PAT_MONO) && !(ctrl & BLT_PAT_SOLID) && !pmono))
            return;
    }

    svga->vram[dst]              = chips_6554x_rop3(chips->dr[4] & 0xff, p, s, svga->vram[dst]);
    svga->changedvram[dst >> 12] = changeframecount;
}

static void
chips_6554x_blt_line_setup(chips_6554x_t *chips)
{
    const int pixb = chips_6554x_blt_pixbytes(chips);

    chips->blt.pos = 0;
    if (chips->dr[4] & BLT_SRC_MONO)
        chips->blt.need = ((chips->blt.bpl / pixb) + 7) >> 3;
    else
        chips->blt.need = chips->blt.bpl;
    chips->blt.total = (chips->blt.skip + chips->blt.need + 3) & ~3;
}

static void
chips_6554x_blt_next_line(chips_6554x_t *chips)
{
    const int up     = !(chips->dr[4] & BLT_INC_Y);
    const int dstoff = (chips->dr[0] >> 16) & 0xfff;

    chips->blt.dst     = up ? (chips->blt.dst - dstoff) : (chips->blt.dst + dstoff);
    chips->blt.pat_row = (chips->blt.pat_row + (up ? 7 : 1)) & 7;
    chips->blt.line++;
}

/* One byte of the CPU's source data for a system-to-screen BitBLT. */
static void
chips_6554x_blt_host_byte(chips_6554x_t *chips, uint8_t val)
{
    const uint32_t ctrl = chips->dr[4];
    const int      pixb = chips_6554x_blt_pixbytes(chips);
    const int      j    = chips->blt.pos - chips->blt.skip;

    if ((j >= 0) && (j < chips->blt.need)) {
        if (ctrl & BLT_SRC_MONO) {
            const int width = chips->blt.bpl / pixb;

            for (int b = 0; b < 8; b++) {
                const int px  = (j << 3) + b;
                const int bit = (val >> (7 - b)) & 1;

                if (px >= width)
                    break;
                for (int lane = 0; lane < pixb; lane++)
                    chips_6554x_blt_put(chips, chips->blt.dst + (px * pixb) + lane,
                                        chips_6554x_blt_color(chips, bit ? 3 : 2, lane), bit, chips->blt.pat_row);
            }
        } else {
            const uint32_t dst = (ctrl & BLT_INC_X) ? (chips->blt.dst + j) : (chips->blt.dst - j);

            chips_6554x_blt_put(chips, dst, val, 1, chips->blt.pat_row);
        }
    }

    if (++chips->blt.pos >= chips->blt.total) {
        /* The rest of the line's last doubleword was discarded; the next line
           starts at the source offset's alignment. */
        chips->blt.skip = (chips->blt.skip + (chips->dr[0] & 0xfff)) & 3;
        chips_6554x_blt_next_line(chips);
        if (chips->blt.line >= chips->blt.lines) {
            chips->blt.active = 0;
            chips->dr[4] &= ~BLT_BUSY;
        } else
            chips_6554x_blt_line_setup(chips);
    }
}

static void
chips_6554x_blt_start(chips_6554x_t *chips)
{
    svga_t        *svga   = &chips->svga;
    const uint32_t ctrl   = chips->dr[4];
    const int      pixb   = chips_6554x_blt_pixbytes(chips);
    const int      inc_x  = !!(ctrl & BLT_INC_X);
    const int      up     = !(ctrl & BLT_INC_Y);
    const int      srcoff = chips->dr[0] & 0xfff;
    const int      bpl    = chips->dr[7] & 0xfff;
    const int      lines  = (chips->dr[7] >> 16) & 0xfff;
    uint32_t       src    = chips->dr[5] & 0x1fffff;
    uint32_t       dst    = chips->dr[6] & 0x1fffff;
    int            row    = (ctrl >> 16) & 7;

    chips_6554x_log("C&T 6554x: BitBLT ctrl %08X src %06X dst %06X off %08X %dx%d pat %06X fg %04X bg %04X\n",
                    ctrl, src, dst, chips->dr[0], bpl, lines, chips->dr[1], chips->dr[3] & 0xffff, chips->dr[2] & 0xffff);

    if (chips->blt.active)
        chips_6554x_log("C&T 6554x: host BitBLT cut short at line %d/%d, byte %d/%d\n",
                        chips->blt.line, chips->blt.lines, chips->blt.pos, chips->blt.total);
    chips->blt.active = 0;

    if (!bpl || !lines)
        return;

    if (((ctrl >> 14) & 3) == 1 && !(ctrl & BLT_SRC_FG)) {
        /* System to screen: the CPU writes the source to display memory. */
        chips->blt.active  = 1;
        chips->blt.dst     = dst;
        chips->blt.skip    = src & 3;
        chips->blt.line    = 0;
        chips->blt.lines   = lines;
        chips->blt.bpl     = bpl;
        chips->blt.pat_row = row;
        chips_6554x_blt_line_setup(chips);
        chips->dr[4] |= BLT_BUSY;
        return;
    }

    for (int y = 0; y < lines; y++) {
        for (int k = 0; k < bpl; k++) {
            const uint32_t d = inc_x ? (dst + k) : (dst - k);
            int            smono = 1;
            uint8_t        s;

            if (ctrl & BLT_SRC_FG)
                s = chips_6554x_blt_color(chips, 3, d % pixb);
            else if (ctrl & BLT_SRC_MONO) {
                /* Font expansion from display memory, left to right only. */
                const int px = k / pixb;

                smono = (svga->vram[(src + (px >> 3)) & svga->vram_mask] >> (7 - (px & 7))) & 1;
                s     = chips_6554x_blt_color(chips, smono ? 3 : 2, k % pixb);
            } else
                s = svga->vram[(inc_x ? (src + k) : (src - k)) & svga->vram_mask];

            chips_6554x_blt_put(chips, d, s, smono, row);
        }

        src = up ? (src - srcoff) : (src + srcoff);
        dst = up ? (dst - ((chips->dr[0] >> 16) & 0xfff)) : (dst + ((chips->dr[0] >> 16) & 0xfff));
        row = (row + (up ? 7 : 1)) & 7;
    }
}

/* ---- Hardware cursor ------------------------------------------------------------ */

static void
chips_6554x_cursor_colors(chips_6554x_t *chips)
{
    chips->cursor_col[0] = video_16to32[chips->dr[9] & 0xffff];
    chips->cursor_col[1] = video_16to32[chips->dr[9] >> 16];
    chips->cursor_col[2] = video_16to32[chips->dr[10] & 0xffff];
    chips->cursor_col[3] = video_16to32[chips->dr[10] >> 16];
}

static void
chips_6554x_cursor_geometry(const chips_6554x_t *chips, int *width, int *height, int *pitch, int *two_bpp)
{
    const uint32_t ctrl = chips->dr[8];

    *two_bpp = 1;
    switch (ctrl & 3) {
        case 1:
            *width = *height = 32;
            *pitch = 8;
            break;
        case 2:
            *width = *height = 64;
            *pitch = 16;
            break;
        default:
            if (ctrl & 0x100) {
                *width = *height = 64;
                *pitch = 16;
            } else {
                /* 128x128 one bit per pixel pop-up. */
                *width = *height = 128;
                *pitch   = 16;
                *two_bpp = 0;
            }
            break;
    }
}

static void
chips_6554x_update_cursor(chips_6554x_t *chips)
{
    svga_t  *svga = &chips->svga;
    int      width;
    int      height;
    int      pitch;
    int      two_bpp;
    int      x    = chips->dr[11] & 0x7ff;
    int      y    = (chips->dr[11] >> 16) & 0x7ff;

    chips_6554x_cursor_geometry(chips, &width, &height, &pitch, &two_bpp);

    if (chips->dr[11] & 0x8000)
        x = -x;
    if (chips->dr[11] & 0x80000000)
        y = -y;

    svga->hwcursor.ena       = !!(chips->dr[8] & 3);
    svga->hwcursor.cur_xsize = width;
    svga->hwcursor.cur_ysize = height;
    svga->hwcursor.x         = x;
    svga->hwcursor.xoff      = 0;
    svga->hwcursor.addr      = chips->dr[12] & 0xffc00;
    if (y < 0) {
        svga->hwcursor.yoff = MIN(-y, height);
        svga->hwcursor.addr += svga->hwcursor.yoff * pitch;
        y = 0;
    } else
        svga->hwcursor.yoff = 0;
    svga->hwcursor.y = y;
}

static void
chips_6554x_hwcursor_draw(svga_t *svga, int displine)
{
    chips_6554x_t *chips = (chips_6554x_t *) svga->priv;
    const int      type  = chips->dr[8] & 3;
    const int      zoom  = ((type == 2) && (chips->dr[8] & 0x200) && !svga->interlace) ? 2 : 1;
    const uint8_t *src   = &svga->vram[svga->hwcursor_latch.addr & svga->vram_mask];
    uint32_t      *line  = svga->monitor->target_buffer->line[displine & 2047];
    int            width;
    int            height;
    int            pitch;
    int            two_bpp;

    chips_6554x_cursor_geometry(chips, &width, &height, &pitch, &two_bpp);

    for (int px = 0; px < width; px++) {
        int idx;

        if (two_bpp) {
            /* Each byte column is an AND (plane 0/2) and XOR (plane 1/3) byte pair. */
            const int a = (src[(px >> 3) << 1] >> (7 - (px & 7))) & 1;
            const int x = (src[((px >> 3) << 1) + 1] >> (7 - (px & 7))) & 1;

            idx = (a << 1) | x;
        } else
            idx = (src[px >> 3] >> (7 - (px & 7))) & 1;

        for (int z = 0; z < zoom; z++) {
            const int ox = svga->hwcursor_latch.x + (px * zoom) + z;
            uint32_t *p;

            if ((ox < 0) || (ox >= svga->hdisp))
                continue;

            p = &line[(ox + svga->x_add) & 2047];
            if (type != 3) {
                /* AND/XOR: 00 colour 0, 01 colour 1, 10 transparent, 11 inverted. */
                if (idx == 3)
                    *p ^= 0xffffff;
                else if (idx != 2)
                    *p = chips->cursor_col[idx];
            } else
                *p = chips->cursor_col[idx];
        }
    }

    svga->hwcursor_latch.addr += pitch;
}

/* ---- 32-bit (DR) registers ------------------------------------------------------- */

static uint8_t
chips_6554x_dr_readb(chips_6554x_t *chips, int idx, int byte)
{
    uint32_t val;

    if (idx > 12)
        return 0xff;

    val = chips->dr[idx];
    switch (idx) {
        case 2:
        case 3:
            /* Only bits 15-0 are kept; they read back duplicated. */
            val = (val & 0xffff) | (val << 16);
            break;
        case 4:
            /* The queue is never full. */
            val = (val & 0x001fffff) | (0x0f << 24);
            break;
        default:
            break;
    }

    return (val >> (byte << 3)) & 0xff;
}

static void
chips_6554x_dr_writeb(chips_6554x_t *chips, int idx, int byte, uint8_t val)
{
    static const uint32_t masks[13] = {
        0x0fff0fff, 0x001fffff, 0x0000ffff, 0x0000ffff, 0x000fffff, 0x001fffff, 0x001fffff,
        0x0fff0fff, 0x0000ffff, 0xffffffff, 0xffffffff, 0x87ff87ff, 0x000ffc00
    };
    const uint32_t shift = byte << 3;
    uint32_t       mask;

    if (idx > 12)
        return;

    mask = masks[idx] & (0xffu << shift);
    if ((idx == 4) && chips->blt.active)
        chips->dr[idx] = (chips->dr[idx] & ~mask) | ((val << shift) & mask) | BLT_BUSY;
    else
        chips->dr[idx] = (chips->dr[idx] & ~mask) | ((val << shift) & mask);

    switch (idx) {
        case 7:
            /* Writing the command register's high byte (the line count) starts it. */
            if (byte == 3)
                chips_6554x_blt_start(chips);
            break;
        case 8:
        case 11:
        case 12:
            chips_6554x_update_cursor(chips);
            break;
        case 9:
        case 10:
            chips_6554x_cursor_colors(chips);
            break;
        default:
            break;
    }
}

static uint8_t
chips_6554x_dr_in(uint16_t addr, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    return chips_6554x_dr_readb(chips, (addr >> 10) & 0x1f, addr & 3);
}

static uint16_t
chips_6554x_dr_inw(uint16_t addr, void *priv)
{
    return chips_6554x_dr_in(addr, priv) | (chips_6554x_dr_in(addr + 1, priv) << 8);
}

static uint32_t
chips_6554x_dr_inl(uint16_t addr, void *priv)
{
    return chips_6554x_dr_inw(addr, priv) | (chips_6554x_dr_inw(addr + 2, priv) << 16);
}

static void
chips_6554x_dr_out(uint16_t addr, uint8_t val, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    chips_6554x_dr_writeb(chips, (addr >> 10) & 0x1f, addr & 3, val);
}

static void
chips_6554x_dr_outw(uint16_t addr, uint16_t val, void *priv)
{
    chips_6554x_dr_out(addr, val & 0xff, priv);
    chips_6554x_dr_out(addr + 1, val >> 8, priv);
}

static void
chips_6554x_dr_outl(uint16_t addr, uint32_t val, void *priv)
{
    chips_6554x_dr_outw(addr, val & 0xffff, priv);
    chips_6554x_dr_outw(addr + 2, val >> 16, priv);
}

/*
   The DR registers sit at XR07's I/O addresses (A15 and A8-A2 from XR07, A9 = 1,
   A14-A10 the register) once XR03 bit 1 enables them. On PCI the data sheet
   names an I/O BAR instead, but the 6554x Windows drivers use the XR07 ports
   there too (or the memory mapped copy), so both decode the same.
*/
static void
chips_6554x_dr_io_update(chips_6554x_t *chips)
{
    const uint8_t xr07 = chips->xr[0x07];
    const int     on   = (chips->xr[0x03] & 0x02) && chips_6554x_io_enabled(chips);
    uint16_t      base = ((xr07 & 0x80) << 8) | ((xr07 & 0x7f) << 2) | 0x200;

    if (chips->dr_io_on) {
        for (int i = 0; i < 13; i++)
            io_removehandler(chips->dr_base | (i << 10), 4,
                             chips_6554x_dr_in, chips_6554x_dr_inw, chips_6554x_dr_inl,
                             chips_6554x_dr_out, chips_6554x_dr_outw, chips_6554x_dr_outl, chips);
        chips->dr_io_on = 0;
    }

    if (on) {
        chips->dr_base = base;
        for (int i = 0; i < 13; i++)
            io_sethandler(base | (i << 10), 4,
                          chips_6554x_dr_in, chips_6554x_dr_inw, chips_6554x_dr_inl,
                          chips_6554x_dr_out, chips_6554x_dr_outw, chips_6554x_dr_outl, chips);
        chips->dr_io_on = 1;
    }
}

static int
chips_6554x_mmio_reg(uint32_t addr)
{
    addr &= 0xffff;

    if (!(addr & 0x8000) || ((addr & 0x3fc) != 0x3d0))
        return -1;

    return (addr >> 10) & 0x1f;
}

static uint8_t
chips_6554x_mmio_read(uint32_t addr, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;
    const int      idx   = chips_6554x_mmio_reg(addr);

    if (idx < 0)
        return 0xff;

    return chips_6554x_dr_readb(chips, idx, addr & 3);
}

static uint16_t
chips_6554x_mmio_readw(uint32_t addr, void *priv)
{
    return chips_6554x_mmio_read(addr, priv) | (chips_6554x_mmio_read(addr + 1, priv) << 8);
}

static uint32_t
chips_6554x_mmio_readl(uint32_t addr, void *priv)
{
    return chips_6554x_mmio_readw(addr, priv) | (chips_6554x_mmio_readw(addr + 2, priv) << 16);
}

static void
chips_6554x_mmio_write(uint32_t addr, uint8_t val, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;
    const int      idx   = chips_6554x_mmio_reg(addr);

    if (idx >= 0)
        chips_6554x_dr_writeb(chips, idx, addr & 3, val);
}

static void
chips_6554x_mmio_writew(uint32_t addr, uint16_t val, void *priv)
{
    chips_6554x_mmio_write(addr, val & 0xff, priv);
    chips_6554x_mmio_write(addr + 1, val >> 8, priv);
}

static void
chips_6554x_mmio_writel(uint32_t addr, uint32_t val, void *priv)
{
    chips_6554x_mmio_writew(addr, val & 0xffff, priv);
    chips_6554x_mmio_writew(addr + 2, val >> 16, priv);
}

/* ---- Extension (XR) registers ------------------------------------------------------ */

static uint8_t
chips_6554x_xr_read(chips_6554x_t *chips)
{
    const svga_t *svga = &chips->svga;
    uint8_t       ret;

    switch (chips->xr_index) {
        case 0x00:
            /* Chip version: 65545 D8h (first silicon step), 65548 DCh. */
            ret = (chips->chip == CHIPS_65548) ? 0xdc : 0xd8;
            break;
        case 0x01:
            /* Configuration straps: normal clocks (CFG7), A26/A27 off, internal
               oscillator, CFG4 high, bus type in bits 2-0. */
            ret = 0xb8 | ((chips->bus == CHIPS_BUS_PCI) ? 0x06 : ((chips->bus == CHIPS_BUS_VLB) ? 0x07 : 0x05));
            break;
        case 0x02:
            ret = (chips->xr[0x02] & 0x7f) | (svga->attrff ? 0x80 : 0x00);
            break;
        case 0x14:
            ret = chips->xr[0x14] & ~0x0c;
            break;
        default:
            ret = chips->xr[chips->xr_index];
            break;
    }

    return ret;
}

static void
chips_6554x_xr_write(chips_6554x_t *chips, uint8_t val)
{
    svga_t       *svga = &chips->svga;
    const uint8_t idx  = chips->xr_index;
    const uint8_t old  = chips->xr[idx];

    switch (idx) {
        case 0x00:
        case 0x01:
            return;
        default:
            break;
    }

    chips->xr[idx] = val;
    chips_6554x_log("C&T 6554x: XR%02X = %02X\n", idx, val);

    switch (idx) {
        case 0x03:
        case 0x07:
            chips_6554x_dr_io_update(chips);
            break;
        case 0x04:
            /* Bit 2: CRTC address counter bits 16-17. */
            svga->vram_display_mask = (val & 0x04) ? svga->vram_mask : MIN(svga->vram_mask, 0x3ffff);
            if (old != val)
                svga_recalctimings(svga);
            break;
        case 0x08:
            chips_6554x_update_linear(chips);
            break;
        case 0x0b:
            chips_6554x_update_mapping(chips);
            chips_6554x_update_linear(chips);
            break;
        case 0x10:
        case 0x11:
            chips_6554x_update_mapping(chips);
            break;
        case 0x32:
            /* The completed write loads XR30-XR32 into the VCO XR33 bit 5 points at. */
            if (chips->xr[0x33] & 0x20)
                memcpy(chips->mclk, &chips->xr[0x30], 3);
            else
                memcpy(chips->vclk, &chips->xr[0x30], 3);
            svga_recalctimings(svga);
            break;
        case 0x06:
        case 0x0c:
        case 0x0d:
        case 0x16:
        case 0x17:
        case 0x28:
        case 0x33:
            if (old != val) {
                svga->fullchange = changeframecount;
                svga_recalctimings(svga);
            }
            break;
        default:
            break;
    }
}

/* ---- I/O ----------------------------------------------------------------------- */

static void
chips_6554x_out(uint16_t addr, uint8_t val, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;
    svga_t        *svga  = &chips->svga;
    uint8_t        old;
    uint8_t        wmask;

    if (!chips_6554x_io_enabled(chips))
        return;

    if ((addr == 0x3d6) || (addr == 0x3d7) || (addr == 0x3b6) || (addr == 0x3b7)) {
        /* XR08 is also at 3B7h; the rest of the extension registers only at 3D6h/3D7h. */
        if (addr & 1) {
            if ((addr == 0x3d7) || (chips->xr_index == 0x08))
                chips_6554x_xr_write(chips, val);
        } else if (addr == 0x3d6)
            chips->xr_index = val & 0x7f;
        return;
    }

    if ((((addr & 0xfff0) == 0x3d0) || ((addr & 0xfff0) == 0x3b0)) && !(svga->miscout & 1))
        addr ^= 0x60;

    switch (addr) {
        case 0x3c0:
            if (chips->xr[0x15] & 0x01)
                return;
            if (((chips->xr[0x02] >> 3) & 3) == 1) {
                /* 16-bit mapping: the index at 3C0h, the flip-flop always reset. */
                svga->attrff = 0;
                svga_out(addr, val, svga);
                svga->attrff = 0;
                return;
            }
            break;
        case 0x3c1:
            if (chips->xr[0x15] & 0x01)
                return;
            if (((chips->xr[0x02] >> 3) & 3) == 1) {
                svga->attrff = 1;
                svga_out(0x3c0, val, svga);
                svga->attrff = 0;
                return;
            }
            break;

        case 0x3c2:
            if (chips->xr[0x15] & 0x10)
                return;
            svga_out(addr, val, svga);
            svga_recalctimings(svga);
            return;

        case 0x3c3:
            /* Video subsystem enable, local bus only (XR70 bit 7 locks it). */
            if ((chips->bus == CHIPS_BUS_VLB) && !(chips->xr[0x70] & 0x80)) {
                chips->vse = val & 0x01;
                chips_6554x_update_enable(chips);
            }
            return;

        case 0x3c5:
            if (chips->xr[0x15] & 0x01)
                return;
            svga_out(addr, val, svga);
            if (svga->seqaddr == 4)
                chips_6554x_update_mapping(chips);
            return;

        case 0x3c6:
        case 0x3c7:
        case 0x3c8:
        case 0x3c9:
            if (chips->xr[0x15] & 0x20)
                return;
            break;

        case 0x3cf:
            if (chips->xr[0x15] & 0x01)
                return;
            svga_out(addr, val, svga);
            if ((svga->gdcaddr & 0x0f) == 6)
                chips_6554x_update_mapping(chips);
            return;

        case 0x3da:
            if (chips->xr[0x15] & 0x10)
                return;
            svga->fcr = val;
            return;

        case 0x3d4:
            svga->crtcreg = val & 0x3f;
            return;

        case 0x3d5:
            if (svga->crtcreg & 0x20)
                return;
            old                       = svga->crtc[svga->crtcreg];
            wmask                     = chips_crtc_wmask(chips->xr[0x15], svga->crtc[0x11], svga->crtcreg);
            svga->crtc[svga->crtcreg] = (old & ~wmask) | (val & wmask);
            val                       = svga->crtc[svga->crtcreg];
            if (old != val) {
                if ((svga->crtcreg < 0x0e) || (svga->crtcreg > 0x10)) {
                    if ((svga->crtcreg == 0x0c) || (svga->crtcreg == 0x0d)) {
                        svga->fullchange    = 3;
                        svga->memaddr_latch = ((svga->crtc[0xc] << 8) | svga->crtc[0xd]) + ((svga->crtc[8] & 0x60) >> 5);
                        svga->memaddr_latch |= (chips->xr[0x0c] & 0x03) << 16;
                    } else {
                        svga->fullchange = changeframecount;
                        svga_recalctimings(svga);
                    }
                }
            }
            return;

        default:
            break;
    }

    svga_out(addr, val, svga);
}

static uint8_t
chips_6554x_in(uint16_t addr, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;
    svga_t        *svga  = &chips->svga;
    uint8_t        ret;
    uint8_t        sw;

    if (!chips_6554x_io_enabled(chips))
        return 0xff;

    if ((addr == 0x3d6) || (addr == 0x3d7) || (addr == 0x3b6) || (addr == 0x3b7)) {
        if (addr == 0x3d6)
            return chips->xr_index;
        if ((addr == 0x3d7) || (chips->xr_index == 0x08))
            return chips_6554x_xr_read(chips);
        return 0xff;
    }

    if ((((addr & 0xfff0) == 0x3d0) || ((addr & 0xfff0) == 0x3b0)) && !(svga->miscout & 1))
        addr ^= 0x60;

    switch (addr) {
        case 0x3c2:
            ret = svga_in(addr, svga);
            if (chips->xr[0x1f] & 0x80) {
                /* Virtual switch readback in place of the SENSE comparator. */
                sw  = 3 - ((svga->miscout >> 2) & 3);
                ret = (ret & ~0x10) | (((chips->xr[0x1f] >> sw) & 1) << 4);
            }
            break;

        case 0x3c3:
            ret = 0xff;
            break;

        case 0x3da:
            ret = svga_in(addr, svga);
            /* XR14 bit 6: vertical sync status on bit 7, as on an MDA/Hercules. */
            if (chips->xr[0x14] & 0x40)
                ret = (ret & 0x7f) | ((ret & 0x08) ? 0x80 : 0x00);
            break;

        case 0x3d4:
            ret = svga->crtcreg;
            break;

        case 0x3d5:
            if (svga->crtcreg & 0x20)
                ret = 0xff;
            else
                ret = svga->crtc[svga->crtcreg];
            break;

        default:
            ret = svga_in(addr, svga);
            break;
    }

    return ret;
}

static void
chips_6554x_dac_out(uint16_t addr, uint8_t val, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (chips->xr[0x02] & 0x40)
        chips_6554x_out((addr & 0x000f) | 0x03c0, val, priv);
}

static uint8_t
chips_6554x_dac_in(uint16_t addr, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (chips->xr[0x02] & 0x40)
        return chips_6554x_in((addr & 0x000f) | 0x03c0, priv);

    return 0xff;
}

static uint8_t
chips_6554x_setup_in(uint16_t addr, void *priv)
{
    const chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (!(chips->setup_ctrl & 0x10))
        return 0xff;

    switch (addr) {
        case 0x102:
            return chips->global_enable;
        default:
            return 0xff;
    }
}

static void
chips_6554x_setup_out(uint16_t addr, uint8_t val, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (!(chips->setup_ctrl & 0x10))
        return;

    if (addr == 0x102) {
        chips->global_enable = val & 0x01;
        chips_6554x_update_enable(chips);
    }
}

static void
chips_6554x_46e8_out(UNUSED(uint16_t addr), uint8_t val, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    if (chips->xr[0x70] & 0x80)
        return;

    chips->setup_ctrl = val & 0x18;
    chips_6554x_update_enable(chips);
}

/* ---- Timing ---------------------------------------------------------------------- */

static double
chips_6554x_vco(const uint8_t *regs)
{
    /* Fout = Fref * 4 * M / (PSN * N * 2^P); M and N are programmed less 2. */
    const double m   = (regs[1] & 0x7f) + 2;
    const double n   = (regs[2] & 0x7f) + 2;
    const double psn = (regs[0] & 0x01) ? 1.0 : 4.0;
    const int    p   = (regs[0] >> 1) & 7;

    return (14318181.0 * 4.0 * m) / (psn * n * (double) (1 << p));
}

static double
chips_6554x_getclock(const chips_6554x_t *chips)
{
    const svga_t *svga = &chips->svga;
    double        freq;

    if (chips->xr[0x33] & 0x10)
        return chips_6554x_vco(chips->mclk) / 2.0;

    switch ((svga->miscout >> 2) & 3) {
        case 0:
            freq = (chips->xr[0x33] & 0x80) ? 31500000.0 : 25175000.0;
            break;
        case 1:
            freq = (chips->xr[0x33] & 0x80) ? 35500000.0 : 28322000.0;
            break;
        default:
            freq = chips_6554x_vco(chips->vclk);
            break;
    }

    return freq;
}

static void
chips_6554x_recalctimings(svga_t *svga)
{
    chips_6554x_t *chips = (chips_6554x_t *) svga->priv;
    const uint8_t  xr16  = chips->xr[0x16];
    const uint8_t  xr17  = chips->xr[0x17];
    const int      graphics = (svga->gdcreg[6] & 1) || (svga->attrregs[0x10] & 1);
    double         freq  = chips_6554x_getclock(chips);

    if (freq < 1000000.0)
        freq = 25175000.0;
    svga->clock = (cpuclock * (double) (1ULL << 32)) / freq;

    svga->memaddr_latch |= (chips->xr[0x0c] & 0x03) << 16;

    /* Bit 10 of the vertical, bit 8 of the horizontal parameters. */
    if (xr16 & 0x01)
        svga->vtotal += 0x400;
    if (xr16 & 0x02)
        svga->dispend += 0x400;
    if (xr16 & 0x04)
        svga->vsyncstart += 0x400;
    if (xr16 & 0x10)
        svga->vblankstart += 0x400;
    if (xr16 & 0x40)
        svga->split += 0x400;
    if (xr17 & 0x01)
        svga->htotal += 0x100;
    if (xr17 & 0x02) {
        svga->hdisp += 0x100 * svga->dots_per_clock;
        svga->hdisp_time += 0x100;
    }
    if (xr17 & 0x10)
        svga->hblankstart += 0x100;

    svga->interlace = graphics && (chips->xr[0x28] & 0x20);

    /* XR0D bit 0: a half step of the offset in word and doubleword modes. */
    if (chips->xr[0x0d] & 0x01) {
        svga->adv_flags |= FLAG_NO_SHIFT3;
        svga->rowoffset = (svga->rowoffset << 3) | 0x04;
        svga->memaddr_latch <<= 2;
    } else
        svga->adv_flags &= ~FLAG_NO_SHIFT3;

    svga->bpp = 8;

    /* XR28 bit 4: the 8-bit video path (the extended modes; GR05 bit 5 is then 0
       and bit 6 not used), its pixels 8 bits or, by XR06, high/true colour. */
    if (graphics && (chips->xr[0x28] & 0x10) && (svga->render != svga_render_blank)) {
        switch ((chips->xr[0x06] >> 2) & 3) {
            default:
                svga->map8   = svga->pallook;
                svga->render = svga_render_8bpp_highres;
                break;
            case 1:
                svga->bpp    = 15;
                svga->render = svga_render_15bpp_highres;
                break;
            case 2:
                svga->bpp    = 24;
                svga->render = svga_render_24bpp_highres;
                break;
            case 3:
                svga->bpp    = 16;
                svga->render = svga_render_16bpp_highres;
                break;
        }

        /* The CRTC counts bytes: two or three dot clocks a pixel. */
        if (svga->bpp == 24) {
            svga->hdisp /= 3;
            svga->dots_per_clock /= 3;
        } else if (svga->bpp != 8) {
            svga->hdisp >>= 1;
            svga->dots_per_clock >>= 1;
        }
    }
}

/* ---- PCI ---------------------------------------------------------------------- */

static uint8_t
chips_6554x_pci_read(UNUSED(int func), int addr, UNUSED(int len), void *priv)
{
    const chips_6554x_t *chips = (chips_6554x_t *) priv;
    uint8_t              ret   = 0x00;

    switch (addr) {
        case 0x00:
            ret = 0x2c;
            break;
        case 0x01:
            ret = 0x10;
            break;
        case 0x02:
            ret = (chips->chip == CHIPS_65548) ? 0xdc : 0xd8;
            break;
        case 0x04:
            ret = (chips->pci_cmd & 0x63) | 0x80;
            break;
        case 0x05:
            ret = chips->pci_cmd_hi & 0x01;
            break;
        case 0x06:
            ret = 0x80;
            break;
        case 0x07:
            ret = 0x04; /* DEVSEL slow */
            break;
        case 0x08:
            ret = 0x00;
            break;
        case 0x0b:
            ret = 0x03;
            break;
        case 0x12:
            ret = (chips->mbase >> 16) & 0x80;
            break;
        case 0x13:
            ret = chips->mbase >> 24;
            break;
        case 0x14:
            ret = 0x01;
            break;
        case 0x15:
            ret = (chips->iobase >> 8) & 0xfc;
            break;
        case 0x16:
            ret = chips->iobase >> 16;
            break;
        case 0x17:
            ret = chips->iobase >> 24;
            break;
        case 0x30:
            if (chips->has_rom)
                ret = chips->rom_bar & 0x01;
            break;
        case 0x32:
            if (chips->has_rom)
                ret = (chips->rom_bar >> 16) & 0x80;
            break;
        case 0x33:
            if (chips->has_rom)
                ret = chips->rom_bar >> 24;
            break;
        case 0x3c:
            ret = chips->int_line;
            break;
        default:
            break;
    }

    return ret;
}

static void
chips_6554x_update_rom(chips_6554x_t *chips)
{
    if (!chips->has_rom || (chips->bus != CHIPS_BUS_PCI))
        return;

    if ((chips->pci_cmd & PCI_COMMAND_MEM) && (chips->rom_bar & 0x01) && (chips->rom_bar & 0xffff8000))
        mem_mapping_set_addr(&chips->bios_rom.mapping, chips->rom_bar & 0xffff8000, 0x8000);
    else
        mem_mapping_disable(&chips->bios_rom.mapping);
}

static void
chips_6554x_pci_write(UNUSED(int func), int addr, UNUSED(int len), uint8_t val, void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    switch (addr) {
        case 0x04:
            chips->pci_cmd = val & 0x63;
            chips_6554x_update_enable(chips);
            chips_6554x_update_rom(chips);
            break;
        case 0x05:
            chips->pci_cmd_hi = val & 0x01;
            break;
        case 0x12:
            chips->mbase = (chips->mbase & 0xff000000) | ((val & 0x80) << 16);
            chips_6554x_update_linear(chips);
            break;
        case 0x13:
            chips->mbase = (chips->mbase & 0x00800000) | (val << 24);
            chips_6554x_update_linear(chips);
            break;
        case 0x15:
            chips->iobase = (chips->iobase & 0xffff0000) | ((val & 0xfc) << 8);
            break;
        case 0x16:
            chips->iobase = (chips->iobase & 0xff00fc00) | (val << 16);
            break;
        case 0x17:
            chips->iobase = (chips->iobase & 0x00fffc00) | (val << 24);
            break;
        case 0x30:
            if (chips->has_rom) {
                chips->rom_bar = (chips->rom_bar & 0xffff8000) | (val & 0x01);
                chips_6554x_update_rom(chips);
            }
            break;
        case 0x32:
            if (chips->has_rom) {
                chips->rom_bar = (chips->rom_bar & 0xff000001) | ((val & 0x80) << 16);
                chips_6554x_update_rom(chips);
            }
            break;
        case 0x33:
            if (chips->has_rom) {
                chips->rom_bar = (chips->rom_bar & 0x00800001) | (val << 24);
                chips_6554x_update_rom(chips);
            }
            break;
        case 0x3c:
            chips->int_line = val;
            break;
        default:
            break;
    }
}

/* ---- Device ------------------------------------------------------------------- */

static void
chips_6554x_reset_regs(chips_6554x_t *chips)
{
    memset(chips->xr, 0x00, sizeof(chips->xr));
    memset(chips->dr, 0x00, sizeof(chips->dr));
    memset(&chips->blt, 0x00, sizeof(chips->blt));

    chips->xr[0x07] = 0xf4; /* DR00 at 83D0h */
    chips->xr[0x33] = 0x00;

    /* MCLK 60 MHz (VCO 120 MHz, divided by 2), VCLK 2 40 MHz until programmed. */
    chips->mclk[0] = 0x03;
    chips->mclk[1] = 0x45;
    chips->mclk[2] = 0x1d;
    chips->vclk[0] = 0x03;
    chips->vclk[1] = 0x4a;
    chips->vclk[2] = 0x33;

    chips->setup_ctrl    = 0x08;
    chips->global_enable = 0x01;
    chips->vse           = 0x01;

    chips->svga.vram_display_mask = MIN(chips->svga.vram_mask, 0x3ffff);
}

static void *
chips_6554x_init(const device_t *info)
{
    chips_6554x_t *chips = calloc(1, sizeof(chips_6554x_t));
    const char    *romfn = NULL;

    chips->chip     = info->local & 0xff;
    chips->bus      = (info->local >> 8) & 0x0f;
    chips->on_board = !!(info->local & 0x10000);

    chips->vram_size = chips->on_board ? 1024 : device_get_config_int("memory");

    if (!(info->local & 0x20000)) {
        romfn          = (chips->chip == CHIPS_65548) ? BIOS_65548_PATH : BIOS_65545_PATH;
        chips->has_rom = 1;
        if (chips->chip == CHIPS_65548)
            rom_init(&chips->bios_rom, romfn, 0xc0000, 0x8000, 0x7fff, 0x0000, MEM_MAPPING_EXTERNAL);
        else
            rom_init(&chips->bios_rom, romfn, 0xc0000, 0xa000, 0xffff, 0x0000, MEM_MAPPING_EXTERNAL);
    }

    switch (chips->bus) {
        case CHIPS_BUS_PCI:
            video_inform(VIDEO_FLAG_TYPE_SPECIAL, &timing_chips_6554x_pci);
            break;
        case CHIPS_BUS_VLB:
            video_inform(VIDEO_FLAG_TYPE_SPECIAL, &timing_chips_6554x_vlb);
            break;
        default:
            video_inform(VIDEO_FLAG_TYPE_SPECIAL, &timing_chips_6554x_isa);
            break;
    }

    svga_init(info, &chips->svga, chips, chips->vram_size << 10,
              chips_6554x_recalctimings,
              chips_6554x_in, chips_6554x_out,
              chips_6554x_hwcursor_draw,
              NULL);

    mem_mapping_set_handler(&chips->svga.mapping,
                            chips_6554x_read, chips_6554x_readw, chips_6554x_readl,
                            chips_6554x_write, chips_6554x_writew, chips_6554x_writel);
    mem_mapping_set_p(&chips->svga.mapping, chips);

    mem_mapping_add(&chips->linear_mapping, 0, 0,
                    chips_6554x_lin_read, chips_6554x_lin_readw, chips_6554x_lin_readl,
                    chips_6554x_lin_write, chips_6554x_lin_writew, chips_6554x_lin_writel,
                    NULL, MEM_MAPPING_EXTERNAL, chips);
    mem_mapping_add(&chips->mmio_mapping, 0, 0,
                    chips_6554x_mmio_read, chips_6554x_mmio_readw, chips_6554x_mmio_readl,
                    chips_6554x_mmio_write, chips_6554x_mmio_writew, chips_6554x_mmio_writel,
                    NULL, MEM_MAPPING_EXTERNAL, chips);
    mem_mapping_disable(&chips->linear_mapping);
    mem_mapping_disable(&chips->mmio_mapping);

    io_sethandler(0x03b0, 0x0030, chips_6554x_in, NULL, NULL, chips_6554x_out, NULL, NULL, chips);
    io_sethandler(0x83c6, 0x0004, chips_6554x_dac_in, NULL, NULL, chips_6554x_dac_out, NULL, NULL, chips);
    if (chips->bus == CHIPS_BUS_ISA) {
        io_sethandler(0x0102, 0x0001, chips_6554x_setup_in, NULL, NULL, chips_6554x_setup_out, NULL, NULL, chips);
        io_sethandler(0x46e8, 0x0001, NULL, NULL, NULL, chips_6554x_46e8_out, NULL, NULL, chips);
    }

    chips_6554x_reset_regs(chips);
    chips_6554x_cursor_colors(chips);

    chips->svga.miscout = 1;
    chips->svga.bpp     = 8;

    if (chips->bus == CHIPS_BUS_PCI) {
        /* Disabled but for the configuration registers until the BIOS enables it. */
        pci_add_card(chips->on_board ? PCI_ADD_VIDEO : PCI_ADD_NORMAL, chips_6554x_pci_read, chips_6554x_pci_write, chips, &chips->pci_slot);
        if (chips->has_rom)
            mem_mapping_disable(&chips->bios_rom.mapping);
    }

    chips_6554x_update_enable(chips);

    return chips;
}

static int
chips_65545_available(void)
{
    return rom_present(BIOS_65545_PATH);
}

static int
chips_65548_available(void)
{
    return rom_present(BIOS_65548_PATH);
}

static void
chips_6554x_close(void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    svga_close(&chips->svga);

    free(chips);
}

static void
chips_6554x_speed_changed(void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    svga_recalctimings(&chips->svga);
}

static void
chips_6554x_force_redraw(void *priv)
{
    chips_6554x_t *chips = (chips_6554x_t *) priv;

    chips->svga.fullchange = changeframecount;
}

// clang-format off
static const device_config_t chips_6554x_config[] = {
    {
        .name           = "memory",
        .description    = "Memory size",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = 1024,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            { .description = "512 KB", .value =  512 },
            { .description = "1 MB",   .value = 1024 },
            { .description = ""                      }
        },
        .bios           = { { 0 } }
    },
    { .name = "", .description = "", .type = CONFIG_END }
};
// clang-format on

const device_t chips_65545_vlb_device = {
    .name          = "Chips & Technologies 65545 (VLB)",
    .internal_name = "chips_65545_vlb",
    .flags         = DEVICE_VLB,
    .local         = CHIPS_LOCAL(CHIPS_65545, CHIPS_BUS_VLB, 0, 0),
    .init          = chips_6554x_init,
    .close         = chips_6554x_close,
    .reset         = NULL,
    .available     = chips_65545_available,
    .speed_changed = chips_6554x_speed_changed,
    .force_redraw  = chips_6554x_force_redraw,
    .config        = chips_6554x_config
};

const device_t chips_65545_vlb_onboard_device = {
    .name          = "Chips & Technologies 65545 (VLB, On-Board)",
    .internal_name = "chips_65545_vlb_onboard",
    .flags         = DEVICE_ISA16,
    .local         = CHIPS_LOCAL(CHIPS_65545, CHIPS_BUS_VLB, 1, 0),
    .init          = chips_6554x_init,
    .close         = chips_6554x_close,
    .reset         = NULL,
    .available     = chips_65545_available,
    .speed_changed = chips_6554x_speed_changed,
    .force_redraw  = chips_6554x_force_redraw,
    .config        = NULL
};

const device_t chips_65548_pci_device = {
    .name          = "Chips & Technologies 65548 (PCI)",
    .internal_name = "chips_65548_pci",
    .flags         = DEVICE_PCI,
    .local         = CHIPS_LOCAL(CHIPS_65548, CHIPS_BUS_PCI, 0, 0),
    .init          = chips_6554x_init,
    .close         = chips_6554x_close,
    .reset         = NULL,
    .available     = chips_65548_available,
    .speed_changed = chips_6554x_speed_changed,
    .force_redraw  = chips_6554x_force_redraw,
    .config        = chips_6554x_config
};

const device_t chips_65548_pci_onboard_device = {
    .name          = "Chips & Technologies 65548 (PCI, On-Board)",
    .internal_name = "chips_65548_pci_onboard",
    .flags         = DEVICE_PCI,
    .local         = CHIPS_LOCAL(CHIPS_65548, CHIPS_BUS_PCI, 1, 1),
    .init          = chips_6554x_init,
    .close         = chips_6554x_close,
    .reset         = NULL,
    .available     = NULL,
    .speed_changed = chips_6554x_speed_changed,
    .force_redraw  = chips_6554x_force_redraw,
    .config        = NULL
};
