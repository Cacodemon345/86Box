#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <wchar.h>
#include <stdbool.h>
#define HAVE_STDARG_H
#include <86box/86box.h>
#include <86box/machine.h>
#include "cpu.h"
#include <86box/io.h>
#include <86box/pic.h>
#include <86box/mem.h>
#include <86box/device.h>
#include <86box/pci.h>
#include <86box/plat_fallthrough.h>
#include <86box/plat_unused.h>

typedef struct mpc105_t {
    uint32_t local;
    uint8_t  type;
    uint8_t  ctl;

    uint8_t regs[256];
    uint8_t bus_index;
    uint8_t slot;
} mpc105_t;

extern bool little_endian_mem;
extern uint8_t sys_cntl;
extern bool iomap;

static void
mpc105_update_ram_mapping(mpc105_t* dev)
{
    mem_set_mem_state_cpu_both(0, mem_size * 1024, 0);
    for (int i = 0; i < 8; i++) {
        if (dev->regs[0xA0] & (1 << i)) {
            uint32_t begin = 0;
            uint32_t end = 0;

            begin = (((uint32_t)dev->regs[0x80 + i]) | (((uint32_t)dev->regs[0x88 + i] & 0x3) << 8)) << 20;
            end = (((uint32_t)dev->regs[0x90 + i]) | (((uint32_t)dev->regs[0x98 + i] & 0x3) << 8)) << 20;
            end |= 0x000FFFFF;
            if (end > 0x7FFFFFFF) {
                end = 0x7FFFFFFF;
            }
            uint32_t size = (end - begin) + 1;
            mem_set_mem_state_cpu_both(begin, size, MEM_READ_INTERNAL | MEM_WRITE_INTERNAL);
        }
    }
}

static void
mpc105_write(int func, int addr, UNUSED(int len), uint8_t val, void *priv)
{
    mpc105_t *dev = (mpc105_t *) priv;

    if (func > 0)
        return;

    switch (addr) {
        case 0x00:
        case 0x01:
        case 0x02:
        case 0x03:
        case 0x06:
        case 0x08:
        case 0x09:
        case 0x0a:
        case 0x0b:
        case 0x0e:
        case 0x0f:
        case 0x10:
        case 0x11:
        case 0x12:
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x16:
        case 0x17:
        case 0x1e:
        case 0x34:
        case 0x3d:
            return;

        case 0x04:
            val &= 0x67;
            break;


        case 0x1f:
            return;

        case 0x3c:
            return;

        case 0x3e:
            return;

        case 0x3f:
            return;

        case 0x40:
            return;

        case 0x41:
            return;
        
        case 0x42:
            return;

        case 0x70:
        case 0xA8 ... 0xAB:
        case 0xAC ... 0xAF:
        case 0xBA:
        case 0xBB:
        case 0xC0:
        case 0xC4:
            dev->regs[addr] = val;
            return;

        case 0x80 ... 0xA0:
            dev->regs[addr] = val;
            mpc105_update_ram_mapping(dev);
            return;
        
        case 0xF0 ... 0xF3:
            dev->regs[addr] = val;
            mpc105_update_ram_mapping(dev);
            return;

        default:
            return;
    }

    dev->regs[addr] = val;
}

static uint8_t
mpc105_read(int func, int addr, UNUSED(int len), void *priv)
{
    const mpc105_t *dev = (mpc105_t *) priv;
    uint8_t             ret;

    if (func > 0)
        ret = 0xff;
    else
        ret = dev->regs[addr];

    if (func == 0) {
        switch (addr) {
            case 0xA8:
                ret &= ~(1 << 5);
                ret |= little_endian_mem << 5;
                break;
            case 0xBA:
                ret &= ~(1 << 2);
                ret |= iomap << 2;
                break;
            case 0xAA:
                ret &= ~8;
                ret |= !(iomap) ? 8 : 0;
                break;
        }
    }

    return ret;
}

static void
mpc105_reset(void *priv)
{
    mpc105_t *dev = (mpc105_t *) priv;

    memset(dev->regs, 0, sizeof(dev->regs));

    /* IDs */
    dev->regs[0x00] = 0x57;
    dev->regs[0x01] = 0x10;
    dev->regs[0x02] = 0x01;
    dev->regs[0x03] = 0x00;

    dev->regs[0x08] = 0x00;

    /* class */
    dev->regs[0x0a] = 0x00;
    dev->regs[0x0b] = 0x06; /* bridge device */
    dev->regs[0x0e] = 0x01; /* bridge header */

}

static void *
mpc105_init(const device_t *info)
{
    mpc105_t *dev = (mpc105_t *) calloc(1, sizeof(mpc105_t));

    pci_add_card(PCI_ADD_NORTHBRIDGE, mpc105_read, mpc105_write, dev, &dev->slot);

    return dev;
}

/* PCI bridges */
const device_t mpc105_device = {
    .name          = "DEC 21150 PCI Bridge",
    .internal_name = "dec21150",
    .flags         = DEVICE_PCI,
    .local         = 0,
    .init          = mpc105_init,
    .close         = NULL,
    .reset         = mpc105_reset,
    .available     = NULL,
    .speed_changed = NULL,
    .force_redraw  = NULL,
    .config        = NULL
};