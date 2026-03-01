#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdbool.h>
#include <wchar.h>
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

// Port 92's reset is handled elsewhere.
bool little_endian_mem = false;
uint8_t sys_cntl = 0;
bool iomap = 1;

void port_8xx_write(uint16_t port, uint8_t val, void *priv)
{
    switch (port) {
        case 0x92:
            little_endian_mem = val & 0x2;
            break;
        case 0x81c:
            sys_cntl = val;
            break;
        case 0x850:
            iomap = val & 1;
            break;
    }
}

uint8_t port_8xx_read(uint16_t port, void *priv)
{
    switch (port) {
        default:
            return 0xFF;
        case 0x80C:
            return 0xC0;
        case 0x852:
            return 0xFC;
        case 0x850:
            return iomap;
        case 0x81c:
            return sys_cntl;
        case 0x92:
            return 0xff; // let the SIO handle this.
    }
}

void
port_8xx_reset(void* priv)
{
    little_endian_mem = false;
    sys_cntl = 0;
    iomap = 1;
}

void *
port_8xx_init(const device_t* info)
{
    io_sethandler(0x0800, 256, port_8xx_read, NULL, NULL, port_8xx_write, NULL, NULL, NULL);
    return port_8xx_init;
}

void
port_8xx_close(void* priv)
{
    // no-op
}

const device_t port_8xx_device = {
    .name          = "Port 8xx Register",
    .internal_name = "port_8xx",
    .flags         = 0,
    .local         = 0,
    .init          = port_8xx_init,
    .close         = port_8xx_close,
    .reset         = port_8xx_reset,
    .available     = NULL,
    .speed_changed = NULL,
    .force_redraw  = NULL,
    .config        = NULL
};