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

static int port0820_index = 0;

static uint32_t ram_sockets[8];
static uint32_t ram_sockets_end_addr[8];

void port_8xx_write(uint16_t port, uint8_t val, void *priv)
{
    switch (port) {
        case 0x92:
            little_endian_mem = val & 0x2;
            break;
        case 0x81c:
            sys_cntl = val;
            break;
        case 0x820:
        {
            uint8_t socket = val >> 5;
            uint32_t end_address = val & 0x1f;
            ram_sockets_end_addr[socket] = val & 0x1f;
            if (socket > 0 && socket < 7) {
                if (ram_sockets[socket - 1]) {
                    uint32_t size;
                    uint32_t start_address = 0;
                    if (socket > 1) {
                        start_address = ram_sockets_end_addr[socket - 1];
                    }

                    size = end_address - start_address;
                    mem_set_mem_state_cpu_both(start_address * 8 * 1024 * 1024, size * 8 * 1024 * 1024, MEM_READ_INTERNAL | MEM_WRITE_INTERNAL);
                    //memory_region_set_enabled(&s->simm[socket - 1], size != 0);
                    //memory_region_set_address(&s->simm[socket - 1],
                    //                        start_address * 8 * MiB);
                }
            }

            break;
        }
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
        case 0x803:
        {
            uint32_t val = 0;
            int socket;

            /* (1 << socket) indicates 32 MB SIMM at given socket */
            for (socket = 0; socket < 6; socket++) {
                if (ram_sockets[socket] == 32) {
                    val |= (1 << socket);
                }
            }
            return val;
        }
        case 0x804:
        {
            uint32_t val = 0xff;
            int socket;

            /* (1 << socket) indicates SIMM absence at given socket */
            for (socket = 0; socket < 6; socket++) {
                if (ram_sockets[socket]) {
                    val &= ~(1 << socket);
                }
            }
            port0820_index = 0;
            return val;
        }
        case 0x820:
        {
            uint8_t ret = ram_sockets_end_addr[port0820_index & 7];
            port0820_index++;
            return ret;
        }
        case 0x841:
            return 1;
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
    port0820_index = 0;
}

void *
port_8xx_init(const device_t* info)
{
    int socket = 0;
    int mem_size_mb = mem_size / 1024;
    io_sethandler(0x0800, 256, port_8xx_read, NULL, NULL, port_8xx_write, NULL, NULL, NULL);

    port_8xx_reset(info);
    memset(ram_sockets, 0, sizeof(ram_sockets));
    while (socket < 6) {
        if (mem_size_mb >= 64) {
            ram_sockets[socket] = 32;
            ram_sockets[socket + 1] = 32;
            mem_size_mb -= 64;
        } else if (mem_size_mb >= 16) {
            ram_sockets[socket] = 8;
            ram_sockets[socket + 1] = 8;
            mem_size_mb -= 16;
        } else {
            /* Not enough memory */
            break;
        }
        socket += 2;
    }

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