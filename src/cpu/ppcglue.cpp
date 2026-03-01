#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <wchar.h>

extern "C" {
#define HAVE_STDARG_H
#include <86box/86box.h>
#include <86box/device.h>
#include <86box/machine.h>
#include <86box/io.h>
#include <86box/mem.h>
#include <86box/nmi.h>
#include <86box/pic.h>
#include <86box/pci.h>
#include <86box/smram.h>
#include <86box/timer.h>
#include "cpu.h"
#include <86box/plat_unused.h>
#include <86box/bswap.h>
#include <86box/gdbstub.h>
#include <86box/plat_fallthrough.h>
#include <86box/plat_unused.h>

int is_ppc = 0;
}

#include "../pearpc/cpu/cpu.h"
#include "../pearpc/io/io.h"
#include "../pearpc/cpu/cpu_generic/ppc_cpu.h"

bool UAECALL cb_uae_ppc_io_mem_read(uint32_t addr, uint32_t *data, int size)
{
    if (size == 4 && (addr & 3)) {
        uint32_t temp_data = 0;
        uint32_t temp_data_2 = 0;
        cb_uae_ppc_io_mem_read(addr, &temp_data, 2);
        cb_uae_ppc_io_mem_read(addr + 2, &temp_data_2, 2);
        *data = (temp_data << 16) | temp_data_2;
        return true;
    }
    if (size == 2 && (addr & 1)) {
        *data = ((uint32_t)read_mem_b(addr)) << 8;
        *data |= ((uint32_t)read_mem_b(addr + 1));
        return true;
    }
    switch (size)
    {
        case 1:
            *data = read_mem_b(addr);
            break;
        case 2:
            *data = bswap16(read_mem_w(addr));
            break;
        case 4:
            *data = bswap32(read_mem_l(addr));
            break;
    }
    if (addr >= 0x80000000) {
        pclog("Read from bus = %X\n", size == 2 ? bswap16(*data) : (size == 4 ? bswap32(*data) : *data));
    }
    return true;
}
bool UAECALL cb_uae_ppc_io_mem_write(uint32_t addr, uint32_t data, int size)
{
    if (size == 4 && (addr & 3)) {
        cb_uae_ppc_io_mem_write(addr, data >> 16, 2);
        cb_uae_ppc_io_mem_write(addr + 2, data & 0xFFFF, 2);
        return true;
    }
    if (size == 2 && (addr & 1)) {
        write_mem_b(addr, data >> 8);
        write_mem_b(addr + 1, data & 0xFF);
        return true;
    }
    switch (size)
    {
        case 1:
            write_mem_b(addr, data);
            break;
        case 2:
            write_mem_w(addr, bswap16(data));
            break;
        case 4:
            write_mem_l(addr, bswap32(data));
            break;
    }
    if (addr >= 0x80000000) {
        pclog("Write to bus = %X\n", size == 2 ? bswap16(data) : (size == 4 ? bswap32(data) : data));
    }
    return true;
}
bool UAECALL cb_uae_ppc_io_mem_read64(uint32_t addr, uint64_t *data)
{
    uint32_t temp_data = 0, temp_data_2 = 0;

    temp_data = cb_uae_ppc_io_mem_read(addr, &temp_data, 4);
    temp_data_2 = cb_uae_ppc_io_mem_read(addr, &temp_data_2, 4);
    *data = (((uint64_t)temp_data) << 32ull) | ((uint64_t)temp_data_2);
    return true;
}
bool UAECALL cb_uae_ppc_io_mem_write64(uint32_t addr, uint64_t data)
{
    cb_uae_ppc_io_mem_write(addr, data >> 32, 4);
    cb_uae_ppc_io_mem_write(addr + 4, data & 0xFFFFFFFF, 4);
    return true;
}

uae_ppc_io_mem_read_function uae_ppc_io_mem_read = cb_uae_ppc_io_mem_read;
uae_ppc_io_mem_write_function uae_ppc_io_mem_write = cb_uae_ppc_io_mem_write;
uae_ppc_io_mem_read64_function uae_ppc_io_mem_read64 = cb_uae_ppc_io_mem_read64;
uae_ppc_io_mem_write64_function uae_ppc_io_mem_write64 = cb_uae_ppc_io_mem_write64;


void reset_ppc(int hard)
{
    if (hard) {
        ppc_cpu_init(0x00070101);
    }
    ppc_cpu_set_pc(0, (gCPU.msr & MSR_IP) ? 0xfff00100 : 0x00000100);
}

void execppc(int32_t cycs)
{
    while (cycs > 0) {
        int32_t cyc_period = (cycs > 2000) ? 2000 : cycs;
        ppc_cpu_run_single(cyc_period);
        cycs -= cyc_period;
        tsc += cyc_period;

        if (pic.int_pending)
            ppc_cpu_atomic_raise_ext_exception();
        
        if (TIMER_VAL_LESS_THAN_VAL(timer_target, (uint64_t) tsc))
            timer_process();
    }
}
