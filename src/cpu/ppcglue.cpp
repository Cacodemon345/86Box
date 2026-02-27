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
    return true;
}
bool UAECALL cb_uae_ppc_io_mem_write(uint32_t addr, uint32_t data, int size)
{
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
    return true;
}
bool UAECALL cb_uae_ppc_io_mem_read64(uint32_t addr, uint64_t *data)
{
    return false;
}
bool UAECALL cb_uae_ppc_io_mem_write64(uint32_t addr, uint64_t data)
{
    return false;
}

uae_ppc_io_mem_read_function uae_ppc_io_mem_read = cb_uae_ppc_io_mem_read;
uae_ppc_io_mem_write_function uae_ppc_io_mem_write = cb_uae_ppc_io_mem_write;
uae_ppc_io_mem_read64_function uae_ppc_io_mem_read64 = NULL;
uae_ppc_io_mem_write64_function uae_ppc_io_mem_write64 = NULL;


void reset_ppc(int hard)
{
    if (hard) {
        ppc_cpu_init(0x00070101);
    }
    ppc_cpu_set_pc(0, 0xfff00100);
}

void execppc(int32_t cycs)
{
    while (cycs > 0) {
        int32_t cyc_period = (cycs > 2000) ? 2000 : cycs;
        ppc_cpu_run_single(cyc_period);
        cycs -= cyc_period;
        tsc += cyc_period;
        
        if (TIMER_VAL_LESS_THAN_VAL(timer_target, (uint64_t) tsc))
            timer_process();
    }
}
