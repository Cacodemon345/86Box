
#include <cstring>
#include <cstdio>
#include <stdarg.h>
#include <stdint.h>

//#include "system/systhread.h"

//#include "uae/log.h"

int ht_printf(const char *format,...)
{
	//UAE_LOG_VA_ARGS_FULL(format);
	return 0;
}
int ht_fprintf(FILE *f, const char *fmt, ...)
{
	return 0;
}
int ht_vfprintf(FILE *file, const char *fmt, va_list args)
{
	return 0;
}
int ht_snprintf(char *str, size_t count, const char *fmt, ...)
{
	return 0;
}
int ht_vsnprintf(char *str, size_t count, const char *fmt, va_list args)
{
	return 0;
}

void ht_assert_failed(const char *file, int line, const char *assertion)
{
}

#if 0
void prom_quiesce()
{
}
#endif

//#include "sysconfig.h"
//#include "sysdeps.h"
//#include <threaddep/thread.h>

extern "C"
{
#include <86box/86box.h>
#include <86box/mem.h>
#include <86box/thread.h>
extern uint8_t *pccache2;
extern uint32_t pccache;

}
typedef void * sys_mutex;
void uae_ppc_crash(void)
{
	fatal("PPC emulation failure.");
}

static __inline void *
get_ram_ptr(uint32_t a)
{
    if ((a >> 12) == pccache)
        return (void *) (((uintptr_t) &pccache2[a] & 0x00000000ffffffffULL) | ((uintptr_t) &pccache2[0] & 0xffffffff00000000ULL));
    else {
        uint8_t *t = getpccache(a);
        return (void *) (((uintptr_t) &t[a] & 0x00000000ffffffffULL) | ((uintptr_t) &t[0] & 0xffffffff00000000ULL));
    }
}

bool uae_ppc_direct_physical_memory_handle(uint32_t addr, uint8_t *&ptr)
{
	ptr = (uint8_t*)get_ram_ptr(addr);
	return true;
}
int sys_lock_mutex(sys_mutex m)
{
	//uae_sem_wait(&m);
	thread_wait_mutex((mutex_t*)m);
	return 1;
}

void sys_unlock_mutex(sys_mutex m)
{
	thread_release_mutex((mutex_t*)m);
}

int sys_create_mutex(sys_mutex *m)
{
	if (!(*m)) {
		//uae_sem_init(m, 0, 1);
		*m = (sys_mutex*)thread_create_mutex();
	}
	return 1;
}

void sys_destroy_mutex(sys_mutex m)
{
	//uae_sem_destroy(&m);
	thread_close_mutex((mutex_t*)m);
}
