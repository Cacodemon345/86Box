/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef VIDEO_VERITE_CORE_H
#define VIDEO_VERITE_CORE_H

#include <stddef.h>
#include <stdint.h>

#if (defined(__x86_64__) || defined(_M_X64)) && UINTPTR_MAX == UINT64_MAX && !defined(VERITE_NO_JIT)
#    define VERITE_JIT_X64 1
#endif

typedef struct verite_core verite_core;

typedef struct verite_board {
    uint32_t (*read)(void *opaque, uint8_t reg, uint8_t len);
    void (*write)(void *opaque, uint8_t reg, uint32_t value, uint8_t len);
    uint32_t (*dma_read)(void *opaque, uint32_t address);
    /* Optional: reads count consecutive dwords for a DMA transfer. */
    void (*dma_read_block)(void *opaque, uint32_t address, uint32_t *words, unsigned count);
    /* Optional: the DMA registers at 50h-5Bh changed to regs, as of ns in
       the guest's time. Times only go back for a change the guest made. */
    void (*dma_moved)(void *opaque, uint64_t ns, const uint32_t regs[3]);
} verite_board;

verite_core *verite_core_new(void);
void         verite_core_free(verite_core *core);
void         verite_core_reset(verite_core *core);
#ifdef VERITE_JIT_X64
void         verite_core_set_jit(verite_core *core, int enabled);
#endif
uint32_t     verite_core_read(verite_core *core, uint8_t reg, uint8_t len,
                              const uint8_t *vram, size_t size);
void         verite_core_write(verite_core *core, uint8_t reg, uint32_t value, uint8_t len,
                               uint8_t *vram, size_t size, const verite_board *board, void *opaque);
void         verite_core_run(verite_core *core, uint64_t now_ns, uint8_t *vram, size_t size,
                             const verite_board *board, void *opaque);
/* Runs the RISC with the time up to now_ns, letting it get ahead cycles past
   now_ns but running at most cap cycles in this call. Returns the cycles run. */
int64_t      verite_core_advance(verite_core *core, uint64_t now_ns, int64_t ahead, int64_t cap,
                                 uint8_t *vram, size_t size, const verite_board *board, void *opaque);
/* The RISC can't run until the host writes to it. */
int          verite_core_waiting(const verite_core *core);
/* The last run stopped after the RISC polled the board for too long. */
int          verite_core_polling(const verite_core *core);
int          verite_core_input_full(const verite_core *core);
/* The host appends to a running DMA by patching the end of its list once it
   reads the DMA busy. A RISC run apart from the guest's time could reach the
   end before the guest's patch and stop, losing it. With defer set, a DMA at
   the end of its list stays busy there, as of the RISC's time then. Once the
   guest's time reaches it, verite_core_finish_dma, called in step with the
   guest, rereads the end and stops the DMA unless the list was extended.
   Until then the guest reads the DMA busy, as it would in step. */
void         verite_core_defer_dma_end(verite_core *core, int defer);
int          verite_core_dma_parked(const verite_core *core);
int          verite_core_dma_busy(const verite_core *core);
uint64_t     verite_core_dma_end_ns(const verite_core *core);
/* How far the RISC has run, in the guest's time. */
uint64_t     verite_core_time_ns(const verite_core *core);
int          verite_core_dma_end_due(uint64_t end_ns, uint64_t now_ns);
void         verite_core_finish_dma(verite_core *core, uint64_t now_ns, const verite_board *board, void *opaque);
void         verite_core_retrace(verite_core *core);
void         verite_core_set_precharge(verite_core *core, int slow);
void         verite_core_refresh(verite_core *core, unsigned cycles);
uint32_t     verite_core_status(verite_core *core);
uint32_t     verite_core_bank(const verite_core *core);
uint8_t      verite_core_mode(const verite_core *core);

#endif
