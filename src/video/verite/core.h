/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef VIDEO_VERITE_CORE_H
#define VIDEO_VERITE_CORE_H

#include <stddef.h>
#include <stdint.h>

#if UINTPTR_MAX == UINT64_MAX && !defined(VERITE_NO_JIT)
#    if defined(__x86_64__) || defined(_M_X64)
#        define VERITE_JIT_X64 1
#    elif defined(__aarch64__) || defined(_M_ARM64)
#        define VERITE_JIT_ARM64 1
#    endif
#endif
#if defined(VERITE_JIT_X64) || defined(VERITE_JIT_ARM64)
#    define VERITE_JIT 1
#endif

typedef struct verite_core verite_core;

typedef struct verite_board {
    uint32_t (*read)(void *opaque, uint8_t reg, uint8_t len);
    void (*write)(void *opaque, uint8_t reg, uint32_t value, uint8_t len);
    uint32_t (*dma_read)(void *opaque, uint32_t address);
    /* Optional: reads count consecutive dwords for a DMA transfer. */
    void (*dma_read_block)(void *opaque, uint32_t address, uint32_t *words, unsigned count);
} verite_board;

verite_core *verite_core_new(void);
void         verite_core_free(verite_core *core);
void         verite_core_reset(verite_core *core);
#ifdef VERITE_JIT
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
/* Raises interrupt status bits, as the device's own DMA engine does at the
   end of a list when it feeds the RISC through the data FIFO. */
void         verite_core_raise(verite_core *core, unsigned bits);
void         verite_core_retrace(verite_core *core);
void         verite_core_set_precharge(verite_core *core, int slow);
void         verite_core_refresh(verite_core *core, unsigned cycles);
uint32_t     verite_core_status(verite_core *core);
uint32_t     verite_core_bank(const verite_core *core);
uint8_t      verite_core_mode(const verite_core *core);

#endif
