/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          Helpers shared by the Chips & Technologies VGA controllers.
 *
 * Authors: 86Box contributors.
 *
 *          Copyright 2026 86Box contributors.
 */
#ifndef VIDEO_CHIPS_H
#define VIDEO_CHIPS_H

/* Bits of CRTC register idx that the write protect register XR15 (and CR11 bit 7)
   leave writable (82C45x, 6554x). */
extern uint8_t chips_crtc_wmask(uint8_t xr15, uint8_t cr11, uint8_t idx);

#endif /*VIDEO_CHIPS_H*/
