/*
 * QEMU 3C509B PnP ISA NIC Emulation
 * 
 * Copyright (c) 2004 Antony T Curtis
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
 
/*
 * This simulation was created with the help of 
 * "EtherLink III Parallel Tasking
 *  ISA, EISA, Micro Channel, and PCMCIA 
 *  Adapter Drivers Technical Reference"
 * 3Com Manual Part No. 09-0389-002B
 * Published August 1994.
 *
 * All trademarks and registered marks are acknowledged
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include <wchar.h>
#include <time.h>
#define HAVE_STDARG_H
#include <86box/86box.h>
#include <86box/io.h>
#include <86box/dma.h>
#include <86box/pic.h>
#include <86box/mem.h>
#include <86box/random.h>
#include <86box/device.h>
#include <86box/thread.h>
#include <86box/timer.h>
#include <86box/network.h>
#include <86box/net_dp8390.h>
#include <86box/isapnp.h>
#include <86box/plat_unused.h>

//#define TCM_DEBUG
//#define TCM_DEBUG_MATCH
//#define TCM_DEBUG_EEPROM

#define TCM_FIFO_SIZE 32
#define TCM_STATUS_SIZE 32
#define TCM_EEPROM_SIZE 0x40

#define TCM_ETH_FRAME_MIN 8
#define TCM_ETH_FRAME_MAX 1792
#define TCM_ETH_FRAME_RUNT 60
#define TCM_ETH_FRAME_OVERSIZE 1514

#define TCM_MFG_DATE(Y,M,D) ((((Y) % 100)<<9)|((M)<<5)|(D))

typedef enum { BYTEOP_NONE, BYTEOP_READ, BYTEOP_WRITE } TCM_ByteOp;

struct TCM509_FIFO {
    uint16_t status;
    uint16_t packet[TCM_ETH_FRAME_MAX / 2];
    uint16_t pos, max;
};

#ifndef ETHER_IS_MULTICAST /* Net/Open BSD macro it seems */
#    define ETHER_IS_MULTICAST(a) ((*(uint8_t *) (a)) & 1)
#endif

#define ETHER_ADDR_LEN ETH_ALEN
#define ETH_ALEN       6
#pragma pack(1)
struct ether_header /** @todo Use RTNETETHERHDR? */
{
    uint8_t  ether_dhost[ETH_ALEN]; /**< destination ethernet address */
    uint8_t  ether_shost[ETH_ALEN]; /**< source ethernet address */
    uint16_t ether_type;            /**< packet type ID field */
};
#pragma pack()

typedef struct TCM509State {
    void* dev;
    netcard_t *nic;
    uint16_t status;
    int base, irq, activated;
    uint32_t byteaddr, bytedata;
    uint8_t irq_mask, irq_zero;
    TCM_ByteOp byteop;

    uint16_t padr[3], rx_filter;
    uint8_t macaddr[6];

    int txpos, txend;
    
    int fifo_head, fifo_tail, fifo_used;
    
    uint16_t media_type, net_status, config;
    uint16_t addr_config, res_config;

    uint32_t      base_address;

    uint16_t tx_bytes, rx_bytes;
    uint16_t rx_early_thresh;
    uint16_t tx_avail_thresh;
    uint16_t tx_start_thresh;
    
    uint8_t tx_frames, rx_frames, tx_defer, rx_discard;
    uint8_t tx_late, tx_onecol, tx_collisons, tx_nocd;
    uint8_t cd_lost;

    uint16_t eeprom_data, eeprom_cmd;
    int eeprom_wren;
    
    int tx_head, tx_tail, tx_used;
    
    uint16_t eeprom[TCM_EEPROM_SIZE];
    uint8_t tx_status[TCM_STATUS_SIZE];
    struct TCM509_FIFO fifo[TCM_FIFO_SIZE];
    uint16_t txbuffer[2048+4];

} TCM509State;

#define TCM_WINDOW(S)   (((S)->status >> 13)& 0x07)
#define TCM_STATUS(S)   ((S)->status & (0xff01 | (S)->irq_zero))

#define TCM_TX_ENABLED(S)       !!((S)->net_status & 0x0800)
#define TCM_RX_ENABLED(S)       !!((S)->net_status & 0x0400)
#define TCM_STAT_ENABLED(S)     !!((S)->net_status & 0x0080)
#define TCM_LOOP_ENABLED(S)     !!((S)->net_status & 0xf000)

#define TCM_RX_PADR(S)          !!((S)->rx_filter & 0x0001)
#define TCM_RX_MULTI(S)         !!((S)->rx_filter & 0x0002)
#define TCM_RX_BCAST(S)         !!((S)->rx_filter & 0x0004)
#define TCM_RX_PROM(S)          !!((S)->rx_filter & 0x0008)

static int tcm509_receive(void *opaque, uint8_t *buf, int size);

static void tcm509_ioport_writeb(uint16_t addr, uint8_t val, void *opaque);
static uint8_t tcm509_ioport_readb(uint16_t addr, void *opaque);


static inline uint16_t tcm_read_zero(uint16_t *value) 
{
    uint16_t result = *value;
    *value = 0;
    return result;
}

static inline uint8_t tcm_readb_zero(uint8_t *value) 
{
    uint8_t result = *value;
    *value = 0;
    return result;
}

static inline int padr_match(TCM509State *s, const uint8_t *buf, int size)
{
    struct ether_header *hdr = (void *)buf;
    uint8_t padr[6] = { 
        s->padr[0] >> 8, s->padr[0] & 0xff, 
        s->padr[1] >> 8, s->padr[1] & 0xff, 
        s->padr[2] >> 8, s->padr[2] & 0xff, 
    };    
    int result = TCM_RX_PADR(s) && !memcmp(hdr->ether_dhost, padr, 6);
#ifdef TCM_DEBUG_MATCH
    printf("packet dhost=%02x:%02x:%02x:%02x:%02x:%02x, "
           "padr=%02x:%02x:%02x:%02x:%02x:%02x\n",
           hdr->ether_dhost[0],hdr->ether_dhost[1],hdr->ether_dhost[2],
           hdr->ether_dhost[3],hdr->ether_dhost[4],hdr->ether_dhost[5],
           padr[0],padr[1],padr[2],padr[3],padr[4],padr[5]);
    printf("padr_match result=%d (rx_filter=0x%04x)\n", result, s->rx_filter);
#endif
    return result;
}

static inline int padr_bcast(TCM509State *s, const uint8_t *buf, int size)
{
    static uint8_t BCAST[6] = { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
    struct ether_header *hdr = (void *)buf;
    int result = TCM_RX_BCAST(s) && !memcmp(hdr->ether_dhost, BCAST, 6);
    return result;
}

static inline int ladr_match(TCM509State *s, const uint8_t *buf, int size)
{
    struct ether_header *hdr = (void *)buf;
    return TCM_RX_MULTI(s) && (*(hdr->ether_dhost)&0x01);
}


static void tcm509_update_irq(TCM509State *s) 
{
    s->status &= ~1;    
    if (TCM_STATUS(s) & s->irq_mask) {
        s->status |= 1;
    }
    if (s->irq) {
        int status = s->status & s->config & 1;
        if (status)
            picint(1 << s->irq);
        else
            picintc(1 << s->irq);
        //pic_set_irq(s->irq, s->status & s->config & 1);
    }
}

static void tcm509_reset(void *priv)
{
    TCM509State *s = priv;
    s->config = 0xce80;

    tcm509_update_irq(s);

    s->media_type = 0xa820;
    s->net_status = 0x0002;
    s->addr_config = s->eeprom[0x08];
    s->res_config = s->eeprom[0x09];
    
    s->status = 0;
    s->rx_filter = 0;
    s->txpos = s->txend = 0;
    s->fifo_head = s->fifo_tail = s->fifo_used = 0;
    s->tx_head = s->tx_tail = s->tx_used = 0;

    s->tx_bytes = s->rx_bytes = 0;
    s->tx_frames = s->rx_frames = s->tx_defer = s->rx_discard = 0;
    s->tx_late = s->tx_onecol = s->tx_collisons = s->tx_nocd = 0;
    s->cd_lost = 0;

    s->irq_mask = s->irq_zero = 0;

    memcpy(s->padr, s->eeprom, 6);

    tcm509_update_irq(s);
}


static void tcm509_update_rx(TCM509State *s)
{
    if (s->fifo_used) {
            struct TCM509_FIFO *fifo = s->fifo + s->fifo_head;
            if (fifo->pos == 0)
                s->status |= 0x0010;
    }

    tcm509_update_irq(s);
}

static void tcm509_command(TCM509State *s, uint16_t command) 
{
    switch (command >> 11) {
    case 0x00:
        if (command == 0)
            tcm509_reset(s);
        break;
    case 0x01:
        s->status &= ~0xe000;
        s->status |= command << 13;
        break;
    case 0x03:
        s->net_status &= ~0x0400;
        break;
    case 0x04:
        s->net_status  |= 0x0400;
        break;
    case 0x05:
        s->net_status &= ~0x0400;
        s->fifo_head = s->fifo_tail = s->fifo_used = 0;
        s->rx_filter = 0;
        s->status &= 0x0030;
        tcm509_update_irq(s);
        break;
    case 0x08:
        if (s->fifo_used) {
            s->fifo_used--;
            s->fifo_head++;
            if (s->fifo_head >= TCM_FIFO_SIZE)
                s->fifo_head = 0;
        }
        tcm509_update_irq(s);
        break;
    case 0x09:
        s->net_status |= 0x0800;
        break;
    case 0x0a:
        s->net_status &= ~0x0800;
        break;
    case 0x0b:
        s->net_status &= ~0x0800;
        s->txpos = s->txend = 0;
        s->tx_head = s->tx_tail = s->tx_used = 0;
        s->status &= 0x000c;
        tcm509_update_irq(s);
        break;
    case 0x0c:
        s->status |= 0x0040;
        tcm509_update_irq(s);
        break;
    case 0x0d:
        s->status &= ~(command & 0x0069);
        tcm509_update_irq(s);
        break;
    case 0x0e:
        s->irq_mask = command & 0xfe;
        tcm509_update_irq(s);
        break;
    case 0x0f:
        s->irq_zero = command & 0xfe;
        tcm509_update_irq(s);
        break;
    case 0x10:
        s->rx_filter = command & 0x0f;
        break;
    case 0x11:
        s->rx_early_thresh = command & 0x07fc;
        break;
    case 0x12:
        s->tx_avail_thresh = command & 0x07fc;
        break;
    case 0x13:
        s->tx_start_thresh = command & 0x07fc;
        break;
    case 0x15:
        s->net_status |= 0x0080;
        break;
    case 0x16:
        s->net_status &= ~0x0080;
        break;
    case 0x17:
        /* stop coaxial */
        break;
    default:
#ifdef TCM_DEBUG
        printf("3c509: Unhandled command 0x%02x\n", command >> 11);
#endif
        break;
    }
}

static void tcm509_transmit(TCM509State *d)
{
    int size = d->txbuffer[0] & 2047;
    if (TCM_LOOP_ENABLED(d)) {
#ifdef TCM_DEBUG
        printf("%s: loop size=%d\n", __func__, size);
#endif
        tcm509_receive(d, (uint8_t *)&d->txbuffer[2], size);
    } else {
#ifdef TCM_DEBUG
        printf("%s: xmit size=%d\n", __func__, size);
#endif
        //qemu_send_packet(d->nd, (uint8_t *)&d->txbuffer[2], size);
        network_tx(d->nic, &d->txbuffer[2], size);
    }
    if (d->txbuffer[0] & 0x8000)
        d->status |= 0x0004;
    d->status |= 0x0008;
    d->txpos = 0; d->txend = 0;
    d->tx_used++;
    d->tx_status[d->tx_tail++] = 0x80 | (d->txbuffer[0] & 0x8000? 0x40 : 0);
    if (d->tx_tail >= TCM_STATUS_SIZE)
        d->tx_tail = 0;
    if (TCM_STAT_ENABLED(d)) {
        if (((d->tx_bytes < 0x8000) && (d->tx_bytes + size)) ||
            (d->tx_frames == 0x80))
            d->status |= 0x0080;
        d->tx_bytes += size;
        d->tx_frames++;
    }
    tcm509_update_irq(d);
}

static void tcm509_eeprom_cmd(TCM509State *d, uint8_t cmd)
{
    int i;
    switch (cmd & 0x00d0) {
    case 0x80:
        d->eeprom_data = d->eeprom[cmd & 0x3f];
#ifdef TCM_DEBUG_EEPROM
        printf("read eeprom[0x%02x] = 0x%04x\n", 
                cmd & 0x3f, d->eeprom_data);
#endif
        break;
    case 0x40:
        if (d->eeprom_wren)
            d->eeprom[cmd & 0x3f] &= d->eeprom_data;
#ifdef TCM_DEBUG_EEPROM
        printf("write eeprom[0x%02x] = 0x%04x, wren=%d\n", 
                cmd & 0x3f, d->eeprom_data, d->eeprom_wren);
#endif
        break;
    case 0xd0:
        if (d->eeprom_wren)
            d->eeprom[cmd & 0x3f] = 0xffff;
#ifdef TCM_DEBUG_EEPROM
        printf("erase eeprom[0x%02x], wren=%d\n", 
                d->eeprom_data, d->eeprom_wren);
#endif
        break;
    case 0x00:
        switch (cmd & 0x0030) {
        case 0x30:
#ifdef TCM_DEBUG_EEPROM
            printf("eeprom write enable\n");
#endif
            d->eeprom_wren = 1;
            break;
        case 0x00:
#ifdef TCM_DEBUG_EEPROM
            printf("eeprom write disable\n");
#endif
            d->eeprom_wren = 0;
            break;
        case 0x20:
            if (d->eeprom_wren)
                for (i = 0x3f; i >= 0; i--)
                    d->eeprom[i] = 0xff;
#ifdef TCM_DEBUG_EEPROM
            printf("eeprom erase, wren=%d\n", d->eeprom_wren);
#endif
            break;
        case 0x10:
            if (d->eeprom_wren)
                for (i = 0x3f; i >= 0; i--)
                    d->eeprom[i] &= d->eeprom_data;
#ifdef TCM_DEBUG_EEPROM
            printf("eeprom write 0x%04x, wren=%d\n", 
                d->eeprom_data, d->eeprom_wren);
#endif
            break;
        }
        break;
    }
}

static void tcm509_ioport_write(uint16_t addr, uint16_t val, void *opaque)
{
    TCM509State *d = opaque;
    d->byteop = BYTEOP_NONE;

#ifdef TCM_DEBUG
    printf("%s: register %d:%x value=0x%04x\n", __func__, TCM_WINDOW(d), addr & 0x0f, val);
#endif    

    switch (((addr&0x0f)<<3)|TCM_WINDOW(d)) {
        /* Window 0 Registers - Setup */
    case 0x70:
        tcm509_command(d, val);
        break;
    case 0x60:
        d->eeprom_data = val;
        break;
    case 0x50:
        d->eeprom_cmd &= ~0x00ff;
        d->eeprom_cmd |= val & 0xff;
        tcm509_eeprom_cmd(d, val & 0xff);
        break;
    case 0x40:
        d->res_config &= ~0xf000;
        d->res_config |= val & 0xf000;
        {
            if (d->irq) {
                //pic_set_irq(d->irq, 0);
                picintc(1 << d->irq);
            }
            isapnp_write_reg(d->dev, 0, 0x70, val >> 12);
        }
        break;
    case 0x30:
        d->addr_config &= ~0xc09f;
        d->addr_config |= val & 0xc09f;
        val = (val & 0x1f) * 0x10 + 0x200;
        isapnp_write_reg(d->dev, 0, 0x40, val >> 8);
        isapnp_write_reg(d->dev, 0, 0x41, val & 0xff);
        break;
    case 0x20:
        d->config &= ~0x0005;
        d->config |= val & 0x0005;
        if (d->config & 0x0004)
            tcm509_reset(d);
        break;
        
        /* Window 1 Registers - Operating Set */
    case 0x71:
        tcm509_command(d, val);
        break;
    case 0x51:
        tcm509_ioport_write(addr, val & 0xff, opaque);
        tcm509_ioport_write(addr+1, val >> 8, opaque);
        break;
    case 0x11:
    case 0x01:
        if (d->txend) {
            if (d->txpos < d->txend) {
                d->txbuffer[d->txpos++] = val;
                if (d->txpos == d->txend)
                    tcm509_transmit(d);
            } else {
#ifdef TCM_DEBUG
                printf("%s: txpos > txend (%d > %d\n", __func__, d->txpos, d->txend);
#endif
                d->status |= 0x0002;
                tcm509_update_irq(d);
            }
        } else
        {
            d->txpos = 0;
            d->txbuffer[d->txpos++] = val;
            d->txend = 2 + (((3+(val & 2047))&~3)>>1);
        }
        break;

        /* Window 2 Registers - Station Address Setup/Read */
    case 0x72:
        tcm509_command(d, val);
        break;
    case 0x22:
        d->padr[2] = val;
        break;
    case 0x12:
        d->padr[1] = val;
        break;
    case 0x02:
        d->padr[0] = val;
        break;

        /* Window 3 Registers - FIFO Management */
    case 0x73:
        tcm509_command(d, val);
        break;
        
        /* Window 4 Registers - Diagnostic */
    case 0x74:
        tcm509_command(d, val);
        break;
    case 0x54:
        d->media_type &= ~0x00c6;
        d->media_type |= val & 0x00c6;
        break;
    case 0x44:
        /*ethernet controller status*/
        break;
    case 0x34:
        d->net_status &= ~0xf000;
        d->net_status |= val & 0xf000;
        break;
    case 0x24:
        /*fifo diagnostic*/
        break;

        /* Window 5 Registers - Command Results */
    case 0x75:
        tcm509_command(d, val);
        break;

        /* Window 6 Registers - Command Results */
    case 0x76:
        tcm509_command(d, val);
        break;
    case 0x66:
        d->tx_bytes = val;
        break;
    case 0x65:
        d->rx_bytes = val;
        break;
    case 0x46:
    case 0x36:
    case 0x26:
    case 0x16:
    case 0x06:
        tcm509_ioport_writeb(addr, val & 0xff, opaque);
        tcm509_ioport_writeb(addr+1, val >> 8, opaque);
        break;

        /* Window 7 Registers - Unused */
    case 0x77:
        tcm509_command(d, val);
        break;
    }
}

static uint16_t tcm509_ioport_read(uint16_t addr, void *opaque)
{
    TCM509State *d = opaque;
    d->byteop = BYTEOP_NONE;

#ifdef TCM_DEBUG
    printf("%s: register %d:%x\n", __func__, TCM_WINDOW(d), addr & 0x0f);
#endif    

    switch (((addr&0x0f)<<3)|TCM_WINDOW(d)) {
        /* Window 0 Registers - Setup */
    case 0x70:
        return TCM_STATUS(d);
    case 0x60:
        return d->eeprom_data;
    case 0x50:
        return d->eeprom_cmd;
    case 0x40:
        return d->res_config;
    case 0x30:
        return d->addr_config;
    case 0x20:
        return d->config;
    case 0x10:
        return d->eeprom[0x03];
    case 0x00:
        return d->eeprom[0x07];

        /* Window 1 Registers - Operating Set */
    case 0x71:
        return TCM_STATUS(d);
    case 0x61:
        return sizeof(d->txbuffer) - d->txpos*2;
    case 0x51:
        return tcm509_ioport_readb(addr, opaque) |
             ( tcm509_ioport_readb(addr+1, opaque)<<8 );
    case 0x41:
        d->status &= ~0x0010;
        if (d->fifo_used)
            return d->fifo[d->fifo_head].status;
        return 0x8000;
    case 0x11:
    case 0x01:
        if (d->fifo_used) {
            struct TCM509_FIFO *fifo = d->fifo+d->fifo_head;
            if (fifo->pos < fifo->max) {
                uint16_t value = fifo->packet[fifo->pos++];
                if (fifo->pos == fifo->max) {
                    d->status &= ~0x0010;
                    tcm509_update_irq(d);
                }
                return value;
           }
        }
#ifdef TCM_DEBUG        
        printf("%s: read too many (%d)\n",__func__, d->fifo_used);
#endif
        d->status |= 0x0002;
        tcm509_update_irq(d);        
        return -1;
        
        /* Window 2 Registers - Station Address Setup/Read */
    case 0x72:
        return TCM_STATUS(d);
    case 0x22:
        return d->padr[2];
    case 0x12:
        return d->padr[1];
    case 0x02:
        return d->padr[0];
        
        /* Window 3 Registers - FIFO Management */
    case 0x73:
        return TCM_STATUS(d);
    case 0x63:
        return sizeof(d->txbuffer) - d->txpos*2;
    case 0x53:
        return TCM_ETH_FRAME_MAX * (TCM_FIFO_SIZE - d->fifo_used);
    case 0x23:
        return 0 /* rom control */;
       
        /* Window 4 Registers - Diagnostic */
    case 0x74:
        return TCM_STATUS(d);
    case 0x54:
        return d->media_type;
    case 0x44:
        return 0x0000 /*ethernet controller status*/;
    case 0x34:
        return d->net_status;
    case 0x24:
        return 0x0000 /*fifo diagnostic*/;
        
        /* Window 5 Registers - Command Results & State */
    case 0x75:
        return TCM_STATUS(d);
    case 0x65:
        return d->irq_zero;
    case 0x55:
        return d->irq_mask;
    case 0x45:
        return d->rx_filter;
    case 0x35:
        return d->rx_early_thresh;
    case 0x15:
        return d->tx_avail_thresh;
    case 0x05:
        return d->tx_start_thresh;

        /* Window 6 Registers - Statistics */
    case 0x76:
        return TCM_STATUS(d);
    case 0x66:
        return tcm_read_zero(&d->tx_bytes);
    case 0x56:
        return tcm_read_zero(&d->rx_bytes);
    case 0x46:
    case 0x36:
    case 0x26:
    case 0x16:
    case 0x06:
        return tcm509_ioport_readb(addr, opaque) |
             ( tcm509_ioport_readb(addr+1, opaque) << 8);

        /* Window 7 Registers */
    case 0x77:
        return TCM_STATUS(d);
        
    default:
        return 0;
    }
}

static void tcm509_ioport_writel(uint16_t addr, uint32_t val, void *opaque)
{
#ifdef TCM_DEBUG
    TCM509State *d = opaque;
    printf("%s: register %d:%x value=0x%04ulx\n", __func__, TCM_WINDOW(d), addr & 0x0f, val);
#endif    

    tcm509_ioport_write(addr, val & 0xffff, opaque);
    tcm509_ioport_write(addr+2, val >> 16, opaque);    
}

static uint32_t tcm509_ioport_readl(uint16_t addr, void *opaque)
{
    uint32_t value;

#ifdef TCM_DEBUG
    TCM509State *d = opaque;
    printf("%s: register %d:%x\n", __func__, TCM_WINDOW(d), addr & 0x0f);
#endif    

    value = tcm509_ioport_read(addr, opaque);
    value |= tcm509_ioport_read(addr+2, opaque)<<16;
    return value;
}

static void tcm509_ioport_writeb(uint16_t addr, uint8_t val, void *opaque)
{
    TCM509State *d = opaque;
    TCM_ByteOp byteop = d->byteop;
    uint32_t byteaddr = d->byteaddr;
    d->byteop = BYTEOP_NONE;
    d->byteaddr = addr;

#ifdef TCM_DEBUG
    printf("%s: register %d:%x value=0x%02x\n", __func__, TCM_WINDOW(d), addr & 0x0f, val);
#endif    

    switch (((addr&0x0f)<<3)|TCM_WINDOW(d)) {
        /* Window 6 Registers - Statistics */
    case 0x46:
        d->tx_defer = val;
        break;
    case 0x3e:
        d->rx_frames = val;
        break;
    case 0x36:
        d->tx_frames = val;
        break;
    case 0x2e:
        d->rx_discard = val;
        break;
    case 0x26:
        d->tx_late = val;
        break;
    case 0x1e:
        d->tx_onecol = val;
        break;
    case 0x16:
        d->tx_collisons = val;
        break;
    case 0x0e:
        d->tx_nocd = val;
        break;
    case 0x06:
        d->cd_lost = val;
        break;
    default:
        if ((byteop == BYTEOP_WRITE) && (byteaddr == addr))
            tcm509_ioport_write(addr, (val << 8) | d->bytedata, opaque);
        else {
            d->byteop = BYTEOP_WRITE;
            d->byteaddr = addr;
            d->bytedata = val;
        }
    }
}

static uint8_t tcm509_ioport_readb(uint16_t addr, void *opaque)
{
    TCM509State *d = opaque;
    TCM_ByteOp byteop = d->byteop;
    uint32_t byteaddr = d->byteaddr;
    d->byteop = BYTEOP_NONE;
    d->byteaddr = addr;

#ifdef TCM_DEBUG
    printf("%s: register %d:%x\n", __func__, TCM_WINDOW(d), addr & 0x0f);
#endif    
    
    switch (((addr&0x0f)<<3)|TCM_WINDOW(d)) {

        /* Window 1 Registers - Operational */
    case 0x59:
        if (d->tx_used) {
            uint8_t result = d->tx_status[d->tx_head++];
            if (d->tx_head >= TCM_STATUS_SIZE)
                d->tx_head = 0;
            d->tx_used--;
            if (d->tx_used == 0) {
                d->status &= ~0x0004;
                tcm509_update_irq(d);
            }
            return result;
        } else
            return d->tx_status[d->tx_head] & ~0x80;
    case 0x51:
        return 0xff; /* timer */

        /* Window 6 Registers - Statistics */
    case 0x4e:
        return 0;       /* unused */
    case 0x46:
        return tcm_readb_zero(&d->tx_defer);
    case 0x3e:
        return tcm_readb_zero(&d->rx_frames);
    case 0x36:
        return tcm_readb_zero(&d->tx_frames);        
    case 0x2e:
        return tcm_readb_zero(&d->rx_discard);
    case 0x26:
        return tcm_readb_zero(&d->tx_late);
    case 0x1e:
        return tcm_readb_zero(&d->tx_onecol);
    case 0x16:
        return tcm_readb_zero(&d->tx_collisons);
    case 0x0e:
        return tcm_readb_zero(&d->tx_nocd);
    case 0x06:
        return tcm_readb_zero(&d->cd_lost);
    default:
        if ((byteop == BYTEOP_READ) && (byteaddr == addr)) {
            d->byteop = BYTEOP_NONE;
            return d->bytedata >> 8;
        }
        d->bytedata = tcm509_ioport_read(addr, opaque);
        d->byteop = BYTEOP_READ;
        d->byteaddr = addr;
        return d->bytedata & 0xff;
    }
}


static int tcm509_can_receive(void *opaque)
{
    TCM509State *d = opaque;
    if (d->activated && TCM_RX_ENABLED(d) && d->fifo_used < TCM_FIFO_SIZE)
    {
        return TCM_ETH_FRAME_MAX;
    }
    return 0;
}

static int tcm509_receive(void *opaque, uint8_t *buf, int size)
{
    TCM509State *d = opaque;

#ifdef TCM_DEBUG
    printf("%s: size=%d\n", __func__, size);        
#endif
    int ret = 0;
    if (d->activated && TCM_RX_ENABLED(d) && (d->fifo_used < TCM_FIFO_SIZE) &&
        (TCM_RX_PROM(d) || ladr_match(d, buf, size) || 
         padr_bcast(d, buf, size) || padr_match(d, buf, size)))
    {
        struct TCM509_FIFO *fifo = d->fifo+d->fifo_tail;
        int runt = size < TCM_ETH_FRAME_RUNT;
        int oversize = size > TCM_ETH_FRAME_OVERSIZE;
        
        fifo->status = 0x8000;

        memcpy(fifo->packet, buf, size);
        
        fifo->status |= size;
        if (runt)
            fifo->status |= 0x5800;
        else
        if (oversize)
            fifo->status |= 0x4800;
        d->fifo_tail++;
        if (d->fifo_tail >= TCM_FIFO_SIZE)
          d->fifo_tail = 0;        
        d->fifo_used++;
        
        if (size < 8) size = 8;
        fifo->max = 1 + (((size+3)&~3)>>1);
        fifo->pos = 0;
        fifo->status &= ~0x8000;
        
        if (TCM_STAT_ENABLED(d) && !(fifo->status & 0x4000)) {
            if (((d->rx_bytes < 0x8000) && (d->rx_bytes + size > 0x8000)) ||
                (d->rx_frames == 0x80))
                d->status |= 0x0080;
            d->rx_bytes += size;      
            d->rx_frames++;    
        }
    } else {
        ret = 0;
    }
    tcm509_update_rx(d);
    return ret;
}

static void tcm509_activate(uint8_t ld, isapnp_device_config_t *config, void *priv)
{
    TCM509State *dev = (TCM509State *)priv;

    if (ld)
        return;

    pclog("Card activated\n");

    if (dev->base_address) {
        io_removehandler(dev->base_address, 16, tcm509_ioport_readb, tcm509_ioport_read, tcm509_ioport_readl, tcm509_ioport_writeb, tcm509_ioport_write, tcm509_ioport_writel, dev);
        dev->base_address = 0;
    }

    dev->base_address = 0;
    dev->irq     = 0;
    dev->res_config &= ~0xf000;
    dev->addr_config &= ~0x001f;
    if (config->activate) {
        dev->base_address = config->io[0].base;
        if (dev->base_address != ISAPNP_IO_DISABLED)
            io_sethandler(dev->base_address, 16, tcm509_ioport_readb, tcm509_ioport_read, tcm509_ioport_readl, tcm509_ioport_writeb, tcm509_ioport_write, tcm509_ioport_writel, dev);

        dev->irq    = config->irq[0].irq;

        dev->res_config |= dev->irq << 12;
        dev->addr_config |= (dev->base_address - 0x200)>>4;
        dev->eeprom[0x08] = dev->base_address;
        dev->eeprom[0x09] = dev->irq << 12;
    }
}

static void isa_pnp_config_checksum(uint8_t* data, uint8_t *end, uint32_t* size)
{
    uint8_t *config = data;
    uint8_t csum = 0x6a;
    int i;
    
    /* fixup pnp serial */
    for (i = 0; i < 64; i++) {
        int bit = (config[i/8] & (1<<(i&7))) != 0;
        csum = (csum >> 1) | (((csum ^ (csum >> 1) ^ bit) << 7) & 0xff);
    }
    config[8] = csum;
    
    /* Add end tag */
    *(end++) = 0x78;
    csum = 0;
    while (config != end) {
        csum += *(config++);
    }
    *(end++) = -csum;
    //device->config_size = end - device->config;
    *size = end - data;
#ifdef PNP_DEBUG
    printf("%s: config size = %d\n",device->name, device->config_size);
#endif
}

void* pnp_3c509b_init(const device_t* info)
{
    TCM509State *d;
    uint8_t *pnp_conf;
    int i;
    uint16_t j, k;

    d = calloc(sizeof(TCM509State), 1);
    d->macaddr[0] = 0x00;
    d->macaddr[1] = 0x01;
    d->macaddr[2] = 0x02;
    uint32_t mac = device_get_config_mac("mac", -1);
    if (mac & 0xff000000) {
        /* Generate new local MAC. */
        d->macaddr[3] = random_generate();
        d->macaddr[4] = random_generate();
        d->macaddr[5] = random_generate();
        mac              = (((int) d->macaddr[3]) << 16);
        mac             |= (((int) d->macaddr[4]) << 8);
        mac             |= ((int) d->macaddr[5]);
        device_set_config_mac("mac", mac);
    } else {
        d->macaddr[3] = (mac >> 16) & 0xff;
        d->macaddr[4] = (mac >> 8) & 0xff;
        d->macaddr[5] = (mac & 0xff);
    }
    d->nic = network_attach(d, d->macaddr, tcm509_receive, NULL);
    /* Initialize the EEPROM */
    d->eeprom[0x00] = (d->macaddr[0] << 8) | d->macaddr[1];
    d->eeprom[0x01] = (d->macaddr[2] << 8) | d->macaddr[3];
    d->eeprom[0x02] = (d->macaddr[4] << 8) | d->macaddr[5];
    d->eeprom[0x03] = 0x9050;
    d->eeprom[0x04] = TCM_MFG_DATE(2004,07,07);
    d->eeprom[0x07] = 0x6d50;
    
    d->eeprom[0x08] = 0x0080;   /* addr config reg */
    d->eeprom[0x09] = 0x0f00;   /* res config reg */

    d->eeprom[0x08] = 0;
    d->eeprom[0x09] = 0;
    
    d->eeprom[0x0a] = d->eeprom[0x00];
    d->eeprom[0x0b] = d->eeprom[0x01];
    d->eeprom[0x0c] = d->eeprom[0x02];
    d->eeprom[0x0d] = (19<<8)|(2<<4);
    d->eeprom[0x0e] = 0x0000;
    d->eeprom[0x10] = 0x2083;
    d->eeprom[0x14] = 0x0001;
        
    /* PnP Configuration Data */
    uint8_t* pnp_config = (uint8_t *)(d->eeprom + 0x18);
    pnp_conf = pnp_config;

    *(pnp_conf++) = 0x50;
    *(pnp_conf++) = 0x6D;
    *(pnp_conf++) = 0x50;
    *(pnp_conf++) = 0x90;

    *(pnp_conf++) = 0x00;
    *(pnp_conf++) = 0x00;
    *(pnp_conf++) = 0x00;
    *(pnp_conf++) = 0x00;
    *(pnp_conf++) = 0x00;

    *(pnp_conf++) = 0x0a;       /* small item, PnP version */
    *(pnp_conf++) = 0x10;       /* Version 1.0 */
    *(pnp_conf++) = 0x00;       /* vendor version */
    
    *(pnp_conf++) = 0x82;      /* Large item, type id string */
    *(pnp_conf++) = 0x06;
    *(pnp_conf++) = 0x00;
    memcpy(pnp_conf, "3c509B", 6);
    pnp_conf += 6;
    
    /* logical device */
    *(pnp_conf++) = 0x15;       /* small item, logical device */
    *(uint32_t *)pnp_conf = *(uint32_t *)pnp_config; pnp_conf+=4;
    *(pnp_conf++) = 0x01;       /* logical device flags[0] */
    
    *(pnp_conf++) = 0x1c;
    *(pnp_conf++) = 0x41;
    *(pnp_conf++) = 0xd0;
    *(pnp_conf++) = 0x80;
    *(pnp_conf++) = 0xF7;

    /* io port descriptor */
    *(pnp_conf++) = 0x47;       /* small iten, io port */
    *(pnp_conf++) = 0x01;       /* 16bit decode */
    *(uint16_t *)pnp_conf = 0x280; pnp_conf+=2;
    *(uint16_t *)pnp_conf = 0x3e0; pnp_conf+=2;
    *(pnp_conf++) = 0x10;       /* alignment */
    *(pnp_conf++) = 0x10;       /* ports */
        
    /* irq format */
    *(pnp_conf++) = 0x23;       /* small item, irq */
    *(pnp_conf++) = 0xa8;       /* irq 0-7 */
    *(pnp_conf++) = 0x9e;       /* irq 8-15 */
    *(pnp_conf++) = 0x01;

    *(pnp_conf++) = 0x79;

    /* Calculate the 3Com EEPROM Checksums */    
    for (i = 0x00, j = 0x0000, k = 0x0000; i < 0x0f; i++) {
        switch (i) {
        case 0x08:
        case 0x09:
        case 0x0d:
            k ^= d->eeprom[i];
            break;
        default:
            j ^= d->eeprom[i];
            break;
        }
    }
    d->eeprom[0x0f] = ((j ^ (j << 8)) & 0xff00) | (( k ^ ( k >> 8)) & 0x00ff);

    for (i = 0x10, j = 0x0000, k = 0x0000; i <= 0x40; i++) {
        switch (i) {
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x16:
            k ^= d->eeprom[i];
            break;
        case 0x17:
        case 0x18:
        case 0x19:
        case 0x1a:
        case 0x1b:
        case 0x1c:
        case 0x1d:
        case 0x1e:
        case 0x1f:
            break;
        default:
            j ^= d->eeprom[i];
            break;
        }
    }
    d->eeprom[0x17] = ((j ^ (j << 8)) & 0xff00) | (( k ^ ( k >> 8)) & 0x00ff);

    /* end tag */
    uint32_t pnp_size = pnp_conf - (uint8_t*)&d->eeprom[0x18];
    {
        uint8_t res = 0;
        for (uint16_t j = 9; j <= i; j++)
            res += *(((uint8_t*)&d->eeprom[0x18]) + j);
        *(pnp_conf++) = -res;
        pnp_size++;
    }
    //isa_pnp_config_checksum((uint8_t*)&d->eeprom[0x18], pnp_conf, &pnp_size);
    {
        FILE* f = fopen("pnp.rom", "wb");
        fwrite(&d->eeprom[0x18], pnp_size + 1, 1, f);
        fclose(f);
    }
    d->dev = isapnp_add_card(pnp_config, pnp_size, tcm509_activate, NULL, NULL, NULL, d);

    return d;
}

static void
nic_close(void *priv)
{
    free(priv);
}

static const device_config_t tcm509_config[] = {
    {
        .name           = "mac",
        .description    = "MAC Address",
        .type           = CONFIG_MAC,
        .default_string = NULL,
        .default_int    = -1,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = { { 0 } },
        .bios           = { { 0 } }
    },
    { .name = "", .description = "", .type = CONFIG_END }
};
// clang-format on

const device_t tcm509_device = {
    .name          = "3Com EtherLink III",
    .internal_name = "tcm509",
    .flags         = DEVICE_ISA16,
    .local         = 0,
    .init          = pnp_3c509b_init,
    .close         = nic_close,
    .reset         = tcm509_reset,
    .available     = NULL,
    .speed_changed = NULL,
    .force_redraw  = NULL,
    .config        = tcm509_config
};
