#include <stdint.h>
#include <stdbool.h>

struct pcmcia_socket_t;
typedef struct pcmcia_socket_t pcmcia_socket_t;

struct pcmcia_socket_t {
    /* Socket number (zero-based) */
    uint8_t socket_num;

    /* Card I/O Read and Write functions. */
    uint8_t (*io_read)(uint16_t port, void* priv);
    uint16_t (*io_readw)(uint16_t port, void* priv);
    void (*io_write)(uint16_t port, uint8_t val, void* priv);
    void (*io_writew)(uint16_t port, uint16_t val, void* priv);

    /* Card Memory Read and Write functions. */
    /* REG = 0: Access Attribute Memory. */
    /* REG = 1: Access Common Memory. */
    uint8_t (*mem_read)(uint32_t addr, int reg, void* priv);
    uint16_t (*mem_readw)(uint32_t addr, int reg, void* priv);
    void (*mem_write)(uint32_t addr, uint8_t val, int reg, void* priv);
    void (*mem_writew)(uint32_t addr, uint16_t val, int reg, void* priv);

    /* Fetch exec address of card memory, for EMS/XIP. */
    void* (*mem_get_exec)(uint32_t addr);

    /* Signals power status change to the card. */
    void (*power_status_change)(int vcc, int vpp, pcmcia_socket_t* socket);

    /* Signals card insertion/removal to the socket. */
    void (*card_inserted)(bool inserted, pcmcia_socket_t* socket);

    /* Signals STSCHG to the socket. */
    void (*status_changed)(bool status, pcmcia_socket_t* socket);

    /* Signals READY to the socket. */
    void (*ready_changed)(bool ready, pcmcia_socket_t* socket);

    /* Signals interrupt to the socket. */
    void (*interrupt)(bool set, pcmcia_socket_t* socket);

    /* Resets the card. */
    void (*reset)(void *priv);

    /* Notifies the card of a ATA mode. */
    void (*ata_mode)(bool ata, pcmcia_socket_t* socket);

    /* Opaque pointer to card-specific information. */
    void *card_priv;

    /* Opaque pointer to socket-specific information. */
    void *socket_priv;

    /* Card power status. */
    bool powered_on;
};

typedef struct pcmcia_socket_t pcmcia_socket_t;

bool pcmcia_socket_is_free(pcmcia_socket_t* socket);
pcmcia_socket_t* pcmcia_search_for_slots(void);
/* Set up any relevant stuff to your card before calling this. */
void pcmcia_socket_insert_card(pcmcia_socket_t* socket, void* priv);
void pcmcia_reset(void);
void pcmcia_register_socket(pcmcia_socket_t *socket);
