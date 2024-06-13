#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#define HAVE_STDARG_H
#include <86box/86box.h>
#include <86box/io.h>
#include <86box/device.h>
#include <86box/timer.h>
#include <86box/pcmcia.h>

static pcmcia_socket_t *pcmcia_sockets[4];
static uint8_t          pcmcia_registered_sockets_num = 0;

extern const device_t pd6710_device;

void
pcmcia_reset(void)
{
    pcmcia_registered_sockets_num = 0;
}

void
pcmcia_register_socket(pcmcia_socket_t *socket)
{
    pcmcia_sockets[pcmcia_registered_sockets_num++] = socket;
}

bool
pcmcia_socket_is_free(pcmcia_socket_t *socket)
{
    return !socket->card_priv;
}

pcmcia_socket_t *
pcmcia_search_for_slots(void)
{
    if (!pcmcia_registered_sockets_num) {
        device_add(&pd6710_device);
    }
    for (int i = 0; i < pcmcia_registered_sockets_num; i++) {
        if (pcmcia_sockets[i] && !pcmcia_sockets[i]->card_priv)
            return pcmcia_sockets[i];
    }
    return NULL;
}

void
pcmcia_socket_insert_card(pcmcia_socket_t *socket, void* priv)
{
    if (!socket->card_inserted)
        fatal("No PCMCIA socket insertion function!\n");

    socket->card_priv = priv;
    socket->card_inserted(true, socket);
}
