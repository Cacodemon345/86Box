/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          Interface to SDL3.
 *
 * Authors: Nishi
 *          Cacodemon345
 *
 *          Copyright 2025 Nishi.
 *          Copyright 2026 Cacodemon345.
 */
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include <SDL3/SDL.h>

#include <86box/86box.h>
#include <86box/midi.h>
#include <86box/sound.h>
#include <86box/plat_unused.h>

#define I_NORMAL 0
#define I_MUSIC 1
#define I_WT 2
#define I_CD 3
#define I_FDD 4
#define I_HDD 5
#define I_MIDI 6



static SDL_AudioStream* audio[7] = {0, 0, 0, 0, 0, 0, 0};
static SDL_AudioStream* ext_audio[256];
static uint32_t ext_audio_num = 0;
static SDL_AudioDeviceID dev = 0;
extern bool fast_forward;

static int freqs[7] = {SOUND_FREQ, MUSIC_FREQ, WT_FREQ, CD_FREQ, SOUND_FREQ, SOUND_FREQ, 0};
void
closeal(void)
{
    for (int i = 0; i < sizeof(audio) / sizeof(audio[0]); i++) {
        if (audio[i] != 0)
            SDL_DestroyAudioStream(audio[i]);

        audio[i] = 0;
    }
    for (int i = 0; i < ext_audio_num; i++) {
        if (ext_audio[i])
            SDL_DestroyAudioStream(ext_audio[i]);

        ext_audio[i] = 0;
    }
    SDL_CloseAudioDevice(dev);
    dev = 0;
}

void
inital(void)
{
    int init_midi = 0;

    const char *mdn = midi_out_device_get_internal_name(midi_output_device_current);
    if ((strcmp(mdn, "none") != 0) && (strcmp(mdn, SYSTEM_MIDI_INTERNAL_NAME) != 0))
        init_midi = 1; /* If the device is neither none, nor system MIDI, initialize the
                          MIDI buffer and source, otherwise, do not. */

    ext_audio_num = 0;
    if (SDL_Init(SDL_INIT_AUDIO)) {
        dev = SDL_OpenAudioDevice(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, NULL);
        if (dev) {
            SDL_AudioSpec spec;
            int i = 0;
            for (i = 0; i < sizeof(audio) / sizeof(audio[0]); i++) {
                if (i == I_MIDI && !init_midi)
                    break;
                memset(&spec, 0, sizeof(SDL_AudioSpec));
                spec.channels = 2;
                spec.format = sound_is_float ? SDL_AUDIO_F32LE : SDL_AUDIO_S16LE;
                spec.freq = freqs[i];
                audio[i] = SDL_CreateAudioStream(&spec, &spec);
            }
            SDL_BindAudioStreams(dev, audio, i);
        }
    }
}

void
givealbuffer_common(const void *buf, SDL_AudioStream* src, const uint32_t bytes)
{
    double gain = 0.0;
    if (src == 0 || fast_forward)
        return;
    gain = sound_muted ? 0.0 : pow(10.0, (double) sound_gain / 20.0);

    SDL_SetAudioStreamGain(src, gain);
    SDL_PutAudioStreamData(src, buf, bytes);
}

void
givealbuffer(const void *buf)
{
    givealbuffer_common(buf, audio[I_NORMAL], (SOUNDBUFLEN << 1) * (sound_is_float ? sizeof(float) : sizeof(int16_t)));
}

void
givealbuffer_music(const void *buf)
{
    givealbuffer_common(buf, audio[I_MUSIC], (MUSICBUFLEN << 1) * (sound_is_float ? sizeof(float) : sizeof(int16_t)));
}

void
givealbuffer_wt(const void *buf)
{
    givealbuffer_common(buf, audio[I_WT], (WTBUFLEN << 1) * (sound_is_float ? sizeof(float) : sizeof(int16_t)));
}

void
givealbuffer_cd(const void *buf)
{
    givealbuffer_common(buf, audio[I_CD], (CD_BUFLEN << 1) * (sound_is_float ? sizeof(float) : sizeof(int16_t)));
}

void
givealbuffer_fdd(const void *buf, const uint32_t size)
{
    givealbuffer_common(buf, audio[I_FDD], ((int) size) * (sound_is_float ? sizeof(float) : sizeof(int16_t)));
}

void
givealbuffer_hdd(const void *buf, const uint32_t size)
{
    givealbuffer_common(buf, audio[I_HDD], ((int) size) * (sound_is_float ? sizeof(float) : sizeof(int16_t)));
}

void
givealbuffer_midi(const void *buf, const uint32_t size)
{
    givealbuffer_common(buf, audio[I_MIDI], ((int) size) * (sound_is_float ? sizeof(float) : sizeof(int16_t)));
}

void*
sound_backend_add_source(void)
{
    SDL_AudioSpec spec;
    spec.channels = 2;
    spec.format = sound_is_float ? SDL_AUDIO_F32 : SDL_AUDIO_S16;
    spec.freq = 44100;
    ext_audio[ext_audio_num] = SDL_CreateAudioStream(&spec, &spec);
    if (ext_audio[ext_audio_num]) {
        SDL_BindAudioStream(dev, ext_audio[ext_audio_num]);
    }
    return ext_audio[ext_audio_num];
}

void
sound_backend_buffer(void *priv, void *buf, uint32_t bytes)
{
    return givealbuffer_common(buf, priv, bytes);
}

int
sound_backend_set_format(void *priv, uint8_t *format, uint8_t *channels, uint32_t *freq)
{
    SDL_AudioSpec spec;
    spec.channels = *channels;
    switch (*format) {
        case SOUND_U8:
            spec.format = SDL_AUDIO_U8;
            break;
        case SOUND_FLOAT32:
            spec.format = SDL_AUDIO_F32;
            break;
        case SOUND_S16:
        default:
            spec.format = SDL_AUDIO_S16;
            break;
    }
    spec.freq = *freq;
    return SDL_SetAudioStreamFormat(priv, &spec, &spec);
}

void
al_set_midi(const int freq, UNUSED(const int buf_size))
{
    freqs[I_MIDI] = freq;
    if (audio[I_MIDI]) {
        SDL_AudioSpec spec;
        spec.freq = freq;
        spec.channels = 2;
        spec.format = sound_is_float ? SDL_AUDIO_F32LE : SDL_AUDIO_S16LE;
        SDL_SetAudioStreamFormat(audio[I_MIDI], &spec, &spec);
    }
}
