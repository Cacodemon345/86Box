#include <QApplication>
#ifndef __APPLE__
#include <X11/Xlib.h>
#include <X11/Xdefs.h>
#include <X11/keysym.h>
#endif
#include <unordered_map>
#include <array>
#include <cstdlib>
#include <cinttypes>
#include <stdio.h>

std::array<uint32_t, 256> x11_to_xt_base
{
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0x01,
    0x02,
    0x03,
    0x04,
    0x05,
    0x06,
    0x07,
    0x08,
    0x09,
    0x0A,
    0x0B,
    0x0C,
    0x0D,
    0x0E,
    0x0F,
    0x10,
    0x11,
    0x12,
    0x13,
    0x14,
    0x15,
    0x16,
    0x17,
    0x18,
    0x19,
    0x1A,
    0x1B,
    0x1C,
    0x1D,
    0x1E,
    0x1F,
    0x20,
    0x21,
    0x22,
    0x23,
    0x24,
    0x25,
    0x26,
    0x27,
    0x28,
    0x29,
    0x2A,
    0x2B,
    0x2C,
    0x2D,
    0x2E,
    0x2F,
    0x30,
    0x31,
    0x32,
    0x33,
    0x34,
    0x35,
    0x36,
    0x37,
    0x38,
    0x39,
    0x3A,
    0x3B,
    0x3C,
    0x3D,
    0x3E,
    0x3F,
    0x40,
    0x41,
    0x42,
    0x43,
    0x44,
    0x45,
    0x46,
    0x47,
    0x48,
    0x49,
    0x4A,
    0x4B,
    0x4C,
    0x4D,
    0x4E,
    0x4F,
    0x50,
    0x51,
    0x52,
    0x53,
    0x54,
    0x55,
    0x56,
    0x57,
    0x58,
    0x147,
    0x148,
    0x149,
    0,
    0x14B,
    0,
    0x14D,
    0x14F,
    0x150,
    0x151,
    0x152,
    0x153,
    0x11C,
    0x11D,
    0, // Pause/Break key.
    0x137,
    0x135,
    0x138,
    0, // Ditto as above comment.
    0x15B,
    0x15C,
    0x15D,
};

std::array<uint32_t, 256> x11_to_xt_2
{
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0x01,
    0x02,
    0x03,
    0x04,
    0x05,
    0x06,
    0x07,
    0x08,
    0x09,
    0x0A,
    0x0B,
    0x0C,
    0x0D,
    0x0E,
    0x0F,
    0x10,
    0x11,
    0x12,
    0x13,
    0x14,
    0x15,
    0x16,
    0x17,
    0x18,
    0x19,
    0x1A,
    0x1B,
    0x1C,
    0x1D,
    0x1E,
    0x1F,
    0x20,
    0x21,
    0x22,
    0x23,
    0x24,
    0x25,
    0x26,
    0x27,
    0x28,
    0x29,
    0x2A,
    0x2B,
    0x2C,
    0x2D,
    0x2E,
    0x2F,
    0x30,
    0x31,
    0x32,
    0x33,
    0x34,
    0x35,
    0x36,
    0x37,
    0x38,
    0x39,
    0x3A,
    0x3B,
    0x3C,
    0x3D,
    0x3E,
    0x3F,
    0x40,
    0x41,
    0x42,
    0x43,
    0x44,
    0x45,
    0x46,
    0x47,
    0x48,
    0x49,
    0x4A,
    0x4B,
    0x4C,
    0x4D,
    0x4E,
    0x4F,
    0x50,
    0x51,
    0x52,
    0x53,
    0x54,
    0x55,
    0x56,
    0x57,
    0x58,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0x11C,
    0x11D,
    0x135,
    0x137,
    0x138,
    0,
    0x147,
    0x148,
    0x149,
    0x14B,
    0x14D,
    0x14F,
    0x150,
    0x151,
    0x152,
    0x153
};

std::array<uint32_t, 256> x11_to_xt_vnc
{
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0x1D,
    0x11D,
    0x2A,
    0x36,
    0,
    0,
    0x38,
    0x138,
    0x39,
    0x0B,
    0x02,
    0x03,
    0x04,
    0x05,
    0x06,
    0x07,
    0x08,
    0x09,
    0x0A,
    0x0C,
    0x0D,
    0x1A,
    0x1B,
    0x27,
    0x28,
    0x29,
    0x33,
    0x34,
    0x35,
    0x2B,
    0x1E,
    0x30,
    0x2E,
    0x20,
    0x12,
    0x21,
    0x22,
    0x23,
    0x17,
    0x24,
    0x25,
    0x26,
    0x32,
    0x31,
    0x18,
    0x19,
    0x10,
    0x13,
    0x1F,
    0x14,
    0x16,
    0x2F,
    0x11,
    0x2D,
    0x15,
    0x2C,
    0x0E,
    0x1C,
    0x0F,
    0x01,
    0x153,
    0x147,
    0x14F,
    0x149,
    0x151,
    0x148,
    0x150,
    0x14B,
    0x14D,
};

std::array<uint32_t, 256> darwin_to_xt
{
    0x1E,
    0x1F,
    0x20,
    0x21,
    0x23,
    0x22,
    0x2C,
    0x2D,
    0x2E,
    0x2F,
    0x2B,
    0x30,
    0x10,
    0x11,
    0x12,
    0x13,
    0x15,
    0x14,
    0x02,
    0x03,
    0x04,
    0x05,
    0x07,
    0x06,
    0x0D,
    0x0A,
    0x08,
    0x0C,
    0x09,
    0x0B,
    0x1B,
    0x18,
    0x16,
    0x1A,
    0x17,
    0x19,
    0x1C,
    0x26,
    0x24,
    0x28,
    0x25,
    0x27,
    0x2B,
    0x33,
    0x35,
    0x31,
    0x32,
    0x34,
    0x0F,
    0x39,
    0x29,
    0x0E,
    0x11C,
    0x01,
    0x15C,
    0x15B,
    0x2A,
    0x3A,
    0x38,
    0x1D,
    0x36,
    0x138,
    0x11D,
    0x15C,
    0,
    0x53,
    0,
    0x37,
    0,
    0x4E,
    0,
    0x45,
    0x130,
    0x12E,
    0x120,
    0x135,
    0x11C,
    0,
    0x4A,
    0,
    0,
    0,
    0x52,
    0x4F,
    0x50,
    0x51,
    0x4B,
    0x4C,
    0x4D,
    0x47,
    0,
    0x48,
    0x49,
    0,
    0,
    0,
    0x3F,
    0x40,
    0x41,
    0x3D,
    0x42,
    0x43,
    0,
    0x57,
    0,
    0x137,
    0,
    0x46,
    0,
    0x44,
    0x15D,
    0x58,
    0,
    0, // Pause/Break key.
    0x152,
    0x147,
    0x149,
    0x153,
    0x3E,
    0x14F,
    0x3C,
    0x151,
    0x3B,
    0x14B,
    0x14D,
    0x150,
    0x148,
    0,
};

static std::array<uint32_t, 256>& selected_keycode = x11_to_xt_base;
uint16_t x11_keycode_to_keysym(uint32_t keycode)
{
#ifdef __APPLE__
    return darwin_to_xt[keycode];
#else
    static Display* x11display = nullptr;
    if (QApplication::platformName().contains("wayland"))
    {
        selected_keycode = x11_to_xt_2;
    }
    else if (!x11display)
    {
        x11display = XOpenDisplay(nullptr);
        if (XKeysymToKeycode(x11display, XK_Home) == 110)
        {
            selected_keycode = x11_to_xt_2;
        }
        else if (XKeysymToKeycode(x11display, XK_Home) == 69)
        {
            selected_keycode = x11_to_xt_vnc;
        }
    }
    return selected_keycode[keycode];
#endif
}
