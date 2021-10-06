/*
 * 86Box	A hypervisor and IBM PC system emulator that specializes in
 *		running old operating systems and software designed for IBM
 *		PC systems and compatibles from 1981 through fairly recent
 *		system designs based on the PCI bus.
 *
 *		This file is part of the 86Box distribution.
 *
 *		Rendering module for libSDL2
 *
 *
 * Authors:	Fred N. van Kempen, <decwiz@yahoo.com>
 *		Michael Dr�ing, <michael@drueing.de>
 *      Cacodemon345
 *
 *		Copyright 2018-2020 Fred N. van Kempen.
 *		Copyright 2018-2020 Michael Dr�ing.
 *      Copyright 2021 Cacodemon345.
 *
 *		Redistribution and  use  in source  and binary forms, with
 *		or  without modification, are permitted  provided that the
 *		following conditions are met:
 *
 *		1. Redistributions of  source  code must retain the entire
 *		   above notice, this list of conditions and the following
 *		   disclaimer.
 *
 *		2. Redistributions in binary form must reproduce the above
 *		   copyright  notice,  this list  of  conditions  and  the
 *		   following disclaimer in  the documentation and/or other
 *		   materials provided with the distribution.
 *
 *		3. Neither the  name of the copyright holder nor the names
 *		   of  its  contributors may be used to endorse or promote
 *		   products  derived from  this  software without specific
 *		   prior written permission.
 *
 * THIS SOFTWARE  IS  PROVIDED BY THE  COPYRIGHT  HOLDERS AND CONTRIBUTORS
 * "AS IS" AND  ANY EXPRESS  OR  IMPLIED  WARRANTIES,  INCLUDING, BUT  NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE  ARE  DISCLAIMED. IN  NO  EVENT  SHALL THE COPYRIGHT
 * HOLDER OR  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT  NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES;  LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON  ANY
 * THEORY OF  LIABILITY, WHETHER IN  CONTRACT, STRICT  LIABILITY, OR  TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING  IN ANY  WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <SDL.h>
#include <SDL_messagebox.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/param.h>
/* This #undef is needed because a SDL include header redefines HAVE_STDARG_H. */
#undef HAVE_STDARG_H
#define HAVE_STDARG_H
#include <86box/86box.h>
#include <86box/device.h>
#include <86box/plat.h>
#include <86box/plat_dynld.h>
#include <86box/video.h>
#include <86box/ui.h>
#include <86box/version.h>
#include <86box/unix_sdl.h>
#include <incbin.h>

#define RENDERER_FULL_SCREEN	1
#define RENDERER_HARDWARE	2
#define RENDERER_OPENGL		4

typedef struct sdl_blit_params
{
    int x, y, w, h;
} sdl_blit_params;
extern sdl_blit_params params;
extern int blitreq;

SDL_Window	*sdl_win = NULL;
SDL_Renderer	*sdl_render = NULL;
static SDL_Texture	*sdl_tex = NULL;
int		sdl_w = SCREEN_RES_X, sdl_h = SCREEN_RES_Y;
int		sdl_fs, sdl_flags = -1;
static int		cur_w, cur_h;
static int		cur_wx = 0, cur_wy = 0, cur_ww =0, cur_wh = 0;
static volatile int	sdl_enabled = 1;
static SDL_mutex*	sdl_mutex = NULL;
int mouse_capture;
int title_set = 0;
int resize_pending = 0;
int resize_w = 0;
int resize_h = 0;
float menubarheight = 0.0f;
static uint8_t interpixels[17842176];

const uint16_t sdl_to_xt[0x200] =
{
    [SDL_SCANCODE_ESCAPE] = 0x01,
    [SDL_SCANCODE_1] = 0x02,
    [SDL_SCANCODE_2] = 0x03,
    [SDL_SCANCODE_3] = 0x04,
    [SDL_SCANCODE_4] = 0x05,
    [SDL_SCANCODE_5] = 0x06,
    [SDL_SCANCODE_6] = 0x07,
    [SDL_SCANCODE_7] = 0x08,
    [SDL_SCANCODE_8] = 0x09,
    [SDL_SCANCODE_9] = 0x0A,
    [SDL_SCANCODE_0] = 0x0B,
    [SDL_SCANCODE_MINUS] = 0x0C,
    [SDL_SCANCODE_EQUALS] = 0x0D,
    [SDL_SCANCODE_BACKSPACE] = 0x0E,
    [SDL_SCANCODE_TAB] = 0x0F,
    [SDL_SCANCODE_Q] = 0x10,
    [SDL_SCANCODE_W] = 0x11,
    [SDL_SCANCODE_E] = 0x12,
    [SDL_SCANCODE_R] = 0x13,
    [SDL_SCANCODE_T] = 0x14,
    [SDL_SCANCODE_Y] = 0x15,
    [SDL_SCANCODE_U] = 0x16,
    [SDL_SCANCODE_I] = 0x17,
    [SDL_SCANCODE_O] = 0x18,
    [SDL_SCANCODE_P] = 0x19,
    [SDL_SCANCODE_LEFTBRACKET] = 0x1A,
    [SDL_SCANCODE_RIGHTBRACKET] = 0x1B,
    [SDL_SCANCODE_RETURN] = 0x1C,
    [SDL_SCANCODE_LCTRL] = 0x1D,
    [SDL_SCANCODE_A] = 0x1E,
    [SDL_SCANCODE_S] = 0x1F,
    [SDL_SCANCODE_D] = 0x20,
    [SDL_SCANCODE_F] = 0x21,
    [SDL_SCANCODE_G] = 0x22,
    [SDL_SCANCODE_H] = 0x23,
    [SDL_SCANCODE_J] = 0x24,
    [SDL_SCANCODE_K] = 0x25,
    [SDL_SCANCODE_L] = 0x26,
    [SDL_SCANCODE_SEMICOLON] = 0x27,
    [SDL_SCANCODE_APOSTROPHE] = 0x28,
    [SDL_SCANCODE_GRAVE] = 0x29,
    [SDL_SCANCODE_LSHIFT] = 0x2A,
    [SDL_SCANCODE_BACKSLASH] = 0x2B,
    [SDL_SCANCODE_Z] = 0x2C,
    [SDL_SCANCODE_X] = 0x2D,
    [SDL_SCANCODE_C] = 0x2E,
    [SDL_SCANCODE_V] = 0x2F,
    [SDL_SCANCODE_B] = 0x30,
    [SDL_SCANCODE_N] = 0x31,
    [SDL_SCANCODE_M] = 0x32,
    [SDL_SCANCODE_COMMA] = 0x33,
    [SDL_SCANCODE_PERIOD] = 0x34,
    [SDL_SCANCODE_SLASH] = 0x35,
    [SDL_SCANCODE_RSHIFT] = 0x36,
    [SDL_SCANCODE_KP_MULTIPLY] = 0x37,
    [SDL_SCANCODE_LALT] = 0x38,
    [SDL_SCANCODE_SPACE] = 0x39,
    [SDL_SCANCODE_CAPSLOCK] = 0x3A,
    [SDL_SCANCODE_F1] = 0x3B,
    [SDL_SCANCODE_F2] = 0x3C,
    [SDL_SCANCODE_F3] = 0x3D,
    [SDL_SCANCODE_F4] = 0x3E,
    [SDL_SCANCODE_F5] = 0x3F,
    [SDL_SCANCODE_F6] = 0x40,
    [SDL_SCANCODE_F7] = 0x41,
    [SDL_SCANCODE_F8] = 0x42,
    [SDL_SCANCODE_F9] = 0x43,
    [SDL_SCANCODE_F10] = 0x44,
    [SDL_SCANCODE_NUMLOCKCLEAR] = 0x45,
    [SDL_SCANCODE_SCROLLLOCK] = 0x46,
    [SDL_SCANCODE_HOME] = 0x147,
    [SDL_SCANCODE_UP] = 0x148,
    [SDL_SCANCODE_PAGEUP] = 0x149,
    [SDL_SCANCODE_KP_MINUS] = 0x4A,
    [SDL_SCANCODE_LEFT] = 0x14B,
    [SDL_SCANCODE_KP_5] = 0x4C,
    [SDL_SCANCODE_RIGHT] = 0x14D,
    [SDL_SCANCODE_KP_PLUS] = 0x4E,
    [SDL_SCANCODE_END] = 0x14F,
    [SDL_SCANCODE_DOWN] = 0x150,
    [SDL_SCANCODE_PAGEDOWN] = 0x151,
    [SDL_SCANCODE_INSERT] = 0x152,
    [SDL_SCANCODE_DELETE] = 0x153,
    [SDL_SCANCODE_F11] = 0x57,
    [SDL_SCANCODE_F12] = 0x58,

    [SDL_SCANCODE_KP_ENTER] = 0x11c,
    [SDL_SCANCODE_RCTRL] = 0x11d,
    [SDL_SCANCODE_KP_DIVIDE] = 0x135,
    [SDL_SCANCODE_RALT] = 0x138,
    [SDL_SCANCODE_KP_9] = 0x49,
    [SDL_SCANCODE_KP_8] = 0x48,
    [SDL_SCANCODE_KP_7] = 0x47,
    [SDL_SCANCODE_KP_6] = 0x4D,
    [SDL_SCANCODE_KP_4] = 0x4B,
    [SDL_SCANCODE_KP_3] = 0x51,
    [SDL_SCANCODE_KP_2] = 0x50,
    [SDL_SCANCODE_KP_1] = 0x4F,
    [SDL_SCANCODE_KP_0] = 0x52,
    [SDL_SCANCODE_KP_PERIOD] = 0x53,

    [SDL_SCANCODE_LGUI] = 0x15B,
    [SDL_SCANCODE_RGUI] = 0x15C,
    [SDL_SCANCODE_APPLICATION] = 0x15D,
    [SDL_SCANCODE_PRINTSCREEN] = 0x137
};

extern void RenderImGui();
static void
sdl_integer_scale(double *d, double *g)
{
    double ratio;

    if (*d > *g) {
	ratio = floor(*d / *g);
	*d = *g * ratio;
    } else {
	ratio = ceil(*d / *g);
	*d = *g / ratio;
    }
}

void sdl_reinit_texture();

static void
sdl_stretch(int *w, int *h, int *x, int *y)
{
    double hw, gw, hh, gh, dx, dy, dw, dh, gsr, hsr;
    int real_sdl_w, real_sdl_h;

    SDL_GL_GetDrawableSize(sdl_win, &real_sdl_w, &real_sdl_h);
    real_sdl_h -= menubarheight;
    hw = (double) real_sdl_w;
    hh = (double) real_sdl_h;
    gw = (double) *w;
    gh = (double) *h;
    hsr = hw / hh;

    switch (video_fullscreen_scale) {
	case FULLSCR_SCALE_FULL:
	default:
		*w = real_sdl_w;
		*h = real_sdl_h;
		*x = 0;
		*y = 0;
		break;
	case FULLSCR_SCALE_43:
	case FULLSCR_SCALE_KEEPRATIO:
		if (video_fullscreen_scale == FULLSCR_SCALE_43)
			gsr = 4.0 / 3.0;
		else
			gsr = gw / gh;
		if (gsr <= hsr) {
			dw = hh * gsr;
			dh = hh;
		} else {
			dw = hw;
			dh = hw / gsr;
		}
		dx = (hw - dw) / 2.0;
		dy = (hh - dh) / 2.0;
		*w = (int) dw;
		*h = (int) dh;
		*x = (int) dx;
		*y = (int) dy;
		break;
	case FULLSCR_SCALE_INT:
		gsr = gw / gh;
		if (gsr <= hsr) {
			dw = hh * gsr;
			dh = hh;
		} else {
			dw = hw;
			dh = hw / gsr;
		}
		sdl_integer_scale(&dw, &gw);
		sdl_integer_scale(&dh, &gh);
		dx = (hw - dw) / 2.0;
		dy = (hh - dh) / 2.0;
		*w = (int) dw;
		*h = (int) dh;
		*x = (int) dx;
		*y = (int) dy;
		break;
    }
}


void
sdl_blit_shim(int x, int y, int w, int h)
{
    params.x = x;
    params.y = y;
    params.w = w;
    params.h = h;
    if (!(!sdl_enabled || (x < 0) || (y < 0) || (w <= 0) || (h <= 0) || (w > 2048) || (h > 2048) || (buffer32 == NULL) || (sdl_render == NULL) || (sdl_tex == NULL)))
	video_copy(interpixels, &(buffer32->line[y][x]), h * (2048 + 64) * sizeof(uint32_t));
    if (screenshots)
	video_screenshot(interpixels, 0, 0, (2048 + 64));
    blitreq = 1;
    video_blit_complete();
}

void ui_window_title_real();

void
sdl_real_blit(SDL_Rect* r_src)
{
    SDL_Rect r_dst;
    int ret, winx, winy;
    SDL_GL_GetDrawableSize(sdl_win, &winx, &winy);
    winy -= menubarheight;
    SDL_RenderClear(sdl_render);

    r_dst = *r_src;
    r_dst.x = r_dst.y = 0;
    
    if (sdl_fs)
    {
		sdl_stretch(&r_dst.w, &r_dst.h, &r_dst.x, &r_dst.y);
    }
    else
    {
        r_dst.w *= ((float)winx / (float) r_dst.w);
        r_dst.h *= ((float)winy / (float) r_dst.h);
    }
    r_dst.y += menubarheight;
    if (!hide_status_bar) r_dst.h -= (menubarheight * 2);

    ret = SDL_RenderCopy(sdl_render, sdl_tex, r_src, &r_dst);
    if (ret)
	fprintf(stderr, "SDL: unable to copy texture to renderer (%s)\n", SDL_GetError());
    RenderImGui();

    SDL_RenderPresent(sdl_render);
}

void
sdl_blit(int x, int y, int w, int h)
{
    SDL_Rect r_src;

    if (!sdl_enabled || (x < 0) || (y < 0) || (w <= 0) || (h <= 0) || (w > 2048) || (h > 2048) || (buffer32 == NULL) || (sdl_render == NULL) || (sdl_tex == NULL)) {
    r_src.x = x;
    r_src.y = y;
    r_src.w = w;
    r_src.h = h;
    sdl_real_blit(&r_src);
    blitreq = 0;
	return;
    }

    SDL_LockMutex(sdl_mutex);

    if (resize_pending)
    {
        if (!video_fullscreen) sdl_resize(resize_w, resize_h + (hide_status_bar ? 0 : menubarheight * 2) );
        resize_pending = 0;
    }
    r_src.x = x;
    r_src.y = y;
    r_src.w = w;
    r_src.h = h;
    SDL_UpdateTexture(sdl_tex, &r_src, interpixels, (2048 + 64) * 4);
    blitreq = 0;

    sdl_real_blit(&r_src);
    SDL_UnlockMutex(sdl_mutex);
}

static void
sdl_destroy_window(void)
{
    if (sdl_win != NULL) {
    if (window_remember || (vid_resize & 2))
    {
        if (!(vid_resize & 2)) SDL_GetWindowSize(sdl_win, &window_w, &window_h);
        if (strncasecmp(SDL_GetCurrentVideoDriver(), "wayland", 7) != 0)
        {
            SDL_GetWindowPosition(sdl_win, &window_x, &window_y);
        }
    }
	SDL_DestroyWindow(sdl_win);
	sdl_win = NULL;
    }
}


static void
sdl_destroy_texture(void)
{
    /* SDL_DestroyRenderer also automatically destroys all associated textures. */
    if (sdl_render != NULL) {
	SDL_DestroyRenderer(sdl_render);
	sdl_render = NULL;
    }
}

void
sdl_close(void)
{
    if (sdl_mutex != NULL)
	SDL_LockMutex(sdl_mutex);

    /* Unregister our renderer! */
    video_setblit(NULL);

    if (sdl_enabled)
	sdl_enabled = 0;

    if (sdl_mutex != NULL) {
	SDL_DestroyMutex(sdl_mutex);
	sdl_mutex = NULL;
    }

    sdl_destroy_texture();
    sdl_destroy_window();

    /* Quit. */
    SDL_Quit();
    sdl_flags = -1;
}

static int old_capture = 0;

void
sdl_enable(int enable)
{
    if (sdl_flags == -1)
	return;

    SDL_LockMutex(sdl_mutex);
    sdl_enabled = !!enable;

    if (enable == 1) {
	SDL_SetWindowSize(sdl_win, cur_ww, cur_wh);
	sdl_reinit_texture();
    }

    SDL_UnlockMutex(sdl_mutex);
}

static void
sdl_select_best_hw_driver(void)
{
    int i;
    SDL_RendererInfo renderInfo;

    for (i = 0; i < SDL_GetNumRenderDrivers(); ++i)
    {
	SDL_GetRenderDriverInfo(i, &renderInfo);
	if (renderInfo.flags & SDL_RENDERER_ACCELERATED) {
		SDL_SetHint(SDL_HINT_RENDER_DRIVER, renderInfo.name);
		return;
	}
    }
}

extern void HandleSizeChange();

void
sdl_reinit_texture()
{
    SDL_RendererInfo info;
    memset(&info, 0, sizeof(SDL_RendererInfo));
    sdl_destroy_texture();

    if (sdl_flags & RENDERER_HARDWARE) {
	sdl_render = SDL_CreateRenderer(sdl_win, -1, SDL_RENDERER_ACCELERATED);
	SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, video_filter_method ? "1" : "0");
    } else
	sdl_render = SDL_CreateRenderer(sdl_win, -1, SDL_RENDERER_SOFTWARE);

    SDL_GetRendererInfo(sdl_render, &info);

    sdl_tex = SDL_CreateTexture(sdl_render, SDL_PIXELFORMAT_ARGB8888,
				SDL_TEXTUREACCESS_STREAMING, 2048, 2048);
    
    HandleSizeChange();
}

void
sdl_set_fs(int fs)
{
    SDL_LockMutex(sdl_mutex);
    SDL_SetWindowFullscreen(sdl_win, fs ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
    SDL_SetRelativeMouseMode((SDL_bool)mouse_capture);

    sdl_fs = fs;

    if (fs)
	sdl_flags |= RENDERER_FULL_SCREEN;
    else
	sdl_flags &= ~RENDERER_FULL_SCREEN;

    sdl_reinit_texture();
    SDL_UnlockMutex(sdl_mutex);
    device_force_redraw();
}

void
sdl_resize(int x, int y)
{
    int ww = 0, wh = 0, wx = 0, wy = 0;
	static int cur_dpi_scale = 0;
	float windowscale = 1.f;

    if (video_fullscreen & 2)
	return;

    if ((x == cur_w) && (y == cur_h) && cur_dpi_scale == dpi_scale)
	return;

    SDL_LockMutex(sdl_mutex);

    ww = x;
    wh = y;

    cur_w = x;
    cur_h = y;

    cur_wx = wx;
    cur_wy = wy;
    cur_ww = ww;
    cur_wh = wh;
	cur_dpi_scale = dpi_scale;
	if (dpi_scale)
	{
		float ddpi = 96.f;
		if (SDL_GetDisplayDPI(SDL_GetWindowDisplayIndex(sdl_win), &ddpi, NULL, NULL) != -1)
		{
			windowscale = ddpi / 96.f;
		}
	}

    SDL_SetWindowSize(sdl_win, cur_ww * windowscale, cur_wh * windowscale + menubarheight);
    SDL_GL_GetDrawableSize(sdl_win, &sdl_w, &sdl_h);

    sdl_reinit_texture();

    SDL_UnlockMutex(sdl_mutex);
}
void
sdl_reload(void)
{
	if (sdl_flags & RENDERER_HARDWARE)
	{
		SDL_LockMutex(sdl_mutex);

		SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, video_filter_method ? "1" : "0");
		sdl_reinit_texture();

		SDL_UnlockMutex(sdl_mutex);
	}
}

int
plat_vidapi(char* api)
{
    if (strncasecmp(api, "sdl_software", sizeof("sdl_software") - 1) == 0) return 0;
    if (strncasecmp(api, "default", sizeof("default") - 1) == 0) return 1;
    if (strncasecmp(api, "sdl_opengl", sizeof("sdl_opengl") - 1) == 0) return 2;
    return 0;
}

void sdl_determine_renderer(int flags)
{
    if (flags & RENDERER_HARDWARE)
    {
        if (flags & RENDERER_OPENGL)
        {
            SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengl");
        }
        else sdl_select_best_hw_driver();
    }
    else SDL_SetHint(SDL_HINT_RENDER_DRIVER, "software");
}

static int
sdl_init_common(int flags)
{
    wchar_t temp[128];
    SDL_version ver;

    /* Get and log the version of the DLL we are using. */
    SDL_GetVersion(&ver);
    fprintf(stderr, "SDL: version %d.%d.%d\n", ver.major, ver.minor, ver.patch);

    /* Initialize the SDL system. */
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
	fprintf(stderr, "SDL: initialization failed (%s)\n", SDL_GetError());
	return(0);
    }

    sdl_determine_renderer(flags);

    sdl_mutex = SDL_CreateMutex();
    sdl_win = SDL_CreateWindow("86Box", strncasecmp(SDL_GetCurrentVideoDriver(), "wayland", 7) != 0 && window_remember ? window_x : SDL_WINDOWPOS_CENTERED, strncasecmp(SDL_GetCurrentVideoDriver(), "wayland", 7) != 0 && window_remember ? window_y : SDL_WINDOWPOS_CENTERED, vid_resize & 2 ? fixed_size_x : scrnsz_x, vid_resize & 2 ? fixed_size_y : scrnsz_y, SDL_WINDOW_OPENGL | (vid_resize & 1 ? SDL_WINDOW_RESIZABLE : 0));
    if (!sdl_win)
    {
        return -1;
    }
    sdl_set_fs(video_fullscreen);
    if (!(video_fullscreen & 1))
    {
        if (vid_resize & 2)
	        SDL_SetWindowSize(sdl_win, fixed_size_x, fixed_size_y);
        else
	        SDL_SetWindowSize(sdl_win, scrnsz_x, scrnsz_y);
    }
    if ((vid_resize < 2) && window_remember)
    {
        SDL_SetWindowSize(sdl_win, window_w, window_h);
    }

    /* Make sure we get a clean exit. */
    atexit(sdl_close);

    /* Register our renderer! */
    video_setblit(sdl_blit_shim);

    sdl_enabled = 1;
    sdl_flags = flags;

    return(1);
}

int
sdl_inits()
{
    return sdl_init_common(0);
}


int
sdl_inith()
{
    return sdl_init_common(RENDERER_HARDWARE);
}


int
sdl_initho()
{
    return sdl_init_common(RENDERER_HARDWARE | RENDERER_OPENGL);
}


int
sdl_pause(void)
{
    return(0);
}

void
plat_mouse_capture(int on)
{
    SDL_LockMutex(sdl_mutex);
    SDL_SetRelativeMouseMode((SDL_bool)on);
    mouse_capture = on;
    SDL_UnlockMutex(sdl_mutex);
}

#if 0
void plat_resize(int w, int h)
{
    if (vid_resize)
    {
        return;
    }
    SDL_LockMutex(sdl_mutex);
    resize_w = w;
    resize_h = h;
    resize_pending = 1;
    SDL_UnlockMutex(sdl_mutex);
}
#endif

//wchar_t sdl_win_title[512] = { L'8', L'6', L'B', L'o', L'x', 0 };
SDL_mutex* titlemtx = NULL;

void ui_window_title_real()
{
}
extern SDL_threadID eventthread;

#if 0
/* Only activate threading path on macOS, otherwise it will softlock Xorg.
   Wayland doesn't seem to have this issue. */
wchar_t* ui_window_title(wchar_t* str)
{
    if (!str) return sdl_win_title;
#ifdef __APPLE__
    if (eventthread == SDL_ThreadID())
#endif
    {
        memset(sdl_win_title, 0, sizeof(sdl_win_title));
        wcsncpy(sdl_win_title, str, 512);
        ui_window_title_real();
        return str;
    }
#ifdef __APPLE__
    memset(sdl_win_title, 0, sizeof(sdl_win_title));
    wcsncpy(sdl_win_title, str, 512);
    title_set = 1;
#endif
    return str;
}
#endif
void
plat_pause(int p)
{
    static wchar_t oldtitle[512];
    wchar_t title[512];

    dopause = p;
    if (p) {
	wcsncpy(oldtitle, ui_window_title(NULL), sizeof_w(oldtitle) - 1);
	wcscpy(title, oldtitle);
	wcscat(title, L" - PAUSED -");
	ui_window_title(title);
    } else {
	ui_window_title(oldtitle);
    }
}