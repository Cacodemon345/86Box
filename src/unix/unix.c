#ifdef __linux__
#define _FILE_OFFSET_BITS 64
#define _LARGEFILE64_SOURCE 1
#endif
#include <SDL.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <inttypes.h>
#include <dlfcn.h>
#include <wchar.h>

#include <86box/86box.h>
#include <86box/keyboard.h>
#include <86box/mouse.h>
#include <86box/config.h>
#include <86box/plat.h>
#include <86box/plat_dynld.h>
#include <86box/device.h>
#include <86box/gameport.h>
#include <86box/unix_sdl.h>
#include <86box/timer.h>
#include <86box/nvr.h>
#include <86box/ui.h>

extern bool     ImGui_ImplSDL2_Init(SDL_Window* window);
extern void     ImGui_ImplSDL2_Shutdown();
extern void     ImGui_ImplSDL2_NewFrame();
extern bool     ImGui_ImplSDL2_ProcessEvent(const SDL_Event* event);
static int	first_use = 1;
static uint64_t	StartingTime;
static uint64_t Frequency;
int rctrl_is_lalt;
int	update_icons = 1;
int	kbd_req_capture;
int hide_status_bar;
int fixed_size_x = 640;
int fixed_size_y = 480;
extern int title_set;
extern wchar_t sdl_win_title[512];
plat_joystick_t	plat_joystick_state[MAX_PLAT_JOYSTICKS];
joystick_t	joystick_state[MAX_JOYSTICKS];
int		joysticks_present;
SDL_mutex *blitmtx;
SDL_threadID eventthread;
static int exit_event = 0;
int fullscreen_pending = 0;
extern float menubarheight;

static const uint16_t sdl_to_xt[0x200] =
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

typedef struct sdl_blit_params
{
    int x, y, w, h;
} sdl_blit_params;

sdl_blit_params params = { 0, 0, 0, 0 };
int blitreq = 0;
int limitedblitreq = 0;

void* dynld_module(const char *name, dllimp_t *table)
{
    dllimp_t* imp;
    void* modhandle = dlopen(name, RTLD_LAZY | RTLD_GLOBAL);
    if (modhandle)
    {
        for (imp = table; imp->name != NULL; imp++)
        {
            if ((*(void**)imp->func = dlsym(modhandle, imp->name)) == NULL)
            {
                dlclose(modhandle);
                return NULL;
            }
        }
    }
    return modhandle;
}

void
plat_tempfile(char *bufp, char *prefix, char *suffix)
{
    struct tm* calendertime;
    struct timeval t;
    time_t curtime;

    if (prefix != NULL)
	sprintf(bufp, "%s-", prefix);
      else
	strcpy(bufp, "");
    gettimeofday(&t, NULL);
    curtime = time(NULL);
    calendertime = localtime(&curtime);
    sprintf(&bufp[strlen(bufp)], "%d%02d%02d-%02d%02d%02d-%03ld%s", calendertime->tm_year, calendertime->tm_mon, calendertime->tm_mday, calendertime->tm_hour, calendertime->tm_min, calendertime->tm_sec, t.tv_usec / 1000l, suffix);
}

int
plat_getcwd(char *bufp, int max)
{
    return getcwd(bufp, max) != 0;
}

int
plat_chdir(char* str)
{
    return chdir(str);
}

void dynld_close(void *handle)
{
	dlclose(handle);
}

wchar_t* plat_get_string(int i)
{
    switch (i)
    {
        case IDS_2077:
            return L"Click to capture mouse.";
        case IDS_2078:
            return L"Press CTRL-END to release mouse";
        case IDS_2079:
            return L"Press CTRL-END or middle button to release mouse";
        case IDS_2080:
            return L"Failed to initialize FluidSynth";
        case IDS_4099:
            return L"MFM/RLL or ESDI CD-ROM drives never existed";
        case IDS_2093:
            return L"Failed to set up PCap";
        case IDS_2094:
            return L"No PCap devices found";
        case IDS_2110:
            return L"Unable to initialize FreeType";
        case IDS_2111:
            return L"Unable to initialize SDL, libsdl2 is required";
        case IDS_2131:
            return L"libfreetype is required for ESC/P printer emulation.";
        case IDS_2132:
            return L"libgs is required for automatic conversion of PostScript files to PDF.\n\nAny documents sent to the generic PostScript printer will be saved as PostScript (.ps) files.";
        case IDS_2129:
            return L"Make sure libpcap is installed and that you are on a libpcap-compatible network connection.";
        case IDS_2114:
            return L"Unable to initialize Ghostscript";
        case IDS_2063:
            return L"Machine \"%hs\" is not available due to missing ROMs in the roms/machines directory. Switching to an available machine.";
        case IDS_2064:
            return L"Video card \"%hs\" is not available due to missing ROMs in the roms/video directory. Switching to an available video card.";
        case IDS_2128:
            return L"Hardware not available";
        case IDS_2142:
            return L"Monitor in sleep mode";
    }
    return L"";
}

FILE *
plat_fopen(const char *path, const char *mode)
{
    return fopen(path, mode);
}

FILE *
plat_fopen64(const char *path, const char *mode)
{
    return fopen(path, mode);
}

int
plat_path_abs(char *path)
{
    return path[0] == '/';
}

void
plat_path_slash(char *path)
{
    if ((path[strlen(path)-1] != '/')) {
	strcat(path, "/");
    }
}

void
plat_put_backslash(char *s)
{
    int c = strlen(s) - 1;

    if (s[c] != '/')
	   s[c] = '/';
}

/* Return the last element of a pathname. */
char *
plat_get_basename(const char *path)
{
    int c = (int)strlen(path);

    while (c > 0) {
	if (path[c] == '/')
	   return((char *)&path[c + 1]);
       c--;
    }

    return((char *)path);
}
char *
plat_get_filename(char *s)
{
    int c = strlen(s) - 1;

    while (c > 0) {
	if (s[c] == '/' || s[c] == '\\')
	   return(&s[c+1]);
       c--;
    }

    return(s);
}


char *
plat_get_extension(char *s)
{
    int c = strlen(s) - 1;

    if (c <= 0)
	return(s);

    while (c && s[c] != '.')
		c--;

    if (!c)
	return(&s[strlen(s)]);

    return(&s[c+1]);
}


void
plat_append_filename(char *dest, const char *s1, const char *s2)
{
    strcpy(dest, s1);
    plat_path_slash(dest);
    strcat(dest, s2);
}

int
plat_dir_check(char *path)
{
    struct stat dummy;
    if (stat(path, &dummy) < 0)
    {
        return 0;
    }
    return S_ISDIR(dummy.st_mode);
}

int
plat_dir_create(char *path)
{
    return mkdir(path, S_IRWXU);
}

void *
plat_mmap(size_t size, uint8_t executable)
{
#if defined __APPLE__ && defined MAP_JIT
    void *ret = mmap(0, size, PROT_READ | PROT_WRITE | (executable ? PROT_EXEC : 0), MAP_ANON | MAP_PRIVATE | (executable ? MAP_JIT : 0), 0, 0);
#else
    void *ret = mmap(0, size, PROT_READ | PROT_WRITE | (executable ? PROT_EXEC : 0), MAP_ANON | MAP_PRIVATE, 0, 0);
#endif
    return (ret < 0) ? NULL : ret;
}

void
plat_munmap(void *ptr, size_t size)
{
    munmap(ptr, size);
}

uint64_t
plat_timer_read(void)
{
    return SDL_GetPerformanceCounter();
}

static uint64_t
plat_get_ticks_common(void)
{
    uint64_t EndingTime, ElapsedMicroseconds;
    if (first_use) {
	Frequency = SDL_GetPerformanceFrequency();
	StartingTime = SDL_GetPerformanceCounter();
	first_use = 0;
    }
    EndingTime = SDL_GetPerformanceCounter();
    ElapsedMicroseconds = ((EndingTime - StartingTime) * 1000000) / Frequency;
    return ElapsedMicroseconds;
}

uint32_t
plat_get_ticks(void)
{
	return (uint32_t)(plat_get_ticks_common() / 1000);
}

uint32_t
plat_get_micro_ticks(void)
{
	return (uint32_t)plat_get_ticks_common();
}

void plat_remove(char* path)
{
    remove(path);
}



void
plat_delay_ms(uint32_t count)
{
    SDL_Delay(count);
}

void
ui_sb_update_tip(int arg)
{

}

void
plat_get_dirname(char *dest, const char *path)
{
    int c = (int)strlen(path);
    char *ptr;

    ptr = (char *)path;

    while (c > 0) {
	if (path[c] == '/' || path[c] == '\\') {
		ptr = (char *)&path[c];
		break;
	}
 	c--;
    }

    /* Copy to destination. */
    while (path < ptr)
	*dest++ = *path++;
    *dest = '\0';
}
volatile int cpu_thread_run = 1;

int stricmp(const char* s1, const char* s2)
{
    return strcasecmp(s1, s2);
}

int strnicmp(const char *s1, const char *s2, size_t n)
{
    return strncasecmp(s1, s2, n);
}

void
main_thread(void *param)
{
    uint32_t old_time, new_time;
    int drawits, frames;

    SDL_SetThreadPriority(SDL_THREAD_PRIORITY_HIGH);
    framecountx = 0;
    //title_update = 1;
    old_time = SDL_GetTicks();
    drawits = frames = 0;
    while (!is_quit && cpu_thread_run) {
	/* See if it is time to run a frame of code. */
	new_time = SDL_GetTicks();
	drawits += (new_time - old_time);
	old_time = new_time;
	if (drawits > 0 && !dopause) {
		/* Yes, so do one frame now. */
		drawits -= 10;
		if (drawits > 50)
			drawits = 0;

		/* Run a block of code. */
		pc_run();

		/* Every 200 frames we save the machine status. */
		if (++frames >= 200 && nvr_dosave) {
			nvr_save();
			nvr_dosave = 0;
			frames = 0;
		}
	} else	/* Just so we dont overload the host OS. */
		SDL_Delay(1);

	/* If needed, handle a screen resize. */
	if (doresize && !video_fullscreen && !is_quit) {
		if (vid_resize & 2)
			plat_resize(fixed_size_x, fixed_size_y);
		else
			plat_resize(scrnsz_x, scrnsz_y);
		doresize = 0;
	}
    }

    is_quit = 1;
}

thread_t* thMain = NULL;

void
do_start(void)
{
    /* We have not stopped yet. */
    is_quit = 0;

    /* Initialize the high-precision timer. */
    SDL_InitSubSystem(SDL_INIT_TIMER);
    timer_freq = SDL_GetPerformanceFrequency();

    /* Start the emulator, really. */
    thMain = thread_create(main_thread, NULL);
}

void
do_stop(void)
{
    if (SDL_ThreadID() != eventthread)
    {
        exit_event = 1;
        return;
    }
    if (blitreq)
    {
        blitreq = 0;
        extern void video_blit_complete();
        video_blit_complete();
    }

    while(SDL_TryLockMutex(blitmtx) == SDL_MUTEX_TIMEDOUT)
    {
        if (blitreq)
        {
            blitreq = 0;
            extern void video_blit_complete();
            video_blit_complete();
        }
    }
    startblit();

    is_quit = 1;
    sdl_close();

    pc_close(thMain);

    thMain = NULL;
}

int	ui_msgbox(int flags, void *message)
{
    return ui_msgbox_header(flags, NULL, message);
}

int	ui_msgbox_header(int flags, void *header, void* message)
{
    SDL_MessageBoxData msgdata;
    SDL_MessageBoxButtonData msgbtn;
    if (!header) header = (flags & MBX_ANSI) ? "86Box" : L"86Box";
    if (header <= (void*)7168) header = plat_get_string(header);
    if (message <= (void*)7168) message = plat_get_string(message);
    msgbtn.buttonid = 1;
    msgbtn.text = "OK";
    msgbtn.flags = 0;
    memset(&msgdata, 0, sizeof(SDL_MessageBoxData));
    msgdata.numbuttons = 1;
    msgdata.buttons = &msgbtn;
    int msgflags = 0;
    if (msgflags & MBX_FATAL) msgflags |= SDL_MESSAGEBOX_ERROR;
    else if (msgflags & MBX_ERROR || msgflags & MBX_WARNING) msgflags |= SDL_MESSAGEBOX_WARNING;
    else msgflags |= SDL_MESSAGEBOX_INFORMATION;
    msgdata.flags = msgflags;
    if (flags & MBX_ANSI)
    {
        int button = 0;
        msgdata.title = header;
        msgdata.message = message;
        SDL_ShowMessageBox(&msgdata, &button);
        return button;
    }
    else
    {
        int button = 0;
        char *res = SDL_iconv_string("UTF-8", sizeof(wchar_t) == 2 ? "UTF-16LE" : "UTF-32LE", (char *)message, wcslen(message) * sizeof(wchar_t) + sizeof(wchar_t));
        char *res2 = SDL_iconv_string("UTF-8", sizeof(wchar_t) == 2 ? "UTF-16LE" : "UTF-32LE", (char *)header, wcslen(header) * sizeof(wchar_t) + sizeof(wchar_t));
        msgdata.message = res;
        msgdata.title = res2;
        SDL_ShowMessageBox(&msgdata, &button);
        free(res);
        free(res2);
        return button;
    }

    return 0;
}

void plat_get_exe_name(char *s, int size)
{
    char* basepath = SDL_GetBasePath();
    snprintf(s, size, "%s%s", basepath, basepath[strlen(basepath) - 1] == '/' ? "86box" : "/86box");
}

void
plat_power_off(void)
{
    confirm_exit = 0;
    nvr_save();
    config_save();

    /* Deduct a sufficiently large number of cycles that no instructions will
       run before the main thread is terminated */
    cycles -= 99999999;

    cpu_thread_run = 0;
}

extern void     sdl_blit(int x, int y, int w, int h);

typedef struct mouseinputdata
{
    int deltax, deltay, deltaz;
    int mousebuttons;
} mouseinputdata;
SDL_mutex* mousemutex;
static mouseinputdata mousedata;
void mouse_poll()
{
    SDL_LockMutex(mousemutex);
    mouse_x = mousedata.deltax;
    mouse_y = mousedata.deltay;
    mouse_z = mousedata.deltaz;
    mousedata.deltax = mousedata.deltay = mousedata.deltaz = 0;
    mouse_buttons = mousedata.mousebuttons;
    SDL_UnlockMutex(mousemutex);
}

void ui_sb_set_ready(int ready) {}
char* xargv[512];

// From musl.
char *local_strsep(char **str, const char *sep)
{
	char *s = *str, *end;
	if (!s) return NULL;
	end = s + strcspn(s, sep);
	if (*end) *end++ = 0;
	else end = 0;
	*str = end;
	return s;
}

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

bool process_media_commands_3(uint8_t* id, char* fn, uint8_t* wp, int cmdargc)
{
    bool err = false;
    *id = atoi(xargv[1]);
    if (xargv[2][0] == '\'' || xargv[2][0] == '"')
    {
        int curarg = 2;
        for (curarg = 2; curarg < cmdargc; curarg++)
        {
            if (strlen(fn) + strlen(xargv[curarg]) >= PATH_MAX)
            {
                err = true;
                fprintf(stderr, "Path name too long.\n");
            }
            strcat(fn, xargv[curarg] + (xargv[curarg][0] == '\'' || xargv[curarg][0] == '"'));
            if (fn[strlen(fn) - 1] == '\''
                || fn[strlen(fn) - 1] == '"')
            {
                if (curarg + 1 < cmdargc)
                {
                    *wp = atoi(xargv[curarg + 1]);
                }
                break;
            }
            strcat(fn, " ");
        }
    }
    else
    {
        if (strlen(xargv[2]) < PATH_MAX)
        {
            strcpy(fn, xargv[2]);
            *wp = atoi(xargv[3]);
        }
        else
        {
            fprintf(stderr, "Path name too long.\n");
            err = true;
        }
    }
    if (fn[strlen(fn) - 1] == '\''
    || fn[strlen(fn) - 1] == '"') fn[strlen(fn) - 1] = '\0';
    return err;
}

uint32_t timer_onesec(uint32_t interval, void* param)
{
        pc_onesec();
        return interval;
}

extern void InitImGui();
extern bool ImGuiWantsMouseCapture();
extern bool ImGuiWantsKeyboardCapture();
extern bool IsFileDlgOpen();
extern void sdl_real_blit(SDL_Rect* r_src);

extern SDL_Window* sdl_win;
int main(int argc, char** argv)
{
    SDL_Event event;
    void* libedithandle;

    SDL_Init(0);
    pc_init(argc, argv);
    if (! pc_init_modules()) {
        ui_msgbox_header(MBX_FATAL, L"No ROMs found.", L"86Box could not find any usable ROM images.\n\nPlease download a ROM set and extract it into the \"roms\" directory.");
        SDL_Quit();
        return 6;
    }
    
    eventthread = SDL_ThreadID();
    blitmtx = SDL_CreateMutex();
    if (!blitmtx)
    {
        fprintf(stderr, "Failed to create blit mutex: %s", SDL_GetError());
        return -1;
    }

    mousemutex = SDL_CreateMutex();
    switch (vid_api)
    {
        case 0:
            sdl_inits();
            break;
        default:
        case 1:
            sdl_inith();
            break;
        case 2:
            sdl_initho();
            break;
    }

    if (start_in_fullscreen)
    {
        video_fullscreen = 1;
	    sdl_set_fs(1);
    }
    /* Fire up the machine. */
    pc_reset_hard_init();

    /* Set the PAUSE mode depending on the renderer. */
    //plat_pause(0);

    /* Initialize the rendering window, or fullscreen. */

    do_start();

    SDL_AddTimer(1000, timer_onesec, NULL);
    InitImGui();
    while (!is_quit)
    {
        static int mouse_inside = 0;
        while (SDL_PollEvent(&event))
	    {
            if (!mouse_capture) ImGui_ImplSDL2_ProcessEvent(&event);
            switch(event.type)
            {
                case SDL_QUIT:
                {
                    if (IsFileDlgOpen())
                    {
                        int curdopause = dopause;
                        plat_pause(1);
                        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_WARNING, "86Box",
                                    "A file dialog is open. Please close it before exiting 86Box.\n", NULL);
                        plat_pause(curdopause);
                        break;
                    }
                  
                	exit_event = 1;
                	break;
                }
                case SDL_MOUSEWHEEL:
                {
                    if (ImGuiWantsMouseCapture()) break;
                    if (mouse_capture)
                    {
                        if (event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED)
                        {
                            event.wheel.x *= -1;
                            event.wheel.y *= -1;
                        }
                        SDL_LockMutex(mousemutex);
                        mousedata.deltaz = event.wheel.y;
                        SDL_UnlockMutex(mousemutex);
                    }
                    break;
                }
                case SDL_MOUSEMOTION:
                {
                    if (ImGuiWantsMouseCapture()) break;
                    if (mouse_capture)
                    {
                        SDL_LockMutex(mousemutex);
                        mousedata.deltax += event.motion.xrel;
                        mousedata.deltay += event.motion.yrel;
                        SDL_UnlockMutex(mousemutex);
                    }
                    break;
                }
                case SDL_MOUSEBUTTONDOWN:
                case SDL_MOUSEBUTTONUP:
                {
                    if (ImGuiWantsMouseCapture()) break;
                    if ((event.button.button == SDL_BUTTON_LEFT)
                    && !(mouse_capture)
                    && event.button.state == SDL_RELEASED
                    && (mouse_inside || video_fullscreen))
                    {
                        plat_mouse_capture(1);
                        break;
                    }
                    if (mouse_get_buttons() < 3 && event.button.button == SDL_BUTTON_MIDDLE)
                    {
                        plat_mouse_capture(0);
                        break;
                    }
                    if (mouse_capture)
                    {
                        int buttonmask = 0;

                        switch(event.button.button)
                        {
                            case SDL_BUTTON_LEFT:
                                buttonmask = 1;
                                break;
                            case SDL_BUTTON_RIGHT:
                                buttonmask = 2;
                                break;
                            case SDL_BUTTON_MIDDLE:
                                buttonmask = 4;
                                break;
                        }
                        SDL_LockMutex(mousemutex);
                        if (event.button.state == SDL_PRESSED)
                        {
                            mousedata.mousebuttons |= buttonmask;
                        }
                        else mousedata.mousebuttons &= ~buttonmask;
                        SDL_UnlockMutex(mousemutex);
                    }
                    break;
                }
                case SDL_RENDER_DEVICE_RESET:
                case SDL_RENDER_TARGETS_RESET:
                    {    
                        extern void sdl_reinit_texture();
                        sdl_reinit_texture();
                        break;
                    }
                case SDL_KEYDOWN:
                case SDL_KEYUP:
                {
                    if (ImGuiWantsKeyboardCapture()) break;
                    if (kbd_req_capture && !mouse_capture) break;
                    uint16_t xtkey = 0;
                    switch(event.key.keysym.scancode)
                    {
                        default:
                            xtkey = sdl_to_xt[event.key.keysym.scancode];
                            break;
                    }
                    if ((xtkey == 0x11D) && rctrl_is_lalt)
			            xtkey = 0x038;
                    keyboard_input(event.key.state == SDL_PRESSED, xtkey);
                }
                case SDL_WINDOWEVENT:
                {
                    switch (event.window.event)
                    {
                    case SDL_WINDOWEVENT_SIZE_CHANGED:
                    {
                        window_w = event.window.data1;
                        window_h = event.window.data2;
                        if (window_remember) config_save();
                        break;
                    }
                    case SDL_WINDOWEVENT_MOVED:
                    {
                        if (strncasecmp(SDL_GetCurrentVideoDriver(), "wayland", 7) != 0)
                        {
                            window_x = event.window.data1;
                            window_y = event.window.data2;
                            if (window_remember) config_save();
                        }
                        break;
                    }
                    case SDL_WINDOWEVENT_ENTER:
                        mouse_inside = 1;
                        break;
                    case SDL_WINDOWEVENT_LEAVE:
                        mouse_inside = 0;
                        break;
                    }
                }
            }
	    }
        if (mouse_capture && keyboard_ismsexit())
        {
            plat_mouse_capture(0);
        }
        if (blitreq)
        {
            extern void sdl_blit(int x, int y, int w, int h);
            sdl_blit(params.x, params.y, params.w, params.h);
        }
        else
        {
            SDL_Rect srcrect;
            memcpy(&srcrect, &params, sizeof(SDL_Rect));
            sdl_real_blit(&srcrect);
        }
        if (title_set)
        {
            extern void ui_window_title_real();
            ui_window_title_real();
        }
        if (video_fullscreen && keyboard_isfsexit())
        {
            sdl_set_fs(0);
            video_fullscreen = 0;
        }
        if (!(video_fullscreen) && keyboard_isfsenter())
        {
            sdl_set_fs(1);
            video_fullscreen = 1;
        }
        if (fullscreen_pending)
        {
            sdl_set_fs(video_fullscreen);
            fullscreen_pending = 0;
        }
        if ((keyboard_recv(0x1D) || keyboard_recv(0x11D)) && keyboard_recv(0x58))
        {
            pc_send_cad();
        }
        if ((keyboard_recv(0x1D) || keyboard_recv(0x11D)) && keyboard_recv(0x57))
        {
            take_screenshot();
        }
        if (exit_event)
        {
            do_stop();
            break;
        }
    }
    printf("\n");
    SDL_DestroyMutex(blitmtx);
    SDL_DestroyMutex(mousemutex);
    ImGui_ImplSDL2_Shutdown();
    SDL_Quit();
    return 0;
}
char* plat_vidapi_name(int i)
{
    switch (i)
    {
        case 0:
            return "sdl_software";
        case 1:
        default:
            return "default";
        case 2:
            return "sdl_opengl";
    }
}

void
set_language(uint32_t id)
{
    lang_id = id;
}


/* Sets up the program language before initialization. */
uint32_t plat_language_code(char* langcode)
{
    /* or maybe not */ 
    return 0;
}

void joystick_init(void) {}
void joystick_close(void) {}
void joystick_process(void) {}
void startblit()
{
    SDL_LockMutex(blitmtx);
}

void endblit()
{
    SDL_UnlockMutex(blitmtx);
}
