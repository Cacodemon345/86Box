#include <cstdio>

#include <mutex>
#include <thread>
#include <memory>
#include <algorithm>
#include <map>

#include <QDebug>

#include <QDir>
#include <QFileInfo>
#include <QTemporaryFile>
#include <QCoreApplication>
#include <QDateTime>
#include <QApplication>

#include <QLibrary>
#include <QElapsedTimer>

#include "qt_mainwindow.hpp"
#include "qt_progsettings.hpp"

#ifdef Q_OS_UNIX
#include <sys/mman.h>
#endif

#ifdef Q_OS_ANDROID
#include <jni.h>
#endif

// static QByteArray buf;
extern QElapsedTimer elapsed_timer;
extern MainWindow* main_window;
QElapsedTimer elapsed_timer;

static std::atomic_int blitmx_contention = 0;
static std::mutex blitmx;

class CharPointer {
public:
    CharPointer(char* buf, int size) : b(buf), s(size) {}
    CharPointer& operator=(const QByteArray &ba) {
        if (s > 0) {
            strncpy(b, ba.data(), s-1);
            b[s] = 0;
        } else {
            // if we haven't been told the length of b, just assume enough
            // because we didn't get it from emulator code
            strcpy(b, ba.data());
            b[ba.size()] = 0;
        }
        return *this;
    }
private:
    char* b;
    int s;
};

extern "C" {
#ifdef Q_OS_WINDOWS
#define NOMINMAX
#include <windows.h>
#else
#include <strings.h>
#endif
#include <86box/86box.h>
#include <86box/device.h>
#include <86box/gameport.h>
#include <86box/timer.h>
#include <86box/nvr.h>
#include <86box/plat_dynld.h>
#include <86box/config.h>
#include <86box/ui.h>
#include <86box/discord.h>

#include "../cpu/cpu.h"
#include <86box/plat.h>

volatile int cpu_thread_run = 1;
int mouse_capture = 0;
int fixed_size_x = 640;
int fixed_size_y = 480;
int rctrl_is_lalt = 0;
int	update_icons = 0;
int	kbd_req_capture = 0;
int hide_status_bar = 0;
int hide_tool_bar = 0;
uint32_t lang_id = 0x0409, lang_sys = 0x0409; // Multilangual UI variables, for now all set to LCID of en-US

int stricmp(const char* s1, const char* s2)
{
#ifdef Q_OS_WINDOWS
    return _stricmp(s1, s2);
#else
    return strcasecmp(s1, s2);
#endif
}

int strnicmp(const char *s1, const char *s2, size_t n)
{
#ifdef Q_OS_WINDOWS
    return _strnicmp(s1, s2, n);
#else
    return strncasecmp(s1, s2, n);
#endif
}

void
do_stop(void)
{
    QCoreApplication::quit();
}

void plat_get_exe_name(char *s, int size)
{
    QByteArray exepath_temp = QCoreApplication::applicationDirPath().toLocal8Bit();

    memcpy(s, exepath_temp.data(), std::min((qsizetype)exepath_temp.size(),(qsizetype)size));

    plat_path_slash(s);
}

uint32_t
plat_get_ticks(void)
{
    return elapsed_timer.elapsed();
}

uint64_t
plat_timer_read(void)
{
    return elapsed_timer.elapsed();
}

FILE *
plat_fopen(const char *path, const char *mode)
{
    /*
    QString filepath(path);
    if (filepath.isEmpty()) {
        return nullptr;
    }

    qWarning() << "plat_fopen" << filepath;
    bool ok = false;
    QFile file(filepath);
    auto mode_len = strlen(mode);
    for (size_t i = 0; i < mode_len; ++i) {
        switch (mode[i]) {
        case 'r':
            ok = file.open(QIODevice::ReadOnly);
            break;
        case 'w':
            ok = file.open(QIODevice::ReadWrite);
            break;
        case 'b':
        case 't':
            break;
        default:
            qWarning() << "Unhandled open mode" << mode[i];
        }
    }

    if (ok) {
        qDebug() << "filehandle" << file.handle();
        QFile returned;
        qDebug() << "\t" << returned.open(file.handle(), file.openMode(), QFileDevice::FileHandleFlag::DontCloseHandle);
        return fdopen(returned.handle(), mode);
    } else {
        return nullptr;
    }
    */

/* Not sure if any this is necessary, fopen seems to work on Windows -Manaatti
#ifdef Q_OS_WINDOWS
    wchar_t *pathw, *modew;
    int len;
    FILE *fp;

    if (acp_utf8)
        return fopen(path, mode);
    else {
        len = mbstoc16s(NULL, path, 0) + 1;
        pathw = malloc(sizeof(wchar_t) * len);
        mbstoc16s(pathw, path, len);

        len = mbstoc16s(NULL, mode, 0) + 1;
        modew = malloc(sizeof(wchar_t) * len);
        mbstoc16s(modew, mode, len);

        fp = _wfopen(pathw, modew);

        free(pathw);
        free(modew);

        return fp;
    }
#endif
#ifdef Q_OS_UNIX
*/
    return fopen(path, mode);
//#endif
}

FILE *
plat_fopen64(const char *path, const char *mode)
{
    return fopen(path, mode);
}

int
plat_dir_create(char *path)
{
    return QDir().mkdir(path) ? 0 : -1;
}

int
plat_dir_check(char *path)
{
    QFileInfo fi(path);
    return fi.isDir() ? 1 : 0;
}

int
plat_getcwd(char *bufp, int max)
{
    CharPointer(bufp, max) = QDir::currentPath().toUtf8();
    return 0;
}

void
plat_get_dirname(char *dest, const char *path)
{
    QFileInfo fi(path);
    CharPointer(dest, -1) = fi.dir().path().toUtf8();
}

char *
plat_get_extension(char *s)
{
    auto len = strlen(s);
    auto idx = QByteArray::fromRawData(s, len).lastIndexOf('.');
    if (idx >= 0) {
        return s+idx+1;
    }
    return s+len;
}

char *
plat_get_filename(char *s)
{
#ifdef Q_OS_WINDOWS
    int c = strlen(s) - 1;

    while (c > 0) {
	if (s[c] == '/' || s[c] == '\\')
	   return(&s[c+1]);
       c--;
    }

    return(s);
#else
    auto idx = QByteArray::fromRawData(s, strlen(s)).lastIndexOf(QDir::separator().toLatin1());
    if (idx >= 0) {
        return s+idx+1;
    }
    return s;
#endif
}

int
plat_path_abs(char *path)
{
    QFileInfo fi(path);
    return fi.isAbsolute() ? 1 : 0;
}

void
plat_path_slash(char *path)
{
    auto len = strlen(path);
    auto separator = QDir::separator().toLatin1();
    if (path[len-1] != separator) {
        path[len] = separator;
        path[len+1] = 0;
    }
}

void
plat_append_filename(char *dest, const char *s1, const char *s2)
{
    strcpy(dest, s1);
    plat_path_slash(dest);
    strcat(dest, s2);
}

void
plat_tempfile(char *bufp, char *prefix, char *suffix)
{
    QString name;

    if (prefix != nullptr) {
        name.append(QString("%1-").arg(prefix));
    }

     name.append(QDateTime::currentDateTime().toString("yyyyMMdd-hhmmss-zzzz"));
     if (suffix) name.append(suffix);
     sprintf(&bufp[strlen(bufp)], "%s", name.toUtf8().data());
}

void plat_remove(char* path)
{
    QFile(path).remove();
}

void *
plat_mmap(size_t size, uint8_t executable)
{
#if defined Q_OS_WINDOWS
    return VirtualAlloc(NULL, size, MEM_COMMIT, executable ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE);
#elif defined Q_OS_UNIX
#if defined Q_OS_DARWIN && defined MAP_JIT
    void *ret = mmap(0, size, PROT_READ | PROT_WRITE | (executable ? PROT_EXEC : 0), MAP_ANON | MAP_PRIVATE | (executable ? MAP_JIT : 0), -1, 0);
#else
    void *ret = mmap(0, size, PROT_READ | PROT_WRITE | (executable ? PROT_EXEC : 0), MAP_ANON | MAP_PRIVATE, -1, 0);
#endif
    return (ret == MAP_FAILED) ? nullptr : ret;
#endif
}

void
plat_munmap(void *ptr, size_t size)
{
#if defined Q_OS_WINDOWS
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    munmap(ptr, size);
#endif
}

void
plat_pause(int p)
{
    static wchar_t oldtitle[512];
    wchar_t title[512];

    if ((p == 0) && (time_sync & TIME_SYNC_ENABLED))
        nvr_time_sync();

    dopause = p;
    if (p) {
        wcsncpy(oldtitle, ui_window_title(NULL), sizeof_w(oldtitle) - 1);
        wcscpy(title, oldtitle);
        wcscat(title, L" - PAUSED -");
        ui_window_title(title);
    } else {
        ui_window_title(oldtitle);
    }
    discord_update_activity(dopause);
}

// because we can't include nvr.h because it's got fields named new
extern int	nvr_save(void);

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
    main_window->close();
}

void set_language(uint32_t id) {
    lang_id = id;
}

extern "C++"
{
    QMap<uint32_t, QPair<QString, QString>> ProgSettings::lcid_langcode =
    {
        {0x0405, {"cs-CZ", "Czech (Czech Republic)"} },
        {0x0407, {"de-DE", "German (Germany)"} },
        {0x0408, {"en-US", "English (United States)"} },
        {0x0809, {"en-GB", "English (United Kingdom)"} },
        {0x0C0A, {"es-ES", "Spanish (Spain)"} },
        {0x040B, {"fi-FI", "Finnish (Finland)"} },
        {0x040C, {"fr-FR", "French (France)"} },
        {0x041A, {"hr-HR", "Croatian (Croatia)"} },
        {0x040E, {"hu-HU", "Hungarian (Hungary)"} },
        {0x0410, {"it-IT", "Italian (Italy)"} },
        {0x0411, {"ja-JP", "Japanese (Japan)"} },
        {0x0412, {"ko-KR", "Korean (Korea)"} },
        {0x0416, {"pt-BR", "Portuguese (Brazil)"} },
        {0x0816, {"pt-PT", "Portuguese (Portugal)"} },
        {0x0419, {"ru-RU", "Russian (Russia)"} },
        {0x0424, {"sl-SI", "Slovenian (Slovenia)"} },
        {0x041F, {"tr-TR", "Turkish (Turkey)"} },
        {0x0804, {"zh-CN", "Chinese (China)"} },
        {0xFFFF, {"system", "(System Default)"} },
    };
}

/* Sets up the program language before initialization. */
uint32_t plat_language_code(char* langcode) {
    for (auto& curKey : ProgSettings::lcid_langcode.keys())
    {
        if (ProgSettings::lcid_langcode[curKey].first == langcode)
        {
            return curKey;
        }
    }
    return 0xFFFF;
}

/* Converts back the language code to LCID */
void plat_language_code_r(uint32_t lcid, char* outbuf, int len) {
    if (!ProgSettings::lcid_langcode.contains(lcid))
    {
        qstrncpy(outbuf, "system", len);
        return;
    }
    qstrncpy(outbuf, ProgSettings::lcid_langcode[lcid].first.toUtf8().constData(), len);
    return;
}

void* dynld_module(const char *name, dllimp_t *table)
{
    QString libraryName = name;
    QFileInfo fi(libraryName);
    QStringList removeSuffixes = {"dll", "dylib", "so"};
    if (removeSuffixes.contains(fi.suffix())) {
        libraryName = fi.completeBaseName();
    }

    auto lib = std::unique_ptr<QLibrary>(new QLibrary(libraryName));
    if (lib->load()) {
        for (auto imp = table; imp->name != nullptr; imp++)
        {
            auto ptr = lib->resolve(imp->name);
            if (ptr == nullptr) {
                return nullptr;
            }
            auto imp_ptr = reinterpret_cast<void**>(imp->func);
            *imp_ptr = reinterpret_cast<void*>(ptr);
        }
    } else {
        return nullptr;
    }

    return lib.release();
}

void dynld_close(void *handle)
{
    delete reinterpret_cast<QLibrary*>(handle);
}

void startblit()
{
    blitmx_contention++;
    if (blitmx.try_lock()) {
        return;
    }

    blitmx.lock();
}

void endblit()
{
    blitmx_contention--;
    blitmx.unlock();
    if (blitmx_contention > 0) {
        // a deadlock has been observed on linux when toggling via video_toggle_option
        // because the mutex is typically unfair on linux
        // => sleep if there's contention
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

}

#ifdef Q_OS_WINDOWS
size_t mbstoc16s(uint16_t dst[], const char src[], int len)
{
    if (src == NULL) return 0;
    if (len < 0) return 0;

    size_t ret = MultiByteToWideChar(CP_UTF8, 0, src, -1, reinterpret_cast<LPWSTR>(dst), dst == NULL ? 0 : len);

    if (!ret) {
	return -1;
    }

    return ret;
}

size_t c16stombs(char dst[], const uint16_t src[], int len)
{
    if (src == NULL) return 0;
    if (len < 0) return 0;

    size_t ret = WideCharToMultiByte(CP_UTF8, 0, reinterpret_cast<LPCWCH>(src), -1, dst, dst == NULL ? 0 : len, NULL, NULL);

    if (!ret) {
	return -1;
    }

    return ret;
}
#endif

#ifdef _WIN32
#define LIB_NAME_FLUIDSYNTH "libfluidsynth.dll"
#define LIB_NAME_GS "gsdll32.dll"
#define LIB_NAME_FREETYPE "freetype.dll"
#define MOUSE_CAPTURE_KEYSEQ "F8+F12"
#else
#define LIB_NAME_FLUIDSYNTH "libfluidsynth"
#define LIB_NAME_GS "libgs"
#define LIB_NAME_FREETYPE "libfreetype"
#define MOUSE_CAPTURE_KEYSEQ "CTRL-END"
#endif
#ifdef Q_OS_MACOS
#define ROMDIR "~/Library/Application Support/net.86box.86box/roms"
#else
#define ROMDIR "roms"
#endif


QMap<int, std::wstring> ProgSettings::translatedstrings;

void ProgSettings::reloadStrings()
{
    translatedstrings.clear();
    translatedstrings[IDS_2077] = QCoreApplication::translate("", "Click to capture mouse").toStdWString();
    translatedstrings[IDS_2078] = QCoreApplication::translate("", "Press F8+F12 to release mouse").replace("F8+F12", MOUSE_CAPTURE_KEYSEQ).replace("CTRL-END", QLocale::system().name() == "de_DE" ? "Strg+Ende" : "CTRL-END").toStdWString();
    translatedstrings[IDS_2079] = QCoreApplication::translate("", "Press F8+F12 or middle button to release mouse").replace("F8+F12", MOUSE_CAPTURE_KEYSEQ).replace("CTRL-END", QLocale::system().name() == "de_DE" ? "Strg+Ende" : "CTRL-END").toStdWString();
    translatedstrings[IDS_2080] = QCoreApplication::translate("", "Failed to initialize FluidSynth").toStdWString();
    translatedstrings[IDS_4099] = QCoreApplication::translate("", "MFM/RLL or ESDI CD-ROM drives never existed").toStdWString();
    translatedstrings[IDS_2093] = QCoreApplication::translate("", "Failed to set up PCap").toStdWString();
    translatedstrings[IDS_2094] = QCoreApplication::translate("", "No PCap devices found").toStdWString();
    translatedstrings[IDS_2110] = QCoreApplication::translate("", "Unable to initialize FreeType").toStdWString();
    translatedstrings[IDS_2111] = QCoreApplication::translate("", "Unable to initialize SDL, libsdl2 is required").toStdWString();
    translatedstrings[IDS_2129] = QCoreApplication::translate("", "Make sure libpcap is installed and that you are on a libpcap-compatible network connection.").toStdWString();
    translatedstrings[IDS_2114] = QCoreApplication::translate("", "Unable to initialize Ghostscript").toStdWString();
    translatedstrings[IDS_2063] = QCoreApplication::translate("", "Machine \"%hs\" is not available due to missing ROMs in the roms/machines directory. Switching to an available machine.").toStdWString();
    translatedstrings[IDS_2064] = QCoreApplication::translate("", "Video card \"%hs\" is not available due to missing ROMs in the roms/video directory. Switching to an available video card.").toStdWString();
    translatedstrings[IDS_2128] = QCoreApplication::translate("", "Hardware not available").toStdWString();
    translatedstrings[IDS_2120] = QCoreApplication::translate("", "No ROMs found").toStdWString();
    translatedstrings[IDS_2056] = QCoreApplication::translate("", "86Box could not find any usable ROM images.\n\nPlease <a href=\"https://github.com/86Box/roms/releases/latest\">download</a> a ROM set and extract it into the \"roms\" directory.").replace("roms", ROMDIR).toStdWString();

    auto flsynthstr = QCoreApplication::translate("", " is required for FluidSynth MIDI output.");
    if (flsynthstr.contains("libfluidsynth"))
    {
        flsynthstr.replace("libfluidsynth", LIB_NAME_FLUIDSYNTH);
    }
    else flsynthstr.prepend(LIB_NAME_FLUIDSYNTH);
    translatedstrings[IDS_2133] = flsynthstr.toStdWString();
    auto gssynthstr = QCoreApplication::translate("", " is required for automatic conversion of PostScript files to PDF.\n\nAny documents sent to the generic PostScript printer will be saved as PostScript (.ps) files.");
    if (gssynthstr.contains("libgs"))
    {
        gssynthstr.replace("libgs", LIB_NAME_GS);
    }
    else gssynthstr.prepend(LIB_NAME_GS);
    translatedstrings[IDS_2132] = flsynthstr.toStdWString();
    auto ftsynthstr = QCoreApplication::translate("", " is required for ESC/P printer emulation.");
    if (ftsynthstr.contains("libfreetype"))
    {
        ftsynthstr.replace("libfreetype", LIB_NAME_FREETYPE);
    }
    else ftsynthstr.prepend(LIB_NAME_FREETYPE);
    translatedstrings[IDS_2131] = ftsynthstr.toStdWString();
}

wchar_t* plat_get_string(int i)
{
    if (ProgSettings::translatedstrings.empty()) ProgSettings::reloadStrings();
    return ProgSettings::translatedstrings[i].data();
}

int
plat_chdir(char *path)
{
    return QDir::setCurrent(QString(path)) ? 0 : -1;
}

#ifdef __ANDROID__
extern "C"
{
JNIEXPORT void JNICALL Java_src_android_src_net_eightsixbox_eightsixbox_EmuActivity_onKeyDownEvent(JNIEnv* env, jobject obj, jint keycode, jboolean down)
{
    QApplication::postEvent(main_window, new QKeyEvent(down ? QEvent::KeyPress : QEvent::KeyRelease, 0, QGuiApplication::keyboardModifiers(), keycode, 0, 0));
}
}
#endif
