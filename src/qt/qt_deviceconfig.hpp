#ifndef QT_DEVICECONFIG_HPP
#define QT_DEVICECONFIG_HPP

#include <QDialog>
#include <QWidget>

#include "qt_settings.hpp"

extern "C" {
struct _device_;
}

namespace Ui {
class DeviceConfig;
}

class Settings;

class DeviceConfig : public QDialog {
    Q_OBJECT

public:
    explicit DeviceConfig(QWidget *parent = nullptr);
    ~DeviceConfig() override;

    static void    ConfigureDevice(const _device_ *device, int instance = 0,
                                   QWidget *settings = nullptr, void* devicePriv = nullptr);
    static QString DeviceName(const _device_ *device, const char *internalName, int bus);

private:
    Ui::DeviceConfig *ui;
    void   ProcessConfig(void *dc, const void *c, bool is_dep, bool is_runtime);
};

#endif // QT_DEVICECONFIG_HPP
