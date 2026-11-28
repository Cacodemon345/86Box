#ifndef QT_SETTINGSDISPLAY_HPP
#define QT_SETTINGSDISPLAY_HPP

#include <QWidget>

extern "C" {
#include <86box/86box.h>
}

class QComboBox;
class QPushButton;

namespace Ui {
class SettingsDisplay;
}

class SettingsCompleter;

class SettingsDisplay : public QWidget {
    Q_OBJECT

public:
    explicit SettingsDisplay(QWidget *parent = nullptr);
    ~SettingsDisplay();

    int  changed();

    void save(int soft);

    bool isLightPenUsable();

public slots:
    void onCurrentMachineChanged(int machineId);

private slots:
    void on_checkBoxVoodoo_stateChanged(int state);
    void on_pushButtonConfigureVoodoo_clicked();
    void on_checkBoxPowerVR_stateChanged(int state);
    void on_pushButtonConfigurePowerVR_clicked();

    void on_checkBox8514_stateChanged(int state);
    void on_pushButtonConfigure8514_clicked();

    void on_checkBoxXga_stateChanged(int state);
    void on_pushButtonConfigureXga_clicked();

    void on_checkBoxDa2_stateChanged(int state);
    void on_pushButtonConfigureDa2_clicked();

    void on_radioButtonDefault_clicked();

    void on_radioButtonCustom_clicked();

    void on_pushButtonExportDefault_clicked();

private:
    Ui::SettingsDisplay *ui;

    int                  gfxcard_cfg_changed[GFXCARD_MAX]{};
    int                  voodoo_cfg_changed                 = 0;
    int                  powervr_cfg_changed                = 0;
    int                  ibm8514_cfg_changed                = 0;
    int                  xga_cfg_changed                    = 0;
    int                  ps55da2_cfg_changed                = 0;

    int machineId = 0;
    int videoCard[GFXCARD_MAX]{};
    QComboBox *videoSelectors[GFXCARD_MAX];
    QPushButton *videoConfigureButtons[GFXCARD_MAX];
    SettingsCompleter *videoCompleters[GFXCARD_MAX];

    QString previousEDIDPath;

    void updateVideoSelectors();
    void updateAccelerators();
    void configureVideoCard(int slot);

    int cga_hue, cga_saturation, cga_sharpness, cga_brightness, cga_contrast;
};

#endif // QT_SETTINGSDISPLAY_HPP
