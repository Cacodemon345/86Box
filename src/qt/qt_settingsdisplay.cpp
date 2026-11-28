/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          Display settings UI module.
 *
 * Authors: Joakim L. Gilje <jgilje@jgilje.net>
 *
 *          Copyright 2021 Joakim L. Gilje
 */
#include <QDebug>
#include <QFileDialog>
#include <QMessageBox>
#include <QStringBuilder>
#include <QLineEdit>
#include <QSignalBlocker>

#include <algorithm>
#include <cstdint>

extern "C" {
#include <86box/86box.h>
#include <86box/device.h>
#include <86box/machine.h>
#include <86box/video.h>
#include <86box/vid_ega.h>
#include <86box/vid_8514a_device.h>
#include <86box/vid_xga_device.h>
#include <86box/vid_ps55da2.h>
#include <86box/vid_ddc.h>
}

#include "qt_deviceconfig.hpp"
#include "qt_models_common.hpp"

#include "qt_settings_completer.hpp"
#include "qt_settingsdisplay.hpp"
#include "ui_qt_settingsdisplay.h"
#include "qt_util.hpp"
#include "qt_defs.hpp"
#include "qt_settingsinput.hpp"

SettingsDisplay::SettingsDisplay(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::SettingsDisplay)
{
    ui->setupUi(this);

    videoSelectors[0] = ui->comboBoxVideo;
    videoSelectors[1] = ui->comboBoxVideoSecondary;
    videoSelectors[2] = ui->comboBoxVideoThird;
    videoSelectors[3] = ui->comboBoxVideoFourth;
    videoConfigureButtons[0] = ui->pushButtonConfigureVideo;
    videoConfigureButtons[1] = ui->pushButtonConfigureVideoSecondary;
    videoConfigureButtons[2] = ui->pushButtonConfigureVideoThird;
    videoConfigureButtons[3] = ui->pushButtonConfigureVideoFourth;

    std::copy_n(gfxcard, GFXCARD_MAX, videoCard);
    for (int slot = 0; slot < GFXCARD_MAX; slot++) {
        videoCompleters[slot] = new SettingsCompleter(videoSelectors[slot], nullptr);
        connect(videoSelectors[slot], qOverload<int>(&QComboBox::currentIndexChanged), this, [this, slot](int row) {
            if (row < 0)
                return;
            videoCard[slot] = videoSelectors[slot]->itemData(row).toInt();
            updateVideoSelectors();
        });
        connect(videoConfigureButtons[slot], &QPushButton::clicked, this, [this, slot] { configureVideoCard(slot); });
    }

    ui->lineEditCustomEDID->setFilter(tr("EDID") % util::DlgFilter({ "bin", "dat", "edid", "txt" }) % tr("All files") % util::DlgFilter({ "*" }, true));
    connect(ui->lineEditCustomEDID, &FileField::fileSelected, [this](const QString &fileName) {
        uint8_t dummyBuffer[384] = { 0 };
        size_t size = ddc_load_edid(fileName.toUtf8().data(), dummyBuffer, sizeof(dummyBuffer));
        if ((size == 0) || (size > 256)) {
            QMessageBox::critical(this, "EDID", tr((size == 0) ? "EDID file \"%s\" is invalid." : "EDID file \"%s\" is too large.").replace("%s", "%1").arg(fileName));
            this->ui->lineEditCustomEDID->setFileName(this->previousEDIDPath);
        } else
            this->previousEDIDPath = fileName;
    });

    ui->comboBoxScreenType->addItem(tr("RGB Color"), 0);
    ui->comboBoxScreenType->addItem(tr("RGB Grayscale"), 1);
    ui->comboBoxScreenType->addItem(tr("Amber monitor"), 2);
    ui->comboBoxScreenType->addItem(tr("Green monitor"), 3);
    ui->comboBoxScreenType->addItem(tr("White monitor"), 4);
    ui->comboBoxScreenType->setCurrentIndex(video_grayscale);

    ui->comboBoxConversionType->addItem(tr("BT601 (NTSC/PAL)"), 0);
    ui->comboBoxConversionType->addItem(tr("BT709 (HDTV)"), 1);
    ui->comboBoxConversionType->addItem(tr("Average"), 2);
    ui->comboBoxConversionType->setCurrentIndex(video_graytype);

    ui->checkBoxOverscan->setChecked(enable_overscan);
    ui->checkBoxContrast->setChecked(vid_cga_contrast);

    ui->checkBoxInverted->setChecked(invert_display);

    ui->radioButtonDefault->setChecked(monitor_edid == 0);
    ui->radioButtonCustom->setChecked(monitor_edid == 1);
    ui->lineEditCustomEDID->setFileName(monitor_edid_path);
    ui->lineEditCustomEDID->setEnabled(monitor_edid == 1);
    previousEDIDPath = ui->lineEditCustomEDID->fileName();

    ui->checkBoxVoodoo->setChecked(voodoo_enabled);
    ui->checkBoxPowerVR->setChecked(powervr_enabled);
    ui->checkBox8514->setChecked(ibm8514_standalone_enabled);
    ui->checkBoxXga->setChecked(xga_standalone_enabled);
    ui->checkBoxDa2->setChecked(da2_standalone_enabled);

    onCurrentMachineChanged(machine);
}

SettingsDisplay::~SettingsDisplay()
{
    for (auto *completer : videoCompleters)
        delete completer;

    delete ui;
}

int
SettingsDisplay::changed()
{
    int has_changed  = 0;
    int soft_changed = 0;

    for (int slot = 0; slot < GFXCARD_MAX; slot++)
        has_changed |= gfxcard[slot] != videoCard[slot];

    has_changed  |= (voodoo_enabled             != (ui->checkBoxVoodoo->isChecked() ? 1 : 0));
    has_changed  |= (powervr_enabled            != (ui->checkBoxPowerVR->isChecked() ? 1 : 0));
    has_changed  |= (ibm8514_standalone_enabled != (ui->checkBox8514->isChecked() ? 1 : 0));
    has_changed  |= (xga_standalone_enabled     != (ui->checkBoxXga->isChecked() ? 1 : 0));
    has_changed  |= (da2_standalone_enabled     != (ui->checkBoxDa2->isChecked() ? 1 : 0));
    has_changed  |= (monitor_edid               != (ui->radioButtonCustom->isChecked() ? 1 : 0));

    has_changed  |= strcmp(monitor_edid_path, ui->lineEditCustomEDID->fileName().toUtf8().data());

    for (uint8_t i = 0; i < GFXCARD_MAX; i++)
        has_changed  |= gfxcard_cfg_changed[i];
    has_changed  |= voodoo_cfg_changed;
    has_changed  |= powervr_cfg_changed;
    has_changed  |= ibm8514_cfg_changed;
    has_changed  |= xga_cfg_changed;
    has_changed  |= ps55da2_cfg_changed;

    soft_changed |= (video_grayscale                != ui->comboBoxScreenType->currentData().toInt());
    soft_changed |= (video_graytype                 != ui->comboBoxConversionType->currentData().toInt());

    soft_changed |= (enable_overscan                != (ui->checkBoxOverscan->isChecked() ? 1 : 0));
    soft_changed |= (vid_cga_contrast               != (ui->checkBoxContrast->isChecked() ? 1 : 0));

    soft_changed |= (invert_display                 != (ui->checkBoxInverted->isChecked() ? 1 : 0));

    return has_changed ? (SETTINGS_CHANGED | SETTINGS_REQUIRE_HARD_RESET) :
                         (soft_changed ? SETTINGS_CHANGED : 0);
}

void
SettingsDisplay::save(int soft)
{
    video_grayscale = ui->comboBoxScreenType->currentData().toInt();
    video_graytype  = ui->comboBoxConversionType->currentData().toInt();

    update_overscan = 1;

    enable_overscan  = ui->checkBoxOverscan->isChecked() ? 1 : 0;
    vid_cga_contrast = ui->checkBoxContrast->isChecked() ? 1 : 0;

    invert_display = ui->checkBoxInverted->isChecked() ? 1 : 0;

    if (soft)
        goto end;

    std::copy_n(videoCard, GFXCARD_MAX, gfxcard);

    voodoo_enabled             = ui->checkBoxVoodoo->isChecked() ? 1 : 0;
    powervr_enabled            = ui->checkBoxPowerVR->isChecked() ? 1 : 0;
    ibm8514_standalone_enabled = ui->checkBox8514->isChecked() ? 1 : 0;
    xga_standalone_enabled     = ui->checkBoxXga->isChecked() ? 1 : 0;
    da2_standalone_enabled     = ui->checkBoxDa2->isChecked() ? 1 : 0;
    monitor_edid               = ui->radioButtonCustom->isChecked() ? 1 : 0;

    strncpy(monitor_edid_path, ui->lineEditCustomEDID->fileName().toUtf8().data(), sizeof(monitor_edid_path) - 1);

end:
    video_monitor_for_each([](monitor_t *monitor, void *) { cgapal_rebuild_monitor(monitor); }, nullptr);
}

void
SettingsDisplay::onCurrentMachineChanged(int machineId)
{
    this->machineId = machineId;
    updateVideoSelectors();
}

void
SettingsDisplay::updateVideoSelectors()
{
    const auto &choices = Models::Devices(video_card_getdevice, video_get_internal_name, video_card_available, 1);
    const bool fixedVideo = machine_has_flags(machineId, MACHINE_VIDEO_ONLY);
    int retained[GFXCARD_MAX]{};

    if (fixedVideo)
        videoCard[0] = VID_INTERNAL;

    /* Keep compatible choices on machine changes, prioritizing the earlier slots. */
    for (int slot = 0; slot < GFXCARD_MAX; slot++) {
        const int selected = videoCard[slot];
        if (std::any_of(choices.begin(), choices.end(), [selected](const Models::Device &entry) {
                return entry.id == selected && entry.available;
            }) && video_card_can_select(machineId, retained, slot, selected))
            retained[slot] = selected;
    }
    std::copy_n(retained, GFXCARD_MAX, videoCard);

    for (int slot = 0; slot < GFXCARD_MAX; slot++) {
        auto *combo = videoSelectors[slot];
        const QSignalBlocker blockSignals(combo);
        auto *completer = videoCompleters[slot];
        completer->removeRows();
        combo->clear();

        Models::Batch rows(combo->model());
        int selectedRow = 0;
        for (const auto &entry : choices) {
            if (!entry.available || !video_card_can_select(machineId, videoCard, slot, entry.id))
                continue;

            QString name = entry.name;
            if (entry.id == VID_INTERNAL) {
                const auto *device = machine_get_vid_device(machineId);
                if (device && device_available(device))
                    name += QStringLiteral(" (%1)").arg(DeviceConfig::DeviceName(device, device->internal_name, 0));
            }

            const int row = rows.add(name, entry.id);
            completer->addDevice(entry.dev, name);
            if (entry.id == videoCard[slot])
                selectedRow = row;
        }
        rows.commit();
        combo->setCurrentIndex(selectedRow);
        combo->setEnabled(slot == 0 ? !fixedVideo : combo->count() > 1);

        const auto *device = videoCard[slot] == VID_INTERNAL
                                 ? machine_get_vid_device(machineId)
                                 : video_card_getdevice(videoCard[slot]);
        videoConfigureButtons[slot]->setEnabled(device && device_has_config(device) && device_available(device));
    }

    updateAccelerators();
    if (Settings::settings && Settings::settings->input)
        Settings::settings->input->onCurrentMachineChanged(machineId);
}

void
SettingsDisplay::configureVideoCard(int slot)
{
    const int card = videoCard[slot];
    if (card == VID_INTERNAL)
        gfxcard_cfg_changed[slot] |= DeviceConfig::ConfigureDevice(machine_get_vid_device(machineId));
    else
        gfxcard_cfg_changed[slot] |= DeviceConfig::ConfigureDevice(video_card_getdevice(card), slot + 1);
}

void
SettingsDisplay::on_pushButtonConfigureVoodoo_clicked()
{
    voodoo_cfg_changed |= DeviceConfig::ConfigureDevice(&voodoo_device);
}

void
SettingsDisplay::on_pushButtonConfigurePowerVR_clicked()
{
    powervr_cfg_changed |= DeviceConfig::ConfigureDevice(&powervr_pcx_device);
}

void
SettingsDisplay::on_checkBoxPowerVR_stateChanged(int state)
{
    ui->pushButtonConfigurePowerVR->setEnabled(state == Qt::Checked);
}

void
SettingsDisplay::on_pushButtonConfigure8514_clicked()
{
    if (machine_has_bus(machineId, MACHINE_BUS_MCA) > 0) {
        ibm8514_cfg_changed |= DeviceConfig::ConfigureDevice(&ibm8514_mca_device);
    } else {
        ibm8514_cfg_changed |= DeviceConfig::ConfigureDevice(&gen8514_isa_device);
    }
}

void
SettingsDisplay::on_pushButtonConfigureXga_clicked()
{
    if (machine_has_bus(machineId, MACHINE_BUS_MCA) > 0)
        xga_cfg_changed |= DeviceConfig::ConfigureDevice(&xga_device);
}

void
SettingsDisplay::on_pushButtonConfigureDa2_clicked()
{
    ps55da2_cfg_changed |= DeviceConfig::ConfigureDevice(&ps55da2_device);
}

void
SettingsDisplay::updateAccelerators()
{
    static const QRegularExpression voodooRegex("3dfx|voodoo|banshee|raven", QRegularExpression::CaseInsensitiveOption);
    bool machineHasPci = machine_has_bus(machineId, MACHINE_BUS_PCI) > 0;
    ui->checkBoxPowerVR->setEnabled(machineHasPci);
    if (!machineHasPci)
        ui->checkBoxPowerVR->setChecked(false);
    ui->pushButtonConfigurePowerVR->setEnabled(machineHasPci && ui->checkBoxPowerVR->isChecked());

    bool machineHasIsa16 = machine_has_bus(machineId, MACHINE_BUS_ISA16) > 0;
    bool machineHasMca   = machine_has_bus(machineId, MACHINE_BUS_MCA) > 0;

    bool videoCardHas8514 = (video_get_primary_flags(machineId, videoCard[0]) == VIDEO_FLAG_TYPE_8514);
    bool videoCardHasXga  = (video_get_primary_flags(machineId, videoCard[0]) == VIDEO_FLAG_TYPE_XGA);
    bool videoCardHasDa2  = (video_get_primary_flags(machineId, videoCard[0]) == VIDEO_FLAG_TYPE_DA2);

    bool machineSupports8514 = ((machineHasIsa16 || machineHasMca) && !videoCardHas8514);
    bool machineSupportsXga  = ((machineHasMca && device_available(&xga_device)) && !videoCardHasXga);
    bool machineSupportsDa2  = (machineHasMca && device_available(&ps55da2_device) && !videoCardHasDa2);

    ui->checkBox8514->setEnabled(machineSupports8514);
    if (!machineSupports8514)
        ui->checkBox8514->setChecked(false);

    ui->pushButtonConfigure8514->setEnabled(ui->checkBox8514->isEnabled() && ui->checkBox8514->isChecked());

    ui->checkBoxXga->setEnabled(machineSupportsXga);
    if (!machineSupportsXga)
        ui->checkBoxXga->setChecked(false);

    ui->checkBoxDa2->setEnabled(machineSupportsDa2);
    if (!machineSupportsDa2)
        ui->checkBoxDa2->setChecked(false);

    ui->pushButtonConfigureXga->setEnabled(ui->checkBoxXga->isEnabled() && ui->checkBoxXga->isChecked());
    ui->pushButtonConfigureDa2->setEnabled(ui->checkBoxDa2->isEnabled() && ui->checkBoxDa2->isChecked());

    const bool hasVoodooCard = std::any_of(std::begin(videoSelectors), std::end(videoSelectors), [&](QComboBox *combo) {
        return combo->currentText().contains(voodooRegex);
    });
    if (hasVoodooCard) {
        const bool currentCardIsVoodoo = std::any_of(std::begin(gfxcard), std::end(gfxcard), [&](int card) {
            return DeviceConfig::DeviceName(video_card_getdevice(card), video_get_internal_name(card), 1).contains(voodooRegex);
        });
        // Preserve a manually configured add-on if a Voodoo card is already installed.
        if (!currentCardIsVoodoo)
            ui->checkBoxVoodoo->setChecked(false);
        ui->checkBoxVoodoo->setDisabled(true);
    } else {
        ui->checkBoxVoodoo->setEnabled(machineHasPci);
        if (!machineHasPci)
            ui->checkBoxVoodoo->setChecked(false);
    }

    ui->pushButtonConfigureVoodoo->setEnabled(machineHasPci && ui->checkBoxVoodoo->isChecked());

}

bool
SettingsDisplay::isLightPenUsable()
{
    return std::any_of(std::begin(videoCard), std::end(videoCard), [](int card) {
        const char *name = video_get_internal_name(card);
        return name && (!strcmp(name, "cga") || !strcmp(name, "plantronics"));
    });
}

void
SettingsDisplay::on_checkBoxVoodoo_stateChanged(int state)
{
    ui->pushButtonConfigureVoodoo->setEnabled(state == Qt::Checked);
}

void
SettingsDisplay::on_checkBox8514_stateChanged(int state)
{
    ui->pushButtonConfigure8514->setEnabled(state == Qt::Checked);
}

void
SettingsDisplay::on_checkBoxXga_stateChanged(int state)
{
    ui->pushButtonConfigureXga->setEnabled(state == Qt::Checked);
}

void
SettingsDisplay::on_checkBoxDa2_stateChanged(int state)
{
    ui->pushButtonConfigureDa2->setEnabled(state == Qt::Checked);
}

void
SettingsDisplay::on_radioButtonDefault_clicked()
{
    ui->radioButtonDefault->setChecked(true);
    ui->radioButtonCustom->setChecked(false);
    ui->lineEditCustomEDID->setEnabled(false);
}

void
SettingsDisplay::on_radioButtonCustom_clicked()
{
    ui->radioButtonDefault->setChecked(false);
    ui->radioButtonCustom->setChecked(true);
    ui->lineEditCustomEDID->setEnabled(true);
}

void
SettingsDisplay::on_pushButtonExportDefault_clicked()
{
    auto str = QFileDialog::getSaveFileName(this, tr("Export EDID"));
    if (!str.isEmpty()) {
        QFile file(str);
        if (file.open(QFile::WriteOnly)) {
            uint8_t *bytes = nullptr;
            auto     size  = ddc_create_default_edid(&bytes);
            file.write((char *) bytes, size);
            file.close();
            free(bytes);
        }
    }
}
