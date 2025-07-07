#include <QOpenGLWindow>
#include <QOpenGLFunctions>

#include "qt_renderercommon.hpp"

class HardwareRenderer : public QOpenGLWindow, protected QOpenGLFunctions, public RendererCommon {
    Q_OBJECT

    HardwareRenderer() {}
};