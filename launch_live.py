"""Run RGB MoCap Live from its private Windows installation."""
import argparse
import ctypes
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from tools.source_launch import configure, verify
from tools.session_lock import app_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path)
    args = parser.parse_args()
    configure(root)
    with app_lock(root):
        from mocap.runtime_registry import activate
        try:
            verify(root)
            activate(root, 'gui')
        except RuntimeError as error:
            raise SystemExit(str(error)) from None
        from mocap.runtime_limits import configure_preview_cpu
        configure_preview_cpu()
        from PySide6.QtGui import QIcon, QSurfaceFormat
        from PySide6.QtWidgets import QApplication
        from mocap.live_window import LiveWindow
        from mocap.theme import configure_application, load_appearance
        if sys.platform == 'win32':
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('toyxyz.RGBMoCapLive')
        surface = QSurfaceFormat.defaultFormat()
        surface.setSwapInterval(0)
        QSurfaceFormat.setDefaultFormat(surface)
        app = QApplication(sys.argv[:1])
        app.setApplicationName('RGB MoCap Live')
        app.setWindowIcon(QIcon(str(root / 'assets/rgbmocap-icon.png')))
        configure_application(app, *load_appearance())
        window = LiveWindow(args.video)
        window.show()
        app.aboutToQuit.connect(window.shutdown_preview)
        return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
