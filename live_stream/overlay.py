"""최소 자막 오버레이 (PyQt6): 프레임 없음 / 항상 위 / 반투명 / 드래그 이동 / 크기 조절 / Esc·✕ 종료.

set_text(str) 는 어느 스레드에서든 호출 가능(Qt 시그널로 UI 스레드에 전달).
"""
from PyQt6 import QtCore, QtGui, QtWidgets

TAIL_CHARS = 200  # 누적 텍스트에서 마지막 ~200자만 표시 (단어 경계에서 자름)


def tail(text, n=TAIL_CHARS):
    if len(text) <= n:
        return text
    t = text[-n:]
    # 단어 중간에서 잘렸으면 첫 공백까지 버린다 (공백 없는 한글/CJK 는 그대로 둔다)
    return t.split(" ", 1)[1] if text[-n - 1] != " " and " " in t else t


class CaptionOverlay(QtWidgets.QWidget):
    _text = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._drag = None
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.WindowStaysOnTopHint
                            | QtCore.Qt.WindowType.Tool)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.label = QtWidgets.QLabel(self)
        self.label.setWordWrap(True)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.label.setFont(QtGui.QFont("Malgun Gothic", 24, QtGui.QFont.Weight.Bold))
        self.label.setStyleSheet("background: rgba(0,0,0,170); color: white; border-radius: 14px; padding: 10px 18px;")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.label)
        self.close_btn = QtWidgets.QPushButton("✕", self)
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.setStyleSheet("color: white; background: rgba(255,255,255,40); border: none; border-radius: 12px;")
        self.close_btn.clicked.connect(QtWidgets.QApplication.quit)
        self.grip = QtWidgets.QSizeGrip(self)
        self.grip.setFixedSize(16, 16)
        scr = QtWidgets.QApplication.primaryScreen().availableGeometry()
        w, h = int(scr.width() * 0.7), 170
        self.setMinimumSize(300, 80)
        self.resize(w, h)
        self.move(scr.x() + (scr.width() - w) // 2, scr.y() + scr.height() - h - 80)
        self._text.connect(self._apply)
        self.set_text("")

    def set_text(self, text):
        self._text.emit(text)

    @QtCore.pyqtSlot(str)
    def _apply(self, text):
        self.label.setText(tail(text) or "...")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.grip.move(self.width() - 16, self.height() - 16)
        self.close_btn.move(self.width() - 30, 6)
        self.grip.raise_()
        self.close_btn.raise_()

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key.Key_Escape:
            QtWidgets.QApplication.quit()
