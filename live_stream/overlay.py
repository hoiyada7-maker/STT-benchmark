"""최소 자막 오버레이 (PyQt6): 프레임 없음 / 항상 위 / 반투명 / 드래그 이동 / 크기 조절 / Esc·✕ 종료.

set_text(str) 는 어느 스레드에서든 호출 가능(Qt 시그널로 UI 스레드에 전달).
"""
from PyQt6 import QtCore, QtGui, QtWidgets

BOTTOM_PX = 4  # 스크롤바가 맨 아래에서 이만큼 이내면 "맨 아래"로 본다 (자동 따라가기)


class _Follow(QtWidgets.QPlainTextEdit):
    """리사이즈 때도 맨 아래였으면 맨 아래를 유지하는 읽기 전용 텍스트 뷰."""

    def at_bottom(self):
        bar = self.verticalScrollBar()
        return bar.value() >= bar.maximum() - BOTTOM_PX

    def resizeEvent(self, e):
        follow = self.at_bottom()
        super().resizeEvent(e)
        if follow:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class CaptionOverlay(QtWidgets.QWidget):
    _text = QtCore.pyqtSignal(str)

    def __init__(self, color="white", below=None):
        """color: 글자색, below: 이 오버레이 아래에 붙여 여는 다른 CaptionOverlay."""
        super().__init__()
        self._drag = None
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.WindowStaysOnTopHint
                            | QtCore.Qt.WindowType.Tool)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.label = _Follow(self)
        self.label.setReadOnly(True)
        self.label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.NoTextInteraction)
        self.label.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 항상 켜 둔다: 스크롤바가 처음 나타날 때 뷰포트 폭이 줄어 전체가 reflow 되는 것을 방지
        self.label.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.label.setFont(QtGui.QFont("Malgun Gothic", 24, QtGui.QFont.Weight.Bold))
        self.label.setStyleSheet(
            f"QPlainTextEdit {{ background: rgba(0,0,0,170); color: {color}; border: none; border-radius: 14px;"
            " padding: 10px 18px; }"
            "QScrollBar:vertical { background: transparent; width: 8px; margin: 14px 2px; }"
            "QScrollBar::handle:vertical { background: rgba(255,255,255,90); border-radius: 3px; min-height: 24px; }"
            "QScrollBar::handle:vertical:disabled { background: transparent; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }")
        self.label.viewport().installEventFilter(self)   # 텍스트 영역을 눌러도 창 드래그
        self._shown = None
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
        if below:                                  # 번역창: 자막창 바로 아래(화면 밖이면 위로)
            y = below.y() + below.height() + 6
            self.move(below.x(), y if y + h <= scr.bottom() else max(scr.y(), below.y() - h - 6))
        self._text.connect(self._apply)
        self.set_text("")

    def set_text(self, text):
        self._text.emit(text)

    @QtCore.pyqtSlot(str)
    def _apply(self, text):
        text = text or "..."
        old, w = self._shown, self.label
        if text == old:
            return
        follow = w.at_bottom()                     # 맨 아래였을 때만 따라간다
        if old and text.startswith(old):           # 보통: 뒤에 붙은 부분만 추가 -> 앞줄은 움직이지 않음
            cur = QtGui.QTextCursor(w.document())
            cur.movePosition(QtGui.QTextCursor.MoveOperation.End)
            cur.insertText(text[len(old):])
        else:                                      # 드문 수정/자리표시자 교체: 전체 교체
            w.setPlainText(text)
        self._shown = text
        if follow:
            w.verticalScrollBar().setValue(w.verticalScrollBar().maximum())

    def eventFilter(self, obj, e):
        t = e.type()
        if obj is self.label.viewport() and t in (QtCore.QEvent.Type.MouseButtonPress,
                                                  QtCore.QEvent.Type.MouseMove,
                                                  QtCore.QEvent.Type.MouseButtonRelease):
            {QtCore.QEvent.Type.MouseButtonPress: self.mousePressEvent,
             QtCore.QEvent.Type.MouseMove: self.mouseMoveEvent,
             QtCore.QEvent.Type.MouseButtonRelease: self.mouseReleaseEvent}[t](e)
            return True
        return super().eventFilter(obj, e)

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
