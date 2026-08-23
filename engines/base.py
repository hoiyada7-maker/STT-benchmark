"""STT 엔진 공통 인터페이스.

채점(benchmark)과 라이브 캡션(live) 양쪽이 같은 어댑터를 공유한다.
각 엔진은 16kHz mono float32 numpy 배열을 받아 텍스트를 반환한다.
"""

from abc import ABC, abstractmethod

import numpy as np


class SttEngine(ABC):
    """하나의 STT 엔진(모델+런타임)을 감싸는 추상 베이스."""

    #: 표시용 이름 (리포트·메뉴에 사용)
    name = "base"
    #: 채점 시 참고할 모델 크기/세부사항 (예: "small", "medium")
    model_id = ""

    def __init__(self, language="en", device="cpu"):
        self.language = language
        self.device = device

    @abstractmethod
    def load(self) -> None:
        """모델을 1회 로드한다. 이후 transcribe() 를 반복 호출한다."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """전체 오디오(16k mono float32)를 인식해 텍스트를 반환한다."""
