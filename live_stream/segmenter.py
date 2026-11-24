"""스트리밍 STT 의 확정(stable) 텍스트에서 영어 문장을 잘라내는 SentenceSegmenter.

규칙: ① . ? ! (+따옴표/괄호) 뒤에 공백과 다음 단어가 시작되면 문장 끝(약어·소수점 제외)
      ② 확정 텍스트가 pause_s 동안 늘지 않으면 남은 조각을 확정
      ③ 조각이 max_words 단어를 넘으면 마지막 쉼표/세미콜론(없으면 max_words)에서 잘라 왼쪽을 확정
텍스트는 스트림 전체 누적본(append-only)이며, 이미 내보낸 위치(off)를 기억한다.
"""
import re
import time

ABBR = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "etc.", "e.g.", "i.e.", "u.s.", "u.k.",
        "a.m.", "p.m.", "no."}
_END = re.compile(r"[.?!][\"')\]]*\s+(?=\S)")
_WORD = re.compile(r"\S+")
_LEAD = re.compile(r"[\s.?!,;]*")             # 이미 확정한 조각 뒤늦게 붙는 구두점은 버린다


class SentenceSegmenter:
    def __init__(self, pause_s=1.5, max_words=30):
        self.pause_s, self.max_words = pause_s, max_words
        self.off, self.n, self.t_grow = 0, 0, time.monotonic()

    def _is_end(self, text, m):
        """m 이 진짜 문장 끝인지: 직전 토큰이 약어/한 글자 이니셜이면 아니다."""
        tok = text[:m.start() + 1].split()[-1].lstrip("\"'([").lower()
        return tok not in ABBR and not (len(tok) == 2 and tok[0].isalpha() and tok[1] == ".")

    def _split(self, text):
        """text 에서 확정된 문장 목록과 남은 위치를 돌려준다(남은 조각은 확정 안 함)."""
        out, pos = [], 0
        while True:
            m = next((m for m in _END.finditer(text, pos) if self._is_end(text, m)), None)
            if m:
                out.append(text[pos:m.end()].strip()); pos = m.end(); continue
            ws = list(_WORD.finditer(text, pos))
            if len(ws) <= self.max_words:
                return out, pos
            cut = ws[self.max_words - 1].end()          # 기본: max_words 번째 단어 뒤
            c = max(text.rfind(",", pos, cut), text.rfind(";", pos, cut))
            if c > pos:
                cut = c + 1
            out.append(text[pos:cut].strip()); pos = cut

    def feed(self, text, now=None):
        """누적 확정 텍스트를 넣고 새로 확정된 문장 리스트를 돌려준다."""
        now = time.monotonic() if now is None else now
        if len(text) != self.n:
            self.n, self.t_grow = len(text), now
        self.off = _LEAD.match(text, min(self.off, len(text))).end()
        sents, rel = self._split(text[self.off:])
        self.off += rel
        return [s for s in sents if s] + self.tick(text, now)

    def tick(self, text, now=None):
        """유휴 시 호출: 침묵이 pause_s 이상이면 남은 조각을 확정."""
        now = time.monotonic() if now is None else now
        self.off = _LEAD.match(text, min(self.off, len(text))).end()
        rest = text[self.off:].strip()
        if rest and now - self.t_grow >= self.pause_s:
            self.off = len(text)
            return [rest]
        return []

    def flush(self, text):
        rest = text[self.off:].strip()
        self.off = len(text)
        return [rest] if rest else []


if __name__ == "__main__":
    s = SentenceSegmenter(pause_s=1.0, max_words=12)
    t = "Hello Mr. Smith. It costs 3.5 dollars. Is it"
    assert s.feed(t, 0) == ["Hello Mr. Smith.", "It costs 3.5 dollars."], s.off
    assert s.feed(t, 0.5) == []                                   # 아직 침묵 < 1s
    assert s.tick(t, 1.2) == ["Is it"]                            # 침묵 -> 조각 확정
    t += " ok. Dr. Who lives in the U.S. and likes e.g. tea"
    assert s.feed(t, 2) == ["ok."]
    assert s.feed(t, 2.1) == []                                   # Dr./U.S./e.g. 에서는 안 자른다
    t += ", so he drinks it a lot every single day"
    out = s.feed(t, 2.2)
    assert out == ["Dr. Who lives in the U.S. and likes e.g. tea,"], out   # 12단어 초과: 쉼표에서 컷
    assert s.feed(t, 2.3) == []
    assert s.tick(t, 4.0) == ["so he drinks it a lot every single day"]
    s2 = SentenceSegmenter(max_words=5)
    assert s2.feed("one two three four five six seven", 0) == ["one two three four five"]
    assert s2.flush("one two three four five six seven") == ["six seven"]
    t3 = "Final sentence."                                       # 끝 마침표 + 다음 단어 없음 -> pause
    s3 = SentenceSegmenter()
    assert s3.feed(t3, 0) == [] and s3.tick(t3, 1.5) == [t3]
    s4 = SentenceSegmenter(1.0)                                  # 확정 뒤 늦게 도착한 마침표는 버린다
    assert s4.feed("Thanks", 0) == [] and s4.tick("Thanks", 1.5) == ["Thanks"]
    assert s4.feed("Thanks.", 1.6) == [] and s4.tick("Thanks.", 3.0) == []
    assert s4.feed("Thanks. So", 3.1) == [] and s4.tick("Thanks. So", 4.5) == ["So"]
    print("segmenter OK")
