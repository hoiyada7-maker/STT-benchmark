"""정답 대비 전사 채점지(word-level diff) 마크다운 생성.

정답(ref)과 전사(hyp)를 단어 단위로 정렬해 다음 규칙으로 표기한다.

- 맞은 단어        : 그대로
- 틀리게 전사된 단어 : ``~~취소선~~``  (치환의 hyp 쪽, 또는 잉여 삽입)
- 정답에서 채워넣는 단어 : ``[**볼드**]`` (치환의 ref 쪽, 또는 누락)

예) 정답 "I love you" · 전사 "I like you"  ->  ``I ~~like~~[**love**] you``

정렬은 WER 과 동일한 단어 레벤슈타인(최소 편집) 기준으로 구한다.
대소문자·문장부호 차이는 **정렬 키에서만** 무시하고, 화면 표기는 원문 토큰을 그대로 쓴다.
"""

import string


def _norm_token(t: str) -> str:
    """정렬 비교용 정규화: 소문자화 + 양끝 문장부호 제거."""
    return t.lower().strip(string.punctuation)


def align_words(ref_tokens, hyp_tokens):
    """ref/hyp 토큰 리스트를 레벤슈타인으로 정렬해 연산 목록을 반환한다.

    반환: (op, ref_tok, hyp_tok) 튜플 리스트.
      op ∈ {"equal", "sub", "ins", "del"}
        equal : 일치            (ref_tok, hyp_tok)
        sub   : 치환(오인식)     (ref_tok, hyp_tok)
        ins   : 잉여(hyp 에만)   (None,    hyp_tok)
        del   : 누락(ref 에만)   (ref_tok, None)
    """
    r = [_norm_token(t) for t in ref_tokens]
    h = [_norm_token(t) for t in hyp_tokens]
    n, m = len(r), len(h)

    # DP 편집거리 표
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = r[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # 삭제(del)
                dp[i][j - 1] + 1,       # 삽입(ins)
                dp[i - 1][j - 1] + cost,  # 일치/치환
            )

    # 백트레이스
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0 and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + (0 if r[i - 1] == h[j - 1] else 1)
        ):
            op = "equal" if r[i - 1] == h[j - 1] else "sub"
            ops.append((op, ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", None, hyp_tokens[j - 1]))  # hyp 에만 있음
            j -= 1
        else:
            ops.append(("del", ref_tokens[i - 1], None))  # ref 에만 있음
            i -= 1
    ops.reverse()
    return ops


def render_body(ops) -> str:
    """연산 목록을 채점지 본문(마크다운)으로 렌더링한다."""
    out = []
    for op, rt, ht in ops:
        if op == "equal":
            out.append(ht)
        elif op == "sub":
            out.append(f"~~{ht}~~[**{rt}**]")
        elif op == "ins":
            out.append(f"~~{ht}~~")
        elif op == "del":
            out.append(f"[**{rt}**]")
    return " ".join(out)


def build_markdown(reference: str, hypothesis: str, *, title: str, meta: dict) -> str:
    """정답/전사 한 쌍으로 채점지 마크다운 전체 문서를 만든다.

    meta: 제목 아래 표에 넣을 키/값(엔진·모델·언어·WER·CER·RTF 등).
    """
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    ops = align_words(ref_tokens, hyp_tokens)

    n_eq = sum(1 for o in ops if o[0] == "equal")
    n_sub = sum(1 for o in ops if o[0] == "sub")
    n_ins = sum(1 for o in ops if o[0] == "ins")
    n_del = sum(1 for o in ops if o[0] == "del")

    lines = [f"# {title}", ""]

    # 메타 표
    lines.append("| 항목 | 값 |")
    lines.append("|---|---|")
    for k, v in meta.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # 오류 집계
    lines.append("## 오류 집계 (단어 단위)")
    lines.append("")
    lines.append("| 구분 | 개수 |")
    lines.append("|---|---|")
    lines.append(f"| 일치 | {n_eq} |")
    lines.append(f"| 치환(오인식) | {n_sub} |")
    lines.append(f"| 잉여(불필요) | {n_ins} |")
    lines.append(f"| 누락(빠짐) | {n_del} |")
    lines.append(f"| 정답 단어수 | {len(ref_tokens)} |")
    lines.append(f"| 전사 단어수 | {len(hyp_tokens)} |")
    lines.append("")

    # 표기 범례
    lines.append("## 표기 범례")
    lines.append("")
    lines.append("- 일반 텍스트 = 맞은 단어")
    lines.append("- `~~취소선~~` = 틀리게 전사된 단어(오인식 또는 불필요하게 낀 단어)")
    lines.append("- `[**볼드**]` = 정답에서 끌어와 채워넣는 단어(오인식의 정답 또는 빠뜨린 단어)")
    lines.append("- 예) 정답 \"I love you\" · 전사 \"I like you\" → I ~~like~~[**love**] you")
    lines.append("")

    # 채점 본문
    lines.append("## 채점지")
    lines.append("")
    lines.append(render_body(ops))
    lines.append("")

    return "\n".join(lines)
