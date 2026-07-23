# 블로그 발행 가이드

블로그 초안 8편(`blog/00_index.md` + 7편)은 발행 준비가 끝났다. 이미지는 커밋된
`assets/`를 참조하므로 GitHub·정적사이트·외부 플랫폼 어디서든 렌더링된다.

## 옵션 A — GitHub Pages (가장 빠름, 무료, 소유권↑)
저장소에 `_config.yml`(Jekyll cayman 테마)이 이미 있다. 토글만 켜면 된다.

1. GitHub 저장소 → **Settings → Pages**
2. **Source**: `Deploy from a branch`, **Branch**: `master` / `(root)` → Save
3. 1–2분 후 `https://yeonkyunlee.github.io/signal-ml-lab/` 공개
4. README가 홈, 블로그는 `.../signal-ml-lab/blog/00_index` 등으로 접근

> 홈 화면을 블로그 인덱스로 바꾸려면 `blog/00_index.md`를 루트 `index.md`로 복사.

## 옵션 B — velog (한국어 개발 커뮤니티 노출↑)
1. velog.io 로그인 → 새 글
2. `blog/01_*.md` 내용을 붙여넣기 (마크다운 그대로 지원)
3. 이미지는 `assets/`의 PNG를 드래그 업로드 (velog가 자체 CDN에 올림)
4. 시리즈는 velog의 "태그"로 묶기 (예: `signal-ml-lab`)
5. 1편부터 순서대로, 각 편 끝에 다음 편 링크

## 옵션 C — Medium (영문 노출·해외 리치)
1. README(영문)와 블로그를 영문화 후 게시 (현재 블로그는 한국어)
2. Medium 임포트 기능(`Import a story`)에 GitHub Pages 글 URL을 넣으면 자동 변환
3. 코드블록·이미지 확인 후 발행

## 권장 순서
1. **먼저 GitHub Pages 켜기** (5분, 즉시 공개 URL 확보)
2. velog에 1~2편 발행해 반응 확인
3. 좋으면 나머지 편 + 영문 Medium 확장

## 발행 전 체크리스트
- [x] 이미지가 `assets/`(커밋됨) 참조 — 깨진 링크 없음
- [x] 회사/독점 데이터·알고리즘 미포함 (공개/합성 데이터만) — 개인 프로젝트
- [x] 시리즈 인덱스(`blog/00_index.md`) 링크 정상
- [ ] (선택) 영문화
- [ ] (선택) 대표 이미지(썸네일) 1장 선정 — `assets/03_benchmark_snr.png` 추천
