# DRAFT — Genesis repo Issue #2 (masrudyn) — patch_genesis_unified.py

**Status:** DRAFT, NOT POSTED. Awaiting Sander GO.

Per `feedback_github_comment_style.md`:
- Sander voice (warm, disclaimer, conversational)
- ALWAYS English + Russian side-by-side review

---

## Issue context

masrudyn opened 2026-04-26: "What replaced this script?"

Issue is open ~24h+, unanswered. Likely the same migration question that
noonghunna asked in `noonghunna/qwen36-27b-single-3090#2`. We shipped a
backward-compat shim at the repo root (`patch_genesis_unified.py`) on
2026-04-27 (commit 87b8738) that tells users to switch to the modular
invocation. The shim itself works (just emits a DeprecationWarning).

## Draft (English — ready to post)

Hi @masrudyn, thanks for asking!

(Small disclaimer: I'm from Ukraine and my English is still a work in progress, so I'm using AI to help with translation. Hope it reads okay!)

Good catch — `patch_genesis_unified.py` was the monolithic patch loader from Genesis v7.13 and earlier. From v7.14 onwards it's been replaced by a modular package at `vllm/_genesis/` plus a single invocation:

```bash
python3 -m vllm._genesis.patches.apply_all
```

For backward compatibility I shipped a thin shim at the repo root (still called `patch_genesis_unified.py`) on 2026-04-27 that just calls the new modular `apply_all` and emits a `DeprecationWarning`. So if you're using existing compose files or launch scripts that mount the old path, they should keep working without changes. You'll see a warning but everything functions as before.

If you want to migrate to the modern invocation, here's what to change in your launch script / Dockerfile:

```bash
# OLD (v7.13 and earlier):
python3 patch_genesis_unified.py

# NEW (v7.14+):
python3 -m vllm._genesis.patches.apply_all
```

The modular package gives a few things the monolith didn't:
- Per-patch dispatcher matrix (you can see exactly which patches applied + why others were skipped — `python3 -m vllm._genesis.dispatcher` prints the full table)
- Opt-in env flags per patch (e.g., `GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1`)
- Auto-detection of upstream PR drift (skips patches when upstream merged the fix)
- Safety gates (e.g., P67 auto-disables when no spec-decode is configured)

Recent docs that might help:
- [README.md](https://github.com/Sandermage/genesis-vllm-patches/blob/main/README.md) — production status v7.59 (320K context now)
- [CONFIGURATION.md](https://github.com/Sandermage/genesis-vllm-patches/blob/main/CONFIGURATION.md) — every env flag
- [PATCHES.md](https://github.com/Sandermage/genesis-vllm-patches/blob/main/PATCHES.md) — all 63 patches with status

Let me know if you hit any issues with the migration — happy to help debug.

Closing this for now since the shim is in place; please reopen if you find something specific that doesn't work!

— Sander

---

## Черновик (русский — только для проверки тут)

Привет @masrudyn, спасибо за вопрос!

(Маленькая ремарка: я из Украины, мой английский ещё в процессе развития, использую AI для перевода. Надеюсь, читается нормально!)

Хорошая находка — `patch_genesis_unified.py` это был монолитный загрузчик патчей в Genesis v7.13 и раньше. Начиная с v7.14 он заменён модульным пакетом `vllm/_genesis/` плюс одним вызовом:

```bash
python3 -m vllm._genesis.patches.apply_all
```

Для обратной совместимости я выложил тонкий шим в корне репо (тоже называется `patch_genesis_unified.py`) 2026-04-27, который просто вызывает новый модульный `apply_all` и эмитит `DeprecationWarning`. Так что если ты используешь существующие compose файлы / launch скрипты которые маунтят старый путь — они должны продолжать работать без изменений. Увидишь warning, но всё функционально как было.

Если хочешь перейти на современный вызов, вот что поменять в launch скрипте / Dockerfile:

```bash
# СТАРОЕ (v7.13 и раньше):
python3 patch_genesis_unified.py

# НОВОЕ (v7.14+):
python3 -m vllm._genesis.patches.apply_all
```

Модульный пакет даёт несколько штук которых не было в монолите:
- Per-patch dispatcher matrix (видно какие патчи применились + почему другие пропустились — `python3 -m vllm._genesis.dispatcher` печатает полную таблицу)
- Opt-in env флаги per patch (типа `GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1`)
- Auto-detection upstream PR drift (пропускает патчи когда upstream смерджил фикс)
- Safety gates (например, P67 auto-disable когда нет spec-decode в конфиге)

Свежие доки которые могут помочь:
- README.md — production status v7.59 (320K контекст теперь)
- CONFIGURATION.md — каждый env flag
- PATCHES.md — все 63 патча со статусом

Дай знать если упрёшься в проблемы при миграции — рад помочь дебажить.

Закрываю пока что-раз шим уже на месте; пожалуйста переоткрой если найдёшь что-то конкретное что не работает!

— Sander

---

## Pre-post checklist

- [ ] Sander reads Russian + English drafts
- [ ] Sander approves "post it" / "да публикуй"
- [ ] `gh issue comment 2 -R Sandermage/genesis-vllm-patches --body "<English body>"`
- [ ] (optional) `gh issue close 2 -R Sandermage/genesis-vllm-patches` if Sander wants it closed
