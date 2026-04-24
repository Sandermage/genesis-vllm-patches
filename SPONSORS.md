# Project Support · Поддержка проекта

> 🇬🇧 English first · 🇷🇺 Русский ниже

---

## 🇬🇧 A note from the author

Hi. My name is **Sandermage (Aleksandr Barzov)**, I'm from Odessa, Ukraine.

This repository is my personal research project on getting the most out of vLLM for Qwen3-class models on consumer-grade hardware — specifically 2× RTX A5000 (Ampere SM 8.6), the configuration that the upstream officially treats as "best-effort". You'll find runtime patches, defensive guards, monkey-patches, benchmarks and a TDD-style test harness inside. All of it is, and will always remain, **open under Apache-2.0 with no strings attached**.

I want to be very clear about one thing up front:

> **This page is not a request, not an expectation, and not a condition for using anything in this repository.** I am not asking for money. I am not soliciting donations. The code is here for everyone to use, study, fork and improve, freely.

This section exists for one reason only: a few people over time have asked *"how can I say thank you?"* — and it felt rude not to answer them. So this is the answer, with no pressure attached.

### About the test matrix

I can only verify and debug on the hardware that's physically in front of me: 2× RTX A5000. For everything else — AMD ROCm, Intel XPU, Hopper, Blackwell, native FP8, NVFP4 and friends — I write the patches *defensively* (graceful skip when the platform doesn't match), but I genuinely **cannot test them on real silicon**.

A proper cross-vendor test bench (something with NVIDIA RTX PRO 6000 Blackwell, an Intel XPU box, an AMD ROCm CDNA card) is in a different cost bracket than what I can self-fund right now. Building it would let me actually validate cross-platform behaviour rather than rely on the "graceful skip" guard. I've already invested a meaningful amount of my own savings into this line of research, and at some future point — when I have free funds available — I plan to buy more hardware myself, on my own terms and timeline. That's the path I'm comfortable with.

### If you'd like to say thanks

If my work has been useful to you and you'd like to support it of your own initiative — that's appreciated, and I'm grateful for it. Any amount, any way is fine, including not at all. The wallets below are simply there for those who asked:

| Network | Address |
|---|---|
| **USDT (BEP-20)** | `0x1E8C74aC4f37A201733D185b2293e9D69f305306` |
| **USDT (TRC-20)** | `TSyVYTA4PK22w3tZ7vgoc1itjXU5p4Vfks` |
| **ETH** | `0x1E8C74aC4f37A201733D185b2293e9D69f305306` |
| **BTC** | `bc1q9tau6xqgrv5jjgst63yjux550gslq6nm7y7q9f` |
| **PayPal** | `sander.odessa@gmail.com` |

If you'd rather **lend or send a card for the test bench** instead of money — any Hopper / Blackwell / R6000 / H100 / Intel XPU / AMD ROCm class — please reach out at the email above and we'll figure out logistics together.

### What I will and won't do

- I **will** keep publishing everything I do under Apache-2.0, including bench results, methodology and raw logs.
- I **will** credit every upstream author and contributor whose work I build on, in patch docstrings and report files.
- I **will not** ever gate functionality behind donations, paywalls, "premium tiers" or anything similar.
- I **will not** treat support as obligation — if you've sent something, thank you, and that's where the relationship ends; it does not buy priority, custom features, or any kind of claim on my time.

Thank you for reading, and especially thank you for using the work.

— Sander

---

## 🇷🇺 Несколько слов от автора

Привет. Меня зовут **Sandermage (Александр Барзов)**, я из Одессы, Украина.

Этот репозиторий — мой персональный исследовательский проект по тому, как выжать максимум из vLLM для моделей класса Qwen3 на потребительском железе — конкретно на 2× RTX A5000 (Ampere SM 8.6), той конфигурации, которую upstream формально считает "best-effort". Внутри — runtime-патчи, defensive guards, monkey-patches, бенчмарки и TDD-инфраструктура. Всё это есть и навсегда останется **открытым под Apache-2.0 без каких-либо оговорок**.

Хочу сразу очень чётко обозначить одну вещь:

> **Эта страница — не просьба, не ожидание и не условие пользования чем-либо из этого репозитория.** Я не прошу денег. Я не клянчу донаты. Код здесь — для всех, чтобы пользоваться, изучать, форкать и улучшать, свободно.

Этот раздел появился только по одной причине: несколько раз меня спрашивали *«как сказать спасибо?»* — и было бы невежливо им не ответить. Это и есть ответ, без какого-либо давления.

### О тестовой матрице

Я могу проверить и отладить только на том железе, которое физически стоит у меня под рукой: 2× RTX A5000. Для всего остального — AMD ROCm, Intel XPU, Hopper, Blackwell, native FP8, NVFP4 и подобного — я пишу патчи *defensively* (graceful skip когда платформа не подходит), но **на реальном железе их протестировать не могу**.

Собрать нормальный cross-vendor тестовый стенд (что-то уровня NVIDIA RTX PRO 6000 Blackwell, Intel XPU, AMD ROCm CDNA в одном месте) — это совсем другая весовая категория, не та что я могу позволить себе сейчас из личных средств. Такой стенд позволил бы реально валидировать cross-platform поведение, а не полагаться на "graceful skip" guard. Я уже вложил в это направление исследований заметную часть собственных накоплений, и в какой-то момент в будущем — когда будут свободные средства — я планирую докупать железо сам, на своих условиях и в своём темпе. Это путь, в котором мне комфортно.

### Если хочется сказать спасибо

Если моя работа оказалась полезной и есть желание поддержать её **по собственной инициативе** — буду благодарен. Любая сумма, любым способом подойдёт, включая «никак». Кошельки ниже — просто для тех, кто спрашивал:

| Сеть | Адрес |
|---|---|
| **USDT (BEP-20)** | `0x1E8C74aC4f37A201733D185b2293e9D69f305306` |
| **USDT (TRC-20)** | `TSyVYTA4PK22w3tZ7vgoc1itjXU5p4Vfks` |
| **ETH** | `0x1E8C74aC4f37A201733D185b2293e9D69f305306` |
| **BTC** | `bc1q9tau6xqgrv5jjgst63yjux550gslq6nm7y7q9f` |
| **PayPal** | `sander.odessa@gmail.com` |

Если есть желание **передать карту в пользование** для расширения тестового стенда (любой Hopper / Blackwell / R6000 / H100 / Intel XPU / AMD ROCm класс) вместо денег — напишите на e-mail выше, обсудим логистику.

### Что я делаю и чего не делаю

- Я **буду** продолжать публиковать всё что делаю под Apache-2.0, включая результаты бенчмарков, методологию и сырые логи.
- Я **буду** благодарить каждого upstream-автора и контрибьютора, на чьей работе строю свою — в docstring патчей и в файлах отчётов.
- Я **не буду** никогда закрывать функциональность за донатами, paywall, "premium tier" или чем-то подобным.
- Я **не буду** относиться к поддержке как к обязательству. Если что-то прислали — спасибо, и на этом всё. Это не покупает приоритет, кастомные фичи или любое право на моё время.

Спасибо что читаете, и особенно — спасибо что пользуетесь.

— Sander
