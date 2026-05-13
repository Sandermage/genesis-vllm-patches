# Reasoning vs Content контракт для Qwen3-class моделей

**Применимость**: Qwen3-6 (27B, 35B-A3B и аналогичные), которые поддерживают
"thinking-mode". Не относится к non-reasoning моделям.

## Проблема

При запросе с малым `max_tokens` (32/96/192) на Qwen3 model сервер
может вернуть `message.content = null` и весь бюджет токенов в
`message.reasoning`. OpenAI-совместимые клиенты, ожидающие непустой
`content`, видят пустой ответ.

Симптом (live PROD smoke, 27B на dev209):

```json
{
  "finish_reason": "length",
  "message": {
    "role": "assistant",
    "content": null,
    "reasoning": "Here's a thinking process..."
  },
  "usage": {"completion_tokens": 32}
}
```

Все 32 токена ушли в reasoning, content пустой.

## Архитектурное решение Genesis

Patch `PN16 lazy-reasoner` v2 (см. `vllm/sndr_core/middleware/lazy_reasoner.py`)
явно отказался от мутации `enable_thinking` через chat-template (вариант V1).
Причина — CUDA-graph dispatch регрессия на 28% TPS и 6× CV (Wave 6 closure).

Вместо этого PN16 v2 предлагает 4 варианта:

| Вариант | Env-флаг | Поведение | Когда включать |
|---|---|---|---|
| **V3 client override** | (default ON) | Respect client `chat_template_kwargs.enable_thinking` | всегда |
| **V5 soft cap** | `GENESIS_PN16_MAX_THINKING_TOKENS>0` | append `<think>` budget hint в последний user msg | RAG / single-shot |
| **V7 hard max_tokens cap** | `GENESIS_PN16_CLASSIFIER_MAX_TOKENS>0` | classifier detect short prompt → cap `max_tokens` | short-answer / IDE agents |
| **V8 tool-think budget** | `GENESIS_PN16_TOOL_THINK_BUDGET>0` | prepend system-msg "reason ≤ N tokens before tool_call" | tool-call workflows |

## Контракт для клиентов

### Клиент знает, что reasoning не нужен (short answer / system smoke)

**Передавай явно**:

```python
import openai

resp = openai.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "Say OK"}],
    max_tokens=32,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)  # → "OK"
```

`curl` эквивалент:

```bash
curl -fsS -X POST http://localhost:8101/v1/chat/completions \
  -H "Authorization: Bearer genesis-local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role":"user","content":"Say OK"}],
    "max_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Это zero-cost — никакого regression'а CUDA-graph, никакого MTP draft
bias'а (требует thinking-enabled traces): просто PN16 V3 уважает
явный client override.

### Клиент НЕ знает контекст (community OpenAI-compatible SDK)

Опция A — **server-side V7** через env при boot:

```bash
docker run ... \
  -e GENESIS_PN16_CLASSIFIER_MAX_TOKENS=128 \
  ...
```

Classifier эвристически детектит "тривиальные" prompts и режет
`max_tokens` до 128. Chat template НЕ мутируется → CUDA-graphs hit
по-прежнему. Trade-off: ответ на short questions ограничен 128
токенами максимум.

Опция B — **server-side default через chat template**: для preset'а
зашить `enable_thinking=False` дефолтным; reasoning по запросу:

```yaml
# vllm/sndr_core/model_configs/builtin/<preset>.yaml
genesis_env:
  GENESIS_PN16_CLASSIFIER_MAX_TOKENS: "128"
  # либо vllm-side default флаг (если поддерживается):
  # VLLM_DEFAULT_ENABLE_THINKING: "false"
```

## CI smoke gate

Скрипт `tools/openai_smoke.py` проверяет контракт content-not-null:

```bash
make smoke-content                            # default: 127.0.0.1:8101
HOST=http://your-host:8101 make smoke-content
ENABLE_THINKING=false make smoke-content       # подтверждает V3 path
```

Exit codes:

- `0` — content получен
- `2` — content пустой (assertion failure)
- `1` / `3` — HTTP / response shape error

## Связанные файлы

- `vllm/sndr_core/middleware/lazy_reasoner.py` — PN16 v2 импл
- `vllm/sndr_core/integrations/middleware/pn16_lazy_reasoner.py` — wiring
- `tools/openai_smoke.py` — smoke-тест
- `Makefile::smoke-content` — CI gate target
