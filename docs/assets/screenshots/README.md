# GUI / TUI screenshots

Drop rendered captures here; the README and MANUAL reference them.

| File | What to capture | How |
|---|---|---|
| `memory-graph.png` | The Memory panel **Graph view** — force-directed graph, community-colored clouds, a selected node's detail | GUI → Engine → 🧠 Memory → "Graph"; from a deployed `:8811` or `npm run dev` in `gui/web` |
| `memory-list.png` | The Memory panel **List view** — search box, Brain-recall toggle, results, node connections | same panel, "List" |
| `gateway-flow.png` | (optional) a chat client answering from injected memory through the gateway | any OpenAI client pointed at `:8811/v1/chat/completions` |

## How to capture

```bash
# dev (hot-reload), talks to a running product-API
cd gui/web && npm install && npm run dev      # http://127.0.0.1:5173
# or the deployed unified container
docker run -d -p 8811:8800 ... genesis-memory:dev   # http://<host>:8811
```

Open the **Memory** section (Engine group, Brain icon), seed a few memories +
`Rebuild` to form clouds, then screenshot the List and Graph views (PNG, ~1400px
wide). Keep them lightweight; this dir is for documentation assets only.

> The README's architecture + brain-mechanics **Mermaid diagrams** and the ASCII
> panel mockup render directly on GitHub and need no binary assets — these PNGs
> are the optional polished captures of the live React UI.
