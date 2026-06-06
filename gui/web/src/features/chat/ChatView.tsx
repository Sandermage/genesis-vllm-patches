// SPDX-License-Identifier: Apache-2.0
import { useState, useRef, useEffect } from 'react';
import {
  Tile, Button, TextArea, Tag, InlineLoading,
} from '@carbon/react';

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

/**
 * ChatView — operator-facing chat sandbox against the active model.
 *
 * Talks to the engine's ``/v1/chat/completions`` directly. Useful for
 * verifying a live container actually serves requests, sanity-checking
 * tool-call output, and reproducing tool-call bugs reported by users.
 */
export function ChatView(): JSX.Element {
  const [endpoint, setEndpoint] = useState<string>(
    localStorage.getItem('sndr-chat-endpoint') ?? 'http://localhost:8102'
  );
  const [model, setModel] = useState<string>(
    localStorage.getItem('sndr-chat-model') ?? 'qwen3.6-35b-a3b'
  );
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const persist = () => {
    localStorage.setItem('sndr-chat-endpoint', endpoint);
    localStorage.setItem('sndr-chat-model', model);
  };

  const send = async () => {
    if (!input.trim()) return;
    persist();
    setError(null);
    const updatedMessages = [...messages, { role: 'user' as const, content: input }];
    setMessages(updatedMessages);
    setInput('');
    setLoading(true);

    try {
      const apiToken = localStorage.getItem('sndr-api-token') ?? '';
      const resp = await fetch(`${endpoint}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiToken && { Authorization: `Bearer ${apiToken}` }),
        },
        body: JSON.stringify({
          model,
          messages: updatedMessages,
          max_tokens: 512,
          temperature: 0.3,
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
      const data = await resp.json();
      const assistantMessage = data.choices?.[0]?.message?.content ?? '(empty)';
      setMessages([...updatedMessages, { role: 'assistant', content: assistantMessage }]);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-view">
      <h2 className="cds--type-heading-04">Chat</h2>

      <Tile style={{ marginBottom: 16, display: 'flex', gap: 12, alignItems: 'flex-end' }}>
        <div style={{ flex: '1 1 0' }}>
          <label className="cds--label">Endpoint</label>
          <input
            className="cds--text-input"
            value={endpoint}
            onChange={(e) => setEndpoint(e.target.value)}
          />
        </div>
        <div style={{ flex: '1 1 0' }}>
          <label className="cds--label">Model</label>
          <input
            className="cds--text-input"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          />
        </div>
        <Button kind="ghost" size="md" onClick={() => setMessages([])}>Clear</Button>
      </Tile>

      <Tile style={{ minHeight: 300, marginBottom: 16, padding: 0, overflow: 'auto' }}>
        <div style={{ padding: 16, maxHeight: 480, overflow: 'auto' }}>
          {messages.length === 0 && (
            <p className="cds--type-helper-text-01" style={{ textAlign: 'center', padding: 32 }}>
              No messages yet. Type below to start.
            </p>
          )}
          {messages.map((m, i) => (
            <div key={i} style={{ marginBottom: 16 }}>
              <Tag type={m.role === 'user' ? 'blue' : 'green'}>{m.role}</Tag>
              <p style={{ marginTop: 4, whiteSpace: 'pre-wrap' }}>{m.content}</p>
            </div>
          ))}
          {loading && <InlineLoading description="Generating…" />}
          {error && <p style={{ color: 'red' }}>{error}</p>}
          <div ref={messagesEnd} />
        </div>
      </Tile>

      <Tile>
        <TextArea
          id="chat-input"
          labelText="Message"
          placeholder="Type a message…"
          rows={3}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e: any) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              send();
            }
          }}
        />
        <div style={{ marginTop: 8 }}>
          <Button kind="primary" size="md" onClick={send} disabled={loading || !input.trim()}>
            Send (⌘+⏎)
          </Button>
        </div>
      </Tile>
    </div>
  );
}

export default ChatView;
