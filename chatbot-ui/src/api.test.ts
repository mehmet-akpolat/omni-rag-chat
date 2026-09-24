import { afterEach, describe, expect, it, vi } from 'vitest';
import { closeChatSessionOnUnload, endChatSession, getChatSession, listKnowledgeBases, requestHuman, startChatSession, streamAnswer } from './api';

afterEach(() => vi.unstubAllGlobals());

describe('chat API client', () => {
  it('lists sources and creates a human-support ticket', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify([{id: 'kb', name: 'Guide'}]), {status: 200}))
      .mockResolvedValueOnce(new Response(JSON.stringify({ticket_id: 'HUM-123'}), {status: 202}));
    vi.stubGlobal('fetch', fetchMock);
    expect(await listKnowledgeBases('company')).toHaveLength(1);
    expect((await requestHuman('session')).ticket_id).toBe('HUM-123');
  });

  it('parses streamed metadata, tokens, and citations', async () => {
    const events = [
      'event: meta\ndata: {"session_id":"s1"}',
      'event: token\ndata: {"token":"Hello"}',
      'event: citations\ndata: [{"knowledge_base_id":"kb","knowledge_base":"Guide","page_number":1,"excerpt":"Text","score":0.9}]',
      'event: done\ndata: {}',
    ].join('\n\n') + '\n\n';
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(events, {status: 200})));
    let session = '';
    let text = '';
    let citationCount = 0;
    await streamAnswer({message:'Hi', sessionId:'s1', knowledgeBaseIds:['kb'], history:[], signal:new AbortController().signal, onMeta:id => session = id, onToken:token => text += token, onCitations:items => citationCount = items.length});
    expect({session, text, citationCount}).toEqual({session:'s1', text:'Hello', citationCount:1});
  });

  it('starts and ends persisted chat sessions', async () => {
    const session = {id:'s1', company_id:'company', status:'active', ip_address:'127.0.0.1', created_at:'2026-09-17T10:00:00Z', message_count:1};
    const context = {session, company:{id:'company', name:'Acme'}, messages:[], expires_at:'2026-09-17T10:10:00Z'};
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(context), {status:201}))
      .mockResolvedValueOnce(new Response(JSON.stringify(context), {status:200}))
      .mockResolvedValueOnce(new Response(JSON.stringify({...session, status:'closed'}), {status:200}))
      .mockResolvedValueOnce(new Response('{}', {status:404}));
    vi.stubGlobal('fetch', fetchMock);
    expect(await startChatSession('company', 'old-session')).toEqual(context);
    expect(await getChatSession('s1')).toEqual(context);
    await expect(endChatSession('s1', 'closed')).resolves.toBeUndefined();
    await expect(endChatSession('missing', 'timeout')).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenNthCalledWith(1, expect.stringContaining('/chat-sessions'), expect.objectContaining({method:'POST'}));
    expect(fetchMock).toHaveBeenNthCalledWith(3, expect.stringContaining('/chat-sessions/s1'), expect.objectContaining({method:'PATCH'}));
  });

  it('uses a beacon to close a session while the page is unloading', () => {
    const sendBeacon = vi.fn(() => true);
    vi.stubGlobal('navigator', {sendBeacon});
    expect(closeChatSessionOnUnload('session/a')).toBe(true);
    expect(sendBeacon).toHaveBeenCalledWith(expect.stringContaining('/chat-sessions/session%2Fa/close'));
  });

  it('surfaces failed requests', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', {status: 500})));
    await expect(listKnowledgeBases('company')).rejects.toThrow('temporarily unavailable');
    await expect(requestHuman('session')).rejects.toThrow('Could not create');
    await expect(startChatSession('company', 'session')).rejects.toThrow('Could not start');
    await expect(getChatSession('session')).rejects.toThrow('Could not load');
    await expect(endChatSession('session', 'closed')).rejects.toThrow('Could not close');
  });
});
