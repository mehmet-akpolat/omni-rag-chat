import type { ChatSessionContext, ChatSessionStatus, Citation, Company, KnowledgeBase, Message } from './types';
const BASE = import.meta.env.VITE_CHAT_API_URL || 'http://localhost:8002/api/v1';

export function companyLogoUrl(sessionId: string): string { return `${BASE}/chat-sessions/${encodeURIComponent(sessionId)}/company-logo`; }

export async function listCompanies(): Promise<Company[]> {
  const response = await fetch(`${BASE}/companies`); if (!response.ok) throw new Error('Company profiles are temporarily unavailable.'); return response.json();
}

export async function listKnowledgeBases(sessionId: string): Promise<KnowledgeBase[]> {
  const response = await fetch(`${BASE}/knowledge-bases?session_id=${encodeURIComponent(sessionId)}`); if (!response.ok) throw new Error('Knowledge sources are temporarily unavailable.'); return response.json();
}

export async function getChatSession(sessionId: string): Promise<ChatSessionContext> {
  const response = await fetch(`${BASE}/chat-sessions/${encodeURIComponent(sessionId)}`);
  if (!response.ok) throw new Error(response.status === 404 ? 'This chat session does not exist.' : 'Could not load this chat session.');
  return response.json();
}

export async function startChatSession(companyId: string, previousSessionId?: string): Promise<ChatSessionContext> {
  const response = await fetch(`${BASE}/chat-sessions`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({company_id:companyId, previous_session_id:previousSessionId || null})});
  if (!response.ok) throw new Error('Could not start this chat session.');
  return response.json();
}

export async function endChatSession(sessionId: string, status: Exclude<ChatSessionStatus, 'active'>): Promise<void> {
  const response = await fetch(`${BASE}/chat-sessions/${encodeURIComponent(sessionId)}`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({status})});
  if (!response.ok && response.status !== 404) throw new Error('Could not close this chat session.');
}

export function closeChatSessionOnUnload(sessionId: string): boolean {
  return navigator.sendBeacon(`${BASE}/chat-sessions/${encodeURIComponent(sessionId)}/close`);
}

export async function streamAnswer(args: {message: string; sessionId: string; knowledgeBaseIds: string[]; history: Message[]; signal: AbortSignal; onMeta(id: string): void; onToken(token: string): void; onCitations(items: Citation[]): void}) {
  const response = await fetch(`${BASE}/chat/stream`, {method: 'POST', headers: {'Content-Type':'application/json'}, signal: args.signal, body: JSON.stringify({message: args.message, session_id: args.sessionId, knowledge_base_ids: args.knowledgeBaseIds, history: args.history.slice(-12).map(({role, content}) => ({role, content}))})});
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || 'I could not start a response. Please try again.'); }
  if (!response.body) throw new Error('I could not start a response. Please try again.');
  const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = '';
  while (true) {
    const {done, value} = await reader.read(); if (done) break; buffer += decoder.decode(value, {stream:true});
    const events = buffer.split('\n\n'); buffer = events.pop() || '';
    for (const event of events) {
      const type = event.match(/^event: (.+)$/m)?.[1]; const raw = event.match(/^data: (.+)$/m)?.[1]; if (!raw) continue;
      const data = JSON.parse(raw); if (type === 'meta') args.onMeta(data.session_id); if (type === 'token') args.onToken(data.token); if (type === 'citations') args.onCitations(data);
    }
  }
}

export async function requestHuman(sessionId: string): Promise<{ticket_id: string}> {
  const response = await fetch(`${BASE}/escalations`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({session_id:sessionId})});
  if (!response.ok) throw new Error('Could not create a support request.'); return response.json();
}
