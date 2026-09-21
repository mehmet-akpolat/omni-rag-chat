import type { ChatSessionDetail, ChatSessionPage, Company, KnowledgeBase, KnowledgeBasePage, LLMOption, Preview, Strategy } from './types';
const BASE = import.meta.env.VITE_ADMIN_API_URL || 'http://localhost:8001/api/v1';
const CHAT_BASE = import.meta.env.VITE_CHAT_API_URL || 'http://localhost:8002/api/v1';

async function checked<T>(response: Response): Promise<T> {
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || 'Something went wrong'); }
  return response.json();
}

export async function previewPdf(file: File, companyId: string, signal?: AbortSignal): Promise<Preview> {
  const data = new FormData(); data.append('file', file); data.append('company_id', companyId);
  return checked(await fetch(`${BASE}/documents/preview`, { method: 'POST', body: data, signal }));
}
export async function previewUrl(url: string, companyId: string, signal?: AbortSignal): Promise<Preview> {
  return checked(await fetch(`${BASE}/urls/preview`, {method:'POST', headers:{'Content-Type':'application/json'}, signal, body:JSON.stringify({url, company_id:companyId})}));
}
export async function importKnowledgeBase(payload: {document_id: string; company_id: string; name: string; pages: {included_pages: number[]; excluded_pages: number[]}; chunking: {strategy: Strategy; chunk_size: number; overlap: number; similarity_threshold: number; parent_size: number}}): Promise<KnowledgeBase> {
  return checked(await fetch(`${BASE}/knowledge-bases`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) }));
}
export async function listKnowledgeBases(page = 1, pageSize = 10, search = '', companyId = ''): Promise<KnowledgeBasePage> {
  const query = new URLSearchParams({page: String(page), page_size: String(pageSize)});
  if (search.trim()) query.set('search', search.trim());
  if (companyId) query.set('company_id', companyId);
  return checked(await fetch(`${BASE}/knowledge-bases?${query}`));
}

export async function listCompanies(): Promise<Company[]> { return checked(await fetch(`${BASE}/companies`)); }
export async function listLLMOptions(): Promise<LLMOption[]> { return checked(await fetch(`${BASE}/llm-options`)); }
export async function createChatSession(companyId: string): Promise<{session: ChatSessionDetail['session']}> {
  return checked(await fetch(`${CHAT_BASE}/chat-sessions`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({company_id:companyId})}));
}
export async function listChatSessions(companyId: string, dateFrom: string, dateTo: string, page = 1, pageSize = 20): Promise<ChatSessionPage> {
  const query = new URLSearchParams({company_id:companyId, date_from:dateFrom, date_to:dateTo, page:String(page), page_size:String(pageSize)});
  return checked(await fetch(`${BASE}/chat-sessions?${query}`));
}
export async function getChatSession(id: string): Promise<ChatSessionDetail> { return checked(await fetch(`${BASE}/chat-sessions/${encodeURIComponent(id)}`)); }
export async function deleteChatSessions(companyId: string, sessionIds: string[]): Promise<{deleted: number}> {
  return checked(await fetch(`${BASE}/chat-sessions`, {method:'DELETE', headers:{'Content-Type':'application/json'}, body:JSON.stringify({company_id:companyId, session_ids:sessionIds})}));
}
export async function deleteFilteredChatSessions(companyId: string, dateFrom: string, dateTo: string): Promise<{deleted: number}> {
  return checked(await fetch(`${BASE}/chat-sessions`, {method:'DELETE', headers:{'Content-Type':'application/json'}, body:JSON.stringify({company_id:companyId, date_from:dateFrom, date_to:dateTo})}));
}
export function companyLogoUrl(id: string): string { return `${BASE}/companies/${id}/logo`; }
export async function saveCompany(company: Partial<Company> & Pick<Company, 'name'>): Promise<Company> {
  const existing = Boolean(company.id);
  return checked(await fetch(`${BASE}/companies${existing ? `/${company.id}` : ''}`, {method: existing ? 'PUT' : 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(company)}));
}
export async function saveCompanyLogo(id: string, file: File): Promise<Company> {
  const data = new FormData(); data.append('file', file);
  return checked(await fetch(`${BASE}/companies/${id}/logo`, {method:'PUT', body:data}));
}
export async function deleteCompanyLogo(id: string): Promise<void> {
  const response = await fetch(`${BASE}/companies/${id}/logo`, {method:'DELETE'});
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || 'Could not remove this logo'); }
}
export async function deleteCompany(id: string): Promise<void> {
  const response = await fetch(`${BASE}/companies/${id}`, {method:'DELETE'});
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || 'Could not delete this company'); }
}

export async function setKnowledgeBaseEnabled(id: string, enabled: boolean): Promise<KnowledgeBase> {
  return checked(await fetch(`${BASE}/knowledge-bases/${id}`, {
    method: 'PATCH', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled}),
  }));
}

export async function deleteKnowledgeBase(id: string): Promise<void> {
  const response = await fetch(`${BASE}/knowledge-bases/${id}`, {method: 'DELETE'});
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || 'Could not delete this knowledge base');
  }
}
