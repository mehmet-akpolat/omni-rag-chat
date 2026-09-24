export interface Citation { source_number?: number; knowledge_base_id: string; knowledge_base: string; page_number: number; excerpt: string; score: number; section?: string; source_url?: string }
export interface Message { id: string; role: 'user' | 'assistant'; content: string; created_at?: string; citations?: Citation[]; state?: 'streaming' | 'error'; kind?: 'greeting' }
export interface Company { id: string; name: string; has_logo: boolean; logo_mime_type?: string; about: string; phone: string; email: string; address: string; maps_url?: string; bot_alias: string; bot_avatar: 'bot' | 'brain' | 'headset' | 'book'; bot_greet_message: string }
export interface KnowledgeBase { id: string; company_id: string; name: string; source: string; excluded_pages: number[] | null; selected_pages: number | null }
export type ChatSessionStatus = 'active' | 'closed' | 'timeout';
export interface ChatSession { id: string; company_id: string; status: ChatSessionStatus; ip_address: string; created_at: string; ended_at?: string; message_count: number }
export interface ChatSessionMessage { id: string; session_id: string; owner: 'bot' | 'user'; content: string; created_at: string }
export interface ChatSessionContext { session: ChatSession; company: Company; messages: ChatSessionMessage[]; expires_at: string }
