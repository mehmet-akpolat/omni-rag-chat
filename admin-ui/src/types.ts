export type Strategy = 'fixed' | 'recursive' | 'semantic' | 'hierarchical';
export interface PagePreview { page_number: number; excerpt: string }
export type KnowledgeSourceType = 'pdf' | 'url';
export interface Preview { document_id: string; source: string; mime_type: string; total_pages: number; pages: PagePreview[]; title?: string; replaces_knowledge_base_id?: string }
export type BotAvatar = 'bot' | 'brain' | 'headset' | 'book';
export type LLMProvider = 'openai' | 'anthropic' | 'huggingface' | 'ollama';
export interface LLMSelection { provider: LLMProvider; model: string; api_key?: string; has_api_key?: boolean; api_key_masked?: string }
export interface LLMOption { provider: LLMProvider; models: string[] }
export interface Company { id: string; name: string; has_logo: boolean; logo_mime_type?: string; about: string; phone: string; email: string; address: string; maps_url?: string; bot_alias: string; bot_avatar: BotAvatar; bot_greet_message: string; llm?: LLMSelection | null; created_at: string; updated_at: string }
export interface KnowledgeBase { id: string; company_id: string; name: string; source: string; mime_type: string; total_pages: number | null; excluded_pages: number[] | null; selected_pages: number | null; chunk_count: number; status: 'enabled' | 'disabled' | 'indexing'; created_at: string; updated_at: string; chunking: { strategy: Strategy; chunk_size: number; overlap: number; similarity_threshold: number; parent_size: number } }
export interface KnowledgeBasePage { items: KnowledgeBase[]; page: number; page_size: number; total: number; total_pages: number }
export type ChatSessionStatus = 'active' | 'closed' | 'timeout';
export interface ChatSession { id: string; company_id: string; status: ChatSessionStatus; ip_address: string; created_at: string; ended_at?: string; message_count: number }
export interface ChatSessionMessage { id: string; session_id: string; owner: 'bot' | 'user'; content: string; input_tokens: number; output_tokens: number; created_at: string }
export interface ChatSessionPage { items: ChatSession[]; page: number; page_size: number; total: number; total_pages: number }
export interface ChatSessionDetail { session: ChatSession; messages: ChatSessionMessage[] }
export interface AnalyticsDailyPoint { date: string; session_count: number; message_count: number; input_tokens: number; output_tokens: number }
export interface ChatAnalytics { company_id: string; date_from: string; date_to: string; session_count: number; message_count: number; average_messages_per_session: number; input_tokens: number; output_tokens: number; daily: AnalyticsDailyPoint[] }
