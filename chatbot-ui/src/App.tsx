import { KeyboardEvent, useEffect, useRef, useState } from 'react';
import { ArrowUp, BookOpenCheck, Bot, BrainCircuit, Building2, Check, ChevronDown, Clock3, Copy, FileText, Headphones, Headset, Library, Moon, RefreshCw, Square, SquarePen, Sun, ThumbsDown, ThumbsUp, UserRound, X } from 'lucide-react';
import { closeChatSessionOnUnload, companyLogoUrl, endChatSession, getChatSession, listKnowledgeBases, requestHuman, startChatSession, streamAnswer } from './api';
import type { ChatSessionContext, ChatSessionMessage, Citation, Company, KnowledgeBase, Message } from './types';

const makeId = () => globalThis.crypto?.randomUUID?.() || String(Date.now() + Math.random());
const MESSAGE_LIMIT = 400;
const IDLE_SECONDS = positiveNumber(import.meta.env.VITE_CHAT_IDLE_SECONDS, 60);
const KEEP_SESSION_SECONDS = 10;
function avatarIcon(avatar?: Company['bot_avatar'], size = 19) { if (avatar === 'brain') return <BrainCircuit size={size}/>; if (avatar === 'headset') return <Headset size={size}/>; if (avatar === 'book') return <BookOpenCheck size={size}/>; return <Bot size={size}/>; }
function avatarTone(avatar?: Company['bot_avatar']) { return `avatar-tone avatar-tone-${avatar || 'bot'}`; }
export function positiveNumber(value: string | undefined, fallback: number): number { const parsed = Number(value); return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback; }
export function formatMessageTime(value: string | undefined): string { if (!value) return ''; return new Intl.DateTimeFormat(undefined, {hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false}).format(new Date(value)); }
export function formatSessionTime(totalSeconds: number): string { const safe = Math.max(0, Math.ceil(totalSeconds)); return `${String(Math.floor(safe / 60)).padStart(2, '0')}:${String(safe % 60).padStart(2, '0')}`; }
export function messagesForModel(messages: Message[]): Message[] { return messages.filter(message => message.kind !== 'greeting'); }
export function sessionIdFromSearch(search: string): string { return new URLSearchParams(search).get('session_id')?.trim() || ''; }
export function persistedMessages(messages: ChatSessionMessage[]): Message[] {
  return messages.map((message, index) => ({
    id: message.id,
    role: message.owner === 'bot' ? 'assistant' : 'user',
    content: message.content,
    created_at: message.created_at,
    kind: index === 0 && message.owner === 'bot' ? 'greeting' : undefined,
  }));
}
export function messageSegments(content: string): {text: string; href?: string}[] {
  const segments: {text: string; href?: string}[] = [];
  const urlPattern = /\bhttps?:\/\/[^\s<>"']+/gi;
  let cursor = 0;
  const appendText = (text: string) => {
    if (!text) return;
    const previous = segments.at(-1);
    if (previous && !previous.href) previous.text += text;
    else segments.push({text});
  };
  for (const match of content.matchAll(urlPattern)) {
    const start = match.index ?? cursor;
    if (start > cursor) appendText(content.slice(cursor, start));
    const rawUrl = match[0];
    const href = rawUrl.replace(/[),.!?;:]+$/, '');
    const trailingText = rawUrl.slice(href.length);
    segments.push({text: href, href});
    appendText(trailingText);
    cursor = start + rawUrl.length;
  }
  if (cursor < content.length) appendText(content.slice(cursor));
  return segments.length ? segments : [{text: content}];
}
type CompanyLoadState = 'loading' | 'ready' | 'missing' | 'invalid' | 'error';

export function App() {
  const [dark, setDark] = useState(() => localStorage.getItem('omni-theme') === 'dark');
  const [open, setOpen] = useState(true);
  const [requestedSessionId] = useState(() => sessionIdFromSearch(window.location.search));
  const [sources, setSources] = useState<KnowledgeBase[]>([]);
  const [company, setCompany] = useState<Company>();
  const [companyState, setCompanyState] = useState<CompanyLoadState>(() => requestedSessionId ? 'loading' : 'missing');
  const [selected, setSelected] = useState<string[]>([]);
  const [sourceOpen, setSourceOpen] = useState(false);
  const initialNow = useRef(Date.now()).current;
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState(requestedSessionId);
  const [sessionReady, setSessionReady] = useState(false);
  const [sessionStartError, setSessionStartError] = useState('');
  const [sessionLoadAttempt, setSessionLoadAttempt] = useState(0);
  const [sessionExpiresAt, setSessionExpiresAt] = useState(0);
  const [lastActivityAt, setLastActivityAt] = useState(initialNow);
  const [clockNow, setClockNow] = useState(initialNow);
  const [idleWarningDeadline, setIdleWarningDeadline] = useState<number | null>(null);
  const [sessionEnded, setSessionEnded] = useState(false);
  const [input, setInput] = useState('');
  const [generating, setGenerating] = useState(false);
  const [ticket, setTicket] = useState('');
  const [toast, setToast] = useState('');
  const aborter = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const sourcePicker = useRef<HTMLDivElement>(null);
  const sessionSecondsRemaining = Math.max(0, Math.ceil((sessionExpiresAt - clockNow) / 1000));
  const keepSessionSecondsRemaining = idleWarningDeadline === null ? 0 : Math.max(0, Math.ceil((idleWarningDeadline - clockNow) / 1000));

  function applySession(context: ChatSessionContext) {
    const restored = persistedMessages(context.messages);
    const lastUserMessage = [...restored].reverse().find(message => message.role === 'user');
    const now = Date.now();
    setSessionId(context.session.id);
    setCompany(context.company);
    setMessages(restored);
    setSessionExpiresAt(new Date(context.expires_at).getTime());
    setLastActivityAt(lastUserMessage?.created_at ? new Date(lastUserMessage.created_at).getTime() : now);
    setClockNow(now);
    setSessionEnded(context.session.status !== 'active');
    setSessionReady(context.session.status === 'active');
    setCompanyState('ready');
    setSessionStartError('');
  }

  useEffect(() => {
    if (!requestedSessionId) return;
    let cancelled = false;
    const load = async () => {
      setSessionStartError('');
      try {
        const context = await getChatSession(requestedSessionId);
        if (!cancelled) applySession(context);
      } catch (error) {
        if (!cancelled) {
          setSessionReady(false);
          setCompanyState((error as Error).message.includes('does not exist') ? 'invalid' : 'error');
          setSessionStartError((error as Error).message);
        }
      }
    };
    setSessionReady(false);
    void load();
    return () => { cancelled = true; };
  }, [requestedSessionId, sessionLoadAttempt]);
  useEffect(() => { if (!sessionReady || !sessionId) {setSources([]); setSelected([]); return;} listKnowledgeBases(sessionId).then(items => {setSources(items); setSelected(items.map(i => i.id));}).catch(() => undefined); }, [sessionId, sessionReady]);
  useEffect(() => { localStorage.setItem('omni-theme', dark ? 'dark' : 'light'); }, [dark]);
  useEffect(() => { end.current?.scrollIntoView({behavior:'smooth'}); }, [messages]);
  useEffect(() => {
    if (!sessionId || !sessionReady || sessionEnded) return;
    const closeOnPageExit = (event: PageTransitionEvent) => {
      if (!event.persisted) closeChatSessionOnUnload(sessionId);
    };
    window.addEventListener('pagehide', closeOnPageExit);
    return () => window.removeEventListener('pagehide', closeOnPageExit);
  }, [sessionEnded, sessionId, sessionReady]);
  useEffect(() => {
    if (!sourceOpen) return;
    const dismiss = (event: PointerEvent) => {
      if (!sourcePicker.current?.contains(event.target as Node)) setSourceOpen(false);
    };
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') setSourceOpen(false);
    };
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [sourceOpen]);
  useEffect(() => {
    if (companyState !== 'ready' || sessionEnded || !sessionReady) return;
    const timer = window.setInterval(() => setClockNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [companyState, sessionEnded, sessionReady]);
  useEffect(() => {
    if (companyState !== 'ready' || sessionEnded || !sessionReady) return;
    if (clockNow >= sessionExpiresAt) { endSession(); return; }
    if (idleWarningDeadline !== null) {
      if (clockNow >= idleWarningDeadline) endSession();
      return;
    }
    if (clockNow >= lastActivityAt + IDLE_SECONDS * 1000) setIdleWarningDeadline(clockNow + KEEP_SESSION_SECONDS * 1000);
  }, [clockNow, companyState, idleWarningDeadline, lastActivityAt, sessionEnded, sessionExpiresAt, sessionReady]);
  async function send(text = input) {
    const clean = text.trim(); if (!clean || generating || sessionEnded || !sessionReady) return;
    if (clean.length > MESSAGE_LIMIT) { setToast(`Messages are limited to ${MESSAGE_LIMIT} characters.`); setTimeout(() => setToast(''), 2500); return; }
    const sentAt = Date.now(); const createdAt = new Date(sentAt).toISOString();
    setLastActivityAt(sentAt); setClockNow(sentAt); setIdleWarningDeadline(null);
    const user: Message = {id:makeId(), role:'user', content:clean, created_at:createdAt}; const assistantId = makeId();
    const history = [...messages, user]; setMessages([...history, {id:assistantId, role:'assistant', content:'', created_at:createdAt, state:'streaming'}]); setInput(''); setGenerating(true); setTicket('');
    const controller = new AbortController(); aborter.current = controller;
    try {
      await streamAnswer({message:clean, sessionId, knowledgeBaseIds:selected, history:messagesForModel(messages), signal:controller.signal, onMeta:setSessionId, onToken:token => setMessages(current => current.map(m => m.id === assistantId ? {...m, content:m.content + token} : m)), onCitations:(citations:Citation[]) => setMessages(current => current.map(m => m.id === assistantId ? {...m, citations} : m))});
      setMessages(current => current.map(m => m.id === assistantId ? {...m, state:undefined} : m));
    } catch (error) {
      if ((error as Error).name === 'AbortError') setMessages(current => current.map(m => m.id === assistantId ? {...m, content:m.content || 'Response stopped.', state:undefined} : m));
      else {
        if ((error as Error).message.includes('not active')) { setSessionEnded(true); setSessionReady(false); }
        setMessages(current => current.map(m => m.id === assistantId ? {...m, content:'I hit a snag while answering. You can try again, rephrase your question, or ask a human for help.', state:'error'} : m));
      }
    } finally { setGenerating(false); aborter.current = null; }
  }

  function stop() { aborter.current?.abort(); }
  function endSession() { aborter.current?.abort(); setGenerating(false); setSessionEnded(true); setSessionReady(false); setIdleWarningDeadline(null); setSourceOpen(false); void endChatSession(sessionId, 'timeout').catch(error => setToast((error as Error).message)); }
  function keepSession() { const now = Date.now(); setLastActivityAt(now); setClockNow(now); setIdleWarningDeadline(null); }
  async function newChat() { if (!company) return; aborter.current?.abort(); setSessionReady(false); setSessionStartError(''); setIdleWarningDeadline(null); setTicket(''); setSourceOpen(false); try { const context = await startChatSession(company.id, sessionId); applySession(context); const url = new URL(window.location.href); url.search = new URLSearchParams({session_id:context.session.id}).toString(); window.history.replaceState(window.history.state, '', url); } catch (error) { setSessionStartError((error as Error).message); } }
  function closeChat() { aborter.current?.abort(); setGenerating(false); setSessionEnded(true); setSessionReady(false); setOpen(false); void endChatSession(sessionId, 'closed').catch(error => setToast((error as Error).message)); }
  async function escalate() { if (sessionEnded) return; try { const result = await requestHuman(sessionId); setTicket(result.ticket_id); } catch (e) { setToast((e as Error).message); setTimeout(() => setToast(''), 2500); } }
  function keyDown(event: KeyboardEvent<HTMLTextAreaElement>) { if (event.key === 'Enter' && !event.shiftKey) {event.preventDefault(); send();} }
  function copy(text: string) { navigator.clipboard.writeText(text); setToast('Copied to clipboard'); setTimeout(() => setToast(''), 1800); }

  if (companyState !== 'ready') {
    const title = companyState === 'loading' ? 'Loading chat session…' : companyState === 'missing' ? 'Session link required' : companyState === 'invalid' ? 'Session not found' : 'Chat is temporarily unavailable';
    const detail = companyState === 'loading' ? 'Loading the session and its company assistant.' : companyState === 'missing' ? 'Open the chatbot with a session-specific URL containing ?session_id=<session-id>.' : companyState === 'invalid' ? 'The session_id in this URL does not match an available chat session.' : sessionStartError || 'The chat session could not be loaded. Please try again shortly.';
    return <div className={dark ? 'shell dark' : 'shell'}><main className="chat-main"><section className="company-launch-state"><div><Building2 size={26}/></div><h1>{title}</h1><p>{detail}</p>{companyState === 'error' && <button onClick={() => { setCompanyState('loading'); setSessionLoadAttempt(current => current + 1); }}><RefreshCw size={16}/>Try again</button>}</section></main></div>;
  }

  if (!open) return <div className={dark ? 'shell dark' : 'shell'}><button className={`launcher ${avatarTone(company?.bot_avatar)}`} onClick={() => setOpen(true)} aria-label="Open assistant">{avatarIcon(company?.bot_avatar, 24)}<span>Ask {company?.bot_alias || 'AIBot'}</span></button></div>;

  return <div className={dark ? 'shell dark' : 'shell'}>
    <main className="chat-main">
      <header><div className="head-left company-brand"><div className="company-brand-logo">{company?.has_logo ? <img src={companyLogoUrl(sessionId)} alt={`${company.name} logo`}/> : <Building2 size={21}/>}</div><div className="company-brand-copy"><b>{company?.name || 'Company'}</b><span>{company?.bot_alias || 'Knowledge assistant'} · AI assistant</span></div></div><div className="head-actions"><div className={`session-timer ${sessionSecondsRemaining <= 60 ? 'urgent' : ''}`} title="Session time remaining"><Clock3 size={15}/><span>{formatSessionTime(sessionSecondsRemaining)}</span></div><button className="top-new-chat" onClick={newChat} aria-label="New conversation" title="New conversation"><SquarePen size={17}/></button><button className="plain" onClick={escalate} aria-label="Talk to a human" title="Talk to a human"><Headphones size={18}/></button><div className="source-picker" ref={sourcePicker}><button onClick={() => setSourceOpen(current => !current)} aria-expanded={sourceOpen} aria-haspopup="listbox"><Library size={16}/>{selected.length ? `${selected.length} source${selected.length === 1 ? '' : 's'}` : 'Company profile'}<ChevronDown size={14}/></button>{sourceOpen && <div className="source-dropdown" role="listbox" aria-label="Knowledge sources" aria-multiselectable="true">{sources.length ? sources.map(source => <label className={selected.includes(source.id) ? 'selected' : ''} key={source.id}><input type="checkbox" checked={selected.includes(source.id)} onChange={() => setSelected(current => current.includes(source.id) ? current.filter(id => id !== source.id) : [...current, source.id])}/><span>{source.name}</span>{selected.includes(source.id) && <Check size={14}/>}</label>) : <p>No knowledge bases available</p>}</div>}</div><button className="plain" onClick={() => setDark(!dark)} aria-label={dark ? 'Use light theme' : 'Use dark theme'}>{dark ? <Sun size={18}/> : <Moon size={18}/>}</button><button className="plain" onClick={closeChat} aria-label="Close assistant"><X size={19}/></button></div></header>

      <div className="conversation">
        <div className="messages">{messages.map(message => <article className={`message ${message.role}`} key={message.id}>{message.role === 'assistant' && <div className={`assistant-icon ${avatarTone(company?.bot_avatar)}`}>{avatarIcon(company?.bot_avatar, 15)}</div>}<div className="message-body"><div className="message-label"><span>{message.role === 'assistant' ? company?.bot_alias || 'AIBot' : 'You'}</span><time dateTime={message.created_at}>{formatMessageTime(message.created_at)}</time></div><div className={`bubble ${message.state || ''}`}>{message.content ? messageSegments(message.content).map((segment, index) => segment.href ? <a key={`${segment.href}-${index}`} href={segment.href} target="_blank" rel="noopener noreferrer">{segment.text}</a> : <span key={index}>{segment.text}</span>) : <span className="thinking"><i/><i/><i/></span>}</div>{message.citations && message.citations.length > 0 && <div className="citations"><b><FileText size={14}/>Sources</b><div>{message.citations.map((citation, index) => <details key={`${citation.knowledge_base_id}-${citation.page_number}-${citation.source_number || index}`}><summary><span>{citation.source_number || index + 1}</span>{citation.source_url ? <a href={citation.source_url} target="_blank" rel="noopener noreferrer" onClick={event => event.stopPropagation()}>{citation.knowledge_base}</a> : citation.knowledge_base}{citation.page_number > 0 ? ` · page ${citation.page_number}` : citation.section ? ` · ${citation.section}` : ''}<ChevronDown size={13}/></summary><p>{citation.excerpt}</p></details>)}</div></div>}{message.role === 'assistant' && message.kind !== 'greeting' && !message.state && <div className="message-tools"><button onClick={() => copy(message.content)} title="Copy"><Copy size={14}/></button><button title="Helpful"><ThumbsUp size={14}/></button><button title="Not helpful"><ThumbsDown size={14}/></button></div>}{message.state === 'error' && <button className="retry" onClick={() => send(messages.filter(m => m.role === 'user').at(-1)?.content)}><RefreshCw size={14}/>Try again</button>}</div>{message.role === 'user' && <div className="user-icon"><UserRound size={15}/></div>}</article>)}<div ref={end}/></div>
      </div>

      <footer><div className="composer"><textarea aria-label="Message" rows={1} maxLength={MESSAGE_LIMIT} value={input} onChange={e => setInput(e.target.value)} onKeyDown={keyDown} disabled={sessionEnded || !sessionReady} placeholder={sessionEnded ? 'Start a new conversation to continue' : !sessionReady ? 'Starting chat session…' : 'Ask about your documents…'}/><div className="composer-bottom"><span>{selected.length ? `Searching company profile + ${selected.length} selected source${selected.length === 1 ? '' : 's'}` : 'Searching company profile'}</span><div className="composer-actions"><span className="char-counter">{input.length}/{MESSAGE_LIMIT}</span>{generating ? <button className="send stop" onClick={stop}><Square size={13}/></button> : <button className="send" onClick={() => send()} disabled={!input.trim() || !sessionId || sessionEnded || !sessionReady}><ArrowUp size={18}/></button>}</div></div></div><p>AI can make mistakes. Check cited sources for important information.</p></footer>
    </main>
    {idleWarningDeadline !== null && !sessionEnded && <div className="session-dialog-backdrop"><section className="session-dialog" role="alertdialog" aria-modal="true" aria-labelledby="keep-session-title"><div className="session-dialog-icon"><Clock3 size={22}/></div><h2 id="keep-session-title">Keep this session?</h2><p>Your inactive session will end in <b>{keepSessionSecondsRemaining} second{keepSessionSecondsRemaining === 1 ? '' : 's'}</b>.</p><button onClick={keepSession}>Keep session</button></section></div>}{sessionEnded && <div className="session-dialog-backdrop"><section className="session-dialog" role="alertdialog" aria-modal="true" aria-labelledby="session-ended-title"><div className="session-dialog-icon"><Clock3 size={22}/></div><h2 id="session-ended-title">Session ended</h2><p>Start a new conversation to continue chatting with {company?.bot_alias || 'the assistant'}.</p><button onClick={newChat}><SquarePen size={16}/>New conversation</button></section></div>}{sessionStartError && !sessionEnded && <div className="session-dialog-backdrop"><section className="session-dialog" role="alertdialog" aria-modal="true" aria-labelledby="session-start-error-title"><div className="session-dialog-icon"><Clock3 size={22}/></div><h2 id="session-start-error-title">Could not start the session</h2><p>{sessionStartError}</p><button onClick={newChat}><RefreshCw size={16}/>Try again</button></section></div>}{ticket && <div className="ticket"><div><Check size={18}/></div><span><b>Human support requested</b>Your reference is {ticket}. An agent can use this conversation’s context.</span><button onClick={() => setTicket('')}><X size={15}/></button></div>}{toast && <div className="toast">{toast}</div>}
  </div>;
}
