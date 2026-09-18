import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, ArrowLeft, ArrowRight, BookOpen, BookOpenCheck, Bot, BrainCircuit, Building2, CalendarDays, Check, ChevronDown, ChevronUp, CirclePause, CirclePlay, Eye, FileText, Headset, ImagePlus, Layers3, Library, Mail, MapPin, MapPinned, MessageSquareText, Moon, Pencil, Phone, RefreshCw, RotateCcw, Search, Settings2, Sparkles, Sun, Trash2, UploadCloud, UserRound, X } from 'lucide-react';
import { companyLogoUrl, createChatSession, deleteCompany, deleteCompanyLogo, deleteKnowledgeBase, getChatSession, importKnowledgeBase, listChatSessions, listCompanies, listKnowledgeBases, previewPdf, saveCompany, saveCompanyLogo, setKnowledgeBaseEnabled } from './api';
import type { BotAvatar, ChatSession, ChatSessionDetail, Company, KnowledgeBase, Preview, Strategy } from './types';

const strategies: {id: Strategy; name: string; caption: string}[] = [
  {id: 'fixed', name: 'Fixed-size', caption: 'Consistent, predictable windows'},
  {id: 'recursive', name: 'Recursive', caption: 'Preserve natural text structure'},
  {id: 'semantic', name: 'Semantic', caption: 'Group passages by meaning'},
  {id: 'hierarchical', name: 'Hierarchical', caption: 'Parent and child context layers'},
];

type ConfirmationAction = { type: 'disable' | 'delete'; items: KnowledgeBase[] };
const PAGE_SIZE = 10;
const CHAT_SESSION_PAGE_SIZES = [5, 10, 20, 50] as const;
const MAX_CHAT_SESSION_DATE_RANGE_DAYS = 14;
const MESSAGE_TIMEOUT_MS = 5_000;
const CHATBOT_UI_URL = import.meta.env.VITE_CHATBOT_UI_URL || 'http://localhost:5174';
const DEFAULT_BOT_GREET_MESSAGE = "Hello! I'm {{bot_alias}}, the AI assistant for {{company_name}}. How can I help you today?";
const GREETING_PLACEHOLDERS = new Set(['bot_alias', 'company_name']);
const emptyCompany = {name: '', about: '', phone: '', email: '', address: '', maps_url: '', bot_alias: 'AIBot', bot_avatar: 'bot' as BotAvatar, bot_greet_message: DEFAULT_BOT_GREET_MESSAGE};
const avatarOptions: {id: BotAvatar; label: string}[] = [
  {id: 'bot', label: 'General AI'},
  {id: 'brain', label: 'Knowledge expert'},
  {id: 'headset', label: 'Customer support'},
  {id: 'book', label: 'Documentation guide'},
];

const strategyNames: Record<Strategy, string> = {
  fixed: 'Fixed-size',
  recursive: 'Recursive',
  semantic: 'Semantic',
  hierarchical: 'Hierarchical',
};

export function formatCreatedAt(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  }).format(new Date(value));
}

export function dateInputValue(value: Date): string {
  const local = new Date(value.getTime() - value.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 10);
}

export function shiftDateInput(value: string, days: number): string {
  if (!value) return '';
  const shifted = new Date(`${value}T00:00:00Z`);
  shifted.setUTCDate(shifted.getUTCDate() + days);
  return shifted.toISOString().slice(0, 10);
}

export function formatClock(value: string): string {
  return new Intl.DateTimeFormat(undefined, {hour:'2-digit', minute:'2-digit', second:'2-digit'}).format(new Date(value));
}

export function formatSessionDuration(createdAt: string, endedAt?: string): string {
  if (!endedAt) return '-';
  const start = new Date(createdAt).getTime();
  const end = new Date(endedAt).getTime();
  const totalSeconds = Math.max(0, Math.floor((end - start) / 1000));
  if (totalSeconds < 60) return `${totalSeconds} ${totalSeconds === 1 ? 'sec' : 'secs'}`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes} ${minutes === 1 ? 'min' : 'mins'} ${seconds} ${seconds === 1 ? 'sec' : 'secs'}`;
}

export function shouldAutoExpandImport(total: number, query: string): boolean {
  return !query.trim() && total === 0;
}

export function chatbotLaunchUrl(sessionId: string, baseUrl = CHATBOT_UI_URL): string {
  const separator = baseUrl.includes('?') ? '&' : '?';
  return `${baseUrl}${separator}session_id=${encodeURIComponent(sessionId)}`;
}

export function validateGreetingTemplate(template: string): string | undefined {
  let cursor = 0;
  while (cursor < template.length) {
    const opening = template.indexOf('{{', cursor);
    const earlyClosing = template.indexOf('}}', cursor);
    if (earlyClosing >= 0 && (opening < 0 || earlyClosing < opening)) return 'Greeting template contains malformed placeholder syntax';
    if (opening < 0) return undefined;
    const closing = template.indexOf('}}', opening + 2);
    if (closing < 0) return 'Greeting template contains malformed placeholder syntax';
    const keyword = template.slice(opening + 2, closing);
    if (!GREETING_PLACEHOLDERS.has(keyword)) return `Unsupported greeting placeholder "${keyword}". Use only {{bot_alias}} or {{company_name}}.`;
    cursor = closing + 2;
  }
  return undefined;
}

function avatarIcon(avatar: BotAvatar, size = 18) {
  if (avatar === 'brain') return <BrainCircuit size={size}/>;
  if (avatar === 'headset') return <Headset size={size}/>;
  if (avatar === 'book') return <BookOpenCheck size={size}/>;
  return <Bot size={size}/>;
}

function safeExternalUrl(value?: string): string | undefined {
  if (!value?.trim()) return undefined;
  try {
    const parsed = new URL(value);
    return ['http:', 'https:'].includes(parsed.protocol) ? parsed.toString() : undefined;
  } catch { return undefined; }
}

function rangeToPages(value: string, maximum: number): number[] {
  const pages = new Set<number>();
  value.split(',').map(v => v.trim()).filter(Boolean).forEach(part => {
    const [from, to = from] = part.split('-').map(Number);
    if (!Number.isFinite(from) || !Number.isFinite(to)) return;
    for (let page = Math.max(1, from); page <= Math.min(maximum, to); page += 1) pages.add(page);
  });
  return [...pages].sort((a, b) => a - b);
}

export function App() {
  const [dark, setDark] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [strategy, setStrategy] = useState<Strategy>('recursive');
  const [include, setInclude] = useState('');
  const [exclude, setExclude] = useState('');
  const [chunkSize, setChunkSize] = useState(700);
  const [overlap, setOverlap] = useState(100);
  const [parentSize, setParentSize] = useState(1600);
  const [threshold, setThreshold] = useState(.72);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState<KnowledgeBase | null>(null);
  const [items, setItems] = useState<KnowledgeBase[]>([]);
  const [rowBusy, setRowBusy] = useState('');
  const [search, setSearch] = useState('');
  const [confirmation, setConfirmation] = useState<ConfirmationAction | null>(null);
  const [importExpanded, setImportExpanded] = useState(false);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [listLoading, setListLoading] = useState(true);
  const [libraryError, setLibraryError] = useState('');
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [view, setView] = useState<'knowledge' | 'sessions' | 'companies'>('knowledge');
  const [companies, setCompanies] = useState<Company[]>([]);
  const [activeCompanyId, setActiveCompanyId] = useState('');
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [chatSessionPage, setChatSessionPage] = useState(1);
  const [chatSessionPageSize, setChatSessionPageSize] = useState(10);
  const [chatSessionTotal, setChatSessionTotal] = useState(0);
  const [chatSessionTotalPages, setChatSessionTotalPages] = useState(0);
  const [chatSessionsLoading, setChatSessionsLoading] = useState(false);
  const [chatSessionError, setChatSessionError] = useState('');
  const [selectedChatSession, setSelectedChatSession] = useState<ChatSessionDetail | null>(null);
  const [chatSessionFrom, setChatSessionFrom] = useState(() => dateInputValue(new Date(Date.now() - 6 * 86_400_000)));
  const [chatSessionTo, setChatSessionTo] = useState(() => dateInputValue(new Date()));
  const [companyDraft, setCompanyDraft] = useState<Partial<Company> & Pick<Company, 'name'>>(emptyCompany);
  const [companyBusy, setCompanyBusy] = useState(false);
  const [companyError, setCompanyError] = useState('');
  const [companyToDelete, setCompanyToDelete] = useState<Company | null>(null);
  const [companyLogo, setCompanyLogo] = useState<File | null>(null);
  const [companyLogoPreview, setCompanyLogoPreview] = useState('');
  const [removeLogo, setRemoveLogo] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const companyLogoInput = useRef<HTMLInputElement>(null);
  const selectAllInput = useRef<HTMLInputElement>(null);
  const listRequest = useRef(0);
  const chatSessionRequest = useRef(0);
  const importInitializedFor = useRef('');
  const activeCompany = companies.find(company => company.id === activeCompanyId);
  const mapsLink = safeExternalUrl(companyDraft.maps_url);

  const loadKnowledgeBases = useCallback(async (targetPage: number, query: string, companyId: string) => {
    if (!companyId) { setItems([]); setTotal(0); setTotalPages(0); setListLoading(false); return; }
    const requestId = ++listRequest.current;
    setListLoading(true);
    setLibraryError('');
    try {
      const result = await listKnowledgeBases(targetPage, PAGE_SIZE, query, companyId);
      if (requestId !== listRequest.current) return;
      setItems(result.items);
      setSelectedIds(new Set());
      setTotal(result.total);
      setTotalPages(result.total_pages);
      if (targetPage === 1 && importInitializedFor.current !== companyId && !query.trim()) {
        setImportExpanded(shouldAutoExpandImport(result.total, query));
        importInitializedFor.current = companyId;
      }
    } catch (err) {
      if (requestId !== listRequest.current) return;
      setLibraryError(err instanceof Error ? err.message : 'Could not load knowledge bases');
    } finally {
      if (requestId === listRequest.current) setListLoading(false);
    }
  }, []);

  const loadChatSessionPage = useCallback(async (targetPage: number, companyId: string, dateFrom: string, dateTo: string) => {
    if (!companyId) { setChatSessions([]); setChatSessionTotal(0); setChatSessionTotalPages(0); setChatSessionsLoading(false); return; }
    const requestId = ++chatSessionRequest.current;
    setChatSessionsLoading(true); setChatSessionError('');
    try {
      const result = await listChatSessions(companyId, dateFrom, dateTo, targetPage, chatSessionPageSize);
      if (requestId !== chatSessionRequest.current) return;
      setChatSessions(result.items); setChatSessionTotal(result.total); setChatSessionTotalPages(result.total_pages);
    } catch (err) {
      if (requestId !== chatSessionRequest.current) return;
      setChatSessionError(err instanceof Error ? err.message : 'Could not load chat sessions');
    } finally {
      if (requestId === chatSessionRequest.current) setChatSessionsLoading(false);
    }
  }, [chatSessionPageSize]);

  useEffect(() => {
    listCompanies().then(result => {
      setCompanies(result);
    }).catch(err => setCompanyError(err instanceof Error ? err.message : 'Could not load companies'));
  }, []);
  useEffect(() => {
    importInitializedFor.current = '';
    setImportExpanded(false);
    setChatSessionPage(1);
    setSelectedChatSession(null);
  }, [activeCompanyId]);
  useEffect(() => {
    const timeout = window.setTimeout(() => loadKnowledgeBases(page, search, activeCompanyId), 250);
    return () => window.clearTimeout(timeout);
  }, [activeCompanyId, loadKnowledgeBases, page, search]);
  useEffect(() => {
    if (view !== 'sessions') return;
    void loadChatSessionPage(chatSessionPage, activeCompanyId, chatSessionFrom, chatSessionTo);
  }, [activeCompanyId, chatSessionFrom, chatSessionPage, chatSessionTo, loadChatSessionPage, view]);
  useEffect(() => {
    if (!error) return;
    const timeout = window.setTimeout(() => setError(''), MESSAGE_TIMEOUT_MS);
    return () => window.clearTimeout(timeout);
  }, [error]);
  useEffect(() => {
    if (!success) return;
    const timeout = window.setTimeout(() => setSuccess(null), MESSAGE_TIMEOUT_MS);
    return () => window.clearTimeout(timeout);
  }, [success]);
  useEffect(() => {
    if (!libraryError) return;
    const timeout = window.setTimeout(() => setLibraryError(''), MESSAGE_TIMEOUT_MS);
    return () => window.clearTimeout(timeout);
  }, [libraryError]);
  useEffect(() => {
    if (!companyError) return;
    const timeout = window.setTimeout(() => setCompanyError(''), MESSAGE_TIMEOUT_MS);
    return () => window.clearTimeout(timeout);
  }, [companyError]);
  useEffect(() => {
    if (!chatSessionError) return;
    const timeout = window.setTimeout(() => setChatSessionError(''), MESSAGE_TIMEOUT_MS);
    return () => window.clearTimeout(timeout);
  }, [chatSessionError]);
  useEffect(() => {
    if (!companyLogo) { setCompanyLogoPreview(''); return; }
    const previewUrl = URL.createObjectURL(companyLogo);
    setCompanyLogoPreview(previewUrl);
    return () => URL.revokeObjectURL(previewUrl);
  }, [companyLogo]);
  useEffect(() => {
    if (!confirmation) return;
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape' && !rowBusy) setConfirmation(null);
    };
    document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [confirmation, rowBusy]);
  const selected = useMemo(() => preview ? (include ? rangeToPages(include, preview.total_pages) : Array.from({length: preview.total_pages}, (_, i) => i + 1)).filter(p => !rangeToPages(exclude, preview.total_pages).includes(p)) : [], [preview, include, exclude]);
  const selectedItems = useMemo(() => items.filter(item => selectedIds.has(item.id)), [items, selectedIds]);
  const selectedEnabledItems = useMemo(() => selectedItems.filter(item => item.status === 'enabled'), [selectedItems]);
  const selectedDisabledItems = useMemo(() => selectedItems.filter(item => item.status === 'disabled'), [selectedItems]);
  const allItemsSelected = items.length > 0 && items.every(item => selectedIds.has(item.id));
  useEffect(() => {
    if (selectAllInput.current) {
      selectAllInput.current.indeterminate = selectedItems.length > 0 && !allItemsSelected;
    }
  }, [allItemsSelected, selectedItems.length]);

  async function choose(chosen?: File) {
    if (!chosen) return; setError(''); setBusy(true); setFile(chosen);
    if (!activeCompanyId) { setError('Create or select a company before uploading a document'); setBusy(false); return; }
    try { const result = await previewPdf(chosen, activeCompanyId); setPreview(result); setName(chosen.name.replace(/\.pdf$/i, '')); setSuccess(null); }
    catch (err) { setError(err instanceof Error ? err.message : 'Could not read this file'); setFile(null); }
    finally { setBusy(false); }
  }

  async function ingest() {
    if (!preview || !name.trim()) return; setBusy(true); setError('');
    try {
      const kb = await importKnowledgeBase({document_id: preview.document_id, company_id: activeCompanyId, name: name.trim(), pages: {included_pages: rangeToPages(include, preview.total_pages), excluded_pages: rangeToPages(exclude, preview.total_pages)}, chunking: {strategy, chunk_size: chunkSize, overlap, parent_size: parentSize, similarity_threshold: threshold}});
      setSuccess(kb); reset(true); setImportExpanded(false); setPage(1); await loadKnowledgeBases(1, search, activeCompanyId);
    } catch (err) { setError(err instanceof Error ? err.message : 'Import failed'); }
    finally { setBusy(false); }
  }

  function reset(keepSuccess = false) {
    setPreview(null); setFile(null); setName(''); setStrategy('recursive'); setInclude(''); setExclude('');
    setChunkSize(700); setOverlap(100); setParentSize(1600); setThreshold(.72); setError('');
    if (!keepSuccess) setSuccess(null);
    if (input.current) input.current.value = '';
  }

  function chooseCompanyLogo(chosen?: File) {
    if (!chosen) { setCompanyLogo(null); return; }
    if (!['image/png', 'image/svg+xml', 'image/jpeg'].includes(chosen.type)) {
      setCompanyLogo(null); setCompanyError('Logo must be a PNG, SVG, or JPEG image');
      if (companyLogoInput.current) companyLogoInput.current.value = '';
      return;
    }
    if (chosen.size > 512 * 1024) {
      setCompanyLogo(null); setCompanyError('Company logo exceeds the 512 KB upload limit');
      if (companyLogoInput.current) companyLogoInput.current.value = '';
      return;
    }
    setCompanyError(''); setCompanyLogo(chosen); setRemoveLogo(false);
  }

  function resetCompanyEditor() {
    setCompanyDraft(emptyCompany); setCompanyLogo(null); setRemoveLogo(false);
    if (companyLogoInput.current) companyLogoInput.current.value = '';
  }

  function navigateFromMenu(target: 'knowledge' | 'sessions' | 'companies') {
    setConfirmation(null);
    setCompanyToDelete(null);
    if (target === 'knowledge') {
      reset();
      setSearch('');
      setPage(1);
      setSelectedIds(new Set());
      setLibraryError('');
      setImportExpanded(Boolean(activeCompanyId) && shouldAutoExpandImport(total, ''));
      importInitializedFor.current = '';
    } else if (target === 'sessions') {
      const now = Date.now();
      setSelectedChatSession(null);
      setChatSessionPage(1);
      setChatSessionPageSize(10);
      setChatSessionFrom(dateInputValue(new Date(now - 6 * 86_400_000)));
      setChatSessionTo(dateInputValue(new Date(now)));
      setChatSessionError('');
    } else {
      resetCompanyEditor();
      setCompanyError('');
    }
    setView(target);
    window.scrollTo({top: 0, behavior: 'smooth'});
  }

  function editCompany(company: Company) {
    setCompanyDraft(company); setCompanyLogo(null); setRemoveLogo(false);
    if (companyLogoInput.current) companyLogoInput.current.value = '';
  }

  function toggleImportWorkspace() {
    if (!importExpanded) setSuccess(null);
    importInitializedFor.current = activeCompanyId;
    setImportExpanded(current => !current);
  }

  function toggleSelected(id: string) {
    setSelectedIds(current => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  function toggleAllItems() {
    setSelectedIds(allItemsSelected ? new Set() : new Set(items.map(item => item.id)));
  }

  async function reviewChatSession(session: ChatSession) {
    setChatSessionsLoading(true); setChatSessionError('');
    try { setSelectedChatSession(await getChatSession(session.id)); }
    catch (err) { setChatSessionError(err instanceof Error ? err.message : 'Could not load this chat session'); }
    finally { setChatSessionsLoading(false); }
  }

  function returnToChatSessions() {
    setSelectedChatSession(null);
    void loadChatSessionPage(
      chatSessionPage,
      activeCompanyId,
      chatSessionFrom,
      chatSessionTo,
    );
  }

  async function launchChatbot(companyId: string) {
    const chatWindow = window.open('about:blank', '_blank');
    if (!chatWindow) {
      const message = 'Allow pop-ups for this site to open the chatbot.';
      if (view === 'sessions') setChatSessionError(message); else setLibraryError(message);
      return;
    }
    chatWindow.opener = null;
    try {
      const context = await createChatSession(companyId);
      chatWindow.location.replace(chatbotLaunchUrl(context.session.id));
    } catch (err) {
      chatWindow.close();
      const message = err instanceof Error ? err.message : 'Could not start the chatbot session';
      if (view === 'sessions') setChatSessionError(message); else setLibraryError(message);
    }
  }

  async function enableSelectedKnowledgeBases() {
    if (!selectedDisabledItems.length) return;
    setRowBusy('bulk'); setLibraryError('');
    const results = await Promise.allSettled(
      selectedDisabledItems.map(item => setKnowledgeBaseEnabled(item.id, true))
    );
    const updated = results.flatMap(result => result.status === 'fulfilled' ? [result.value] : []);
    const updatedById = new Map(updated.map(item => [item.id, item]));
    setItems(current => current.map(item => updatedById.get(item.id) || item));
    const failed = results.length - updated.length;
    if (failed) {
      setLibraryError(`${failed} knowledge ${failed === 1 ? 'base' : 'bases'} could not be enabled`);
    } else {
      setSelectedIds(new Set());
    }
    setRowBusy('');
  }

  async function confirmKnowledgeBaseAction() {
    if (!confirmation) return;
    const {type, items: targets} = confirmation;
    setRowBusy(targets.length === 1 ? targets[0].id : 'bulk'); setLibraryError('');
    try {
      if (type === 'disable') {
        const results = await Promise.allSettled(
          targets.filter(item => item.status === 'enabled').map(item => setKnowledgeBaseEnabled(item.id, false))
        );
        const updated = results.flatMap(result => result.status === 'fulfilled' ? [result.value] : []);
        const updatedById = new Map(updated.map(item => [item.id, item]));
        setItems(current => current.map(item => updatedById.get(item.id) || item));
        const failed = results.filter(result => result.status === 'rejected').length;
        if (failed) throw new Error(`${failed} knowledge ${failed === 1 ? 'base' : 'bases'} could not be disabled`);
      } else {
        const results = await Promise.allSettled(targets.map(item => deleteKnowledgeBase(item.id)));
        const deleted = results.filter(result => result.status === 'fulfilled').length;
        const targetPage = items.length - deleted === 0 && page > 1 ? page - 1 : page;
        if (targetPage !== page) setPage(targetPage);
        await loadKnowledgeBases(targetPage, search, activeCompanyId);
        const failed = results.length - deleted;
        if (failed) throw new Error(`${failed} knowledge ${failed === 1 ? 'base' : 'bases'} could not be deleted`);
      }
      setSelectedIds(new Set());
      setConfirmation(null);
    } catch (err) { setLibraryError(err instanceof Error ? err.message : `Could not ${type} this knowledge base`); setConfirmation(null); }
    finally { setRowBusy(''); }
  }

  async function submitCompany(event: React.FormEvent) {
    event.preventDefault();
    const templateError = validateGreetingTemplate(companyDraft.bot_greet_message || '');
    if (templateError) { setCompanyError(templateError); return; }
    setCompanyBusy(true); setCompanyError('');
    let saved: Company | null = null;
    try {
      saved = await saveCompany(companyDraft);
      if (companyLogo) saved = await saveCompanyLogo(saved.id, companyLogo);
      else if (removeLogo && saved.has_logo) await deleteCompanyLogo(saved.id);
      const next = await listCompanies();
      setCompanies(next); setActiveCompanyId(saved.id); resetCompanyEditor();
    } catch (err) {
      if (saved) {
        setActiveCompanyId(saved.id); setCompanyDraft(saved);
        listCompanies().then(setCompanies).catch(() => undefined);
      }
      setCompanyError(err instanceof Error ? err.message : 'Could not save company');
    }
    finally { setCompanyBusy(false); }
  }

  async function confirmDeleteCompany() {
    if (!companyToDelete) return;
    setCompanyBusy(true); setCompanyError('');
    try {
      await deleteCompany(companyToDelete.id);
      const next = companies.filter(company => company.id !== companyToDelete.id);
      setCompanies(next); setActiveCompanyId(current => current === companyToDelete.id ? '' : current);
      if (companyDraft.id === companyToDelete.id) resetCompanyEditor();
      setCompanyToDelete(null);
    } catch (err) { setCompanyError(err instanceof Error ? err.message : 'Could not delete company'); setCompanyToDelete(null); }
    finally { setCompanyBusy(false); }
  }

  return <div className={dark ? 'app dark' : 'app'}>
    <aside>
      <div className="brand"><div className="brand-mark"><Layers3 size={20}/></div><span>Omni<span>RAG</span></span></div>
      <nav><button className={view === 'knowledge' ? 'active' : ''} onClick={() => navigateFromMenu('knowledge')}><Library size={18}/>Knowledge Studio</button><button className={view === 'sessions' ? 'active' : ''} onClick={() => navigateFromMenu('sessions')}><MessageSquareText size={18}/>Chat Sessions</button><button className={view === 'companies' ? 'active' : ''} onClick={() => navigateFromMenu('companies')}><Building2 size={18}/>Companies</button></nav>
      <div className="aside-tip"><Sparkles size={18}/><b>Local & private</b><p>Your documents stay within your infrastructure.</p></div>
      <div className="profile"><div className="avatar">AM</div><div><b>Admin workspace</b><small>Local environment</small></div><ChevronDown size={16}/></div>
    </aside>
    <main>
      <header><div><p className="eyebrow">{view === 'knowledge' ? 'KNOWLEDGE STUDIO' : view === 'sessions' ? 'CHAT SESSIONS' : 'COMPANIES'}</p><h1>{view === 'knowledge' ? 'Build a source of truth' : view === 'sessions' ? 'Review conversations' : 'Manage companies'}</h1><p>{view === 'knowledge' ? 'Turn your documents into precise, citation-ready knowledge.' : view === 'sessions' ? 'Inspect company chatbot sessions and read-only conversation history.' : 'Configure chatbot identity, contact details, and company context.'}</p></div><div className="header-actions"><button className="icon-button" onClick={() => setDark(!dark)} aria-label="Toggle theme">{dark ? <Sun size={18}/> : <Moon size={18}/>}</button><button className="secondary"><BookOpen size={17}/>Documentation</button></div></header>

      {view === 'knowledge' ? <>
      {activeCompany ? <section className="company-context-bar">
        <div className="company-context-identity"><div className="company-context-logo">{activeCompany.has_logo ? <img src={companyLogoUrl(activeCompany.id)} alt=""/> : <Building2 size={24}/>}</div><div><p className="eyebrow">ACTIVE COMPANY</p><b>{activeCompany.name}</b><span>{activeCompany.about || 'Knowledge and chatbot context for this company'}</span></div></div>
        <div className="company-context-controls"><label>Switch workspace<select value={activeCompanyId} onChange={event => { setActiveCompanyId(event.target.value); setPage(1); reset(); }} aria-label="Active company">{companies.map(company => <option key={company.id} value={company.id}>{company.name}</option>)}</select></label><button onClick={() => launchChatbot(activeCompany.id)}><Bot size={16}/>Try ChatBot</button></div>
      </section> : companies.length > 0 ? <section className="company-context-bar empty"><div className="company-context-identity"><div className="company-context-logo"><Building2 size={24}/></div><div><p className="eyebrow">CHOOSE A WORKSPACE</p><b>Select a company</b><span>Knowledge bases are shown only after you choose their company.</span></div></div><div className="company-context-controls"><label>Company<select value="" onChange={event => { setActiveCompanyId(event.target.value); setPage(1); reset(); }} aria-label="Select company"><option value="" disabled>Choose company…</option>{companies.map(company => <option key={company.id} value={company.id}>{company.name}</option>)}</select></label><button onClick={() => setView('companies')}><Settings2 size={16}/>Manage companies</button></div></section> : <section className="company-context-bar empty"><div className="company-context-identity"><div className="company-context-logo"><Building2 size={24}/></div><div><p className="eyebrow">COMPANY REQUIRED</p><b>Create your first company</b><span>Knowledge bases and chatbot context belong to a company.</span></div></div><button onClick={() => setView('companies')}><Building2 size={16}/>Create company</button></section>}
      <section className={`workspace ${importExpanded ? '' : 'collapsed'}`}>
        <div className="workspace-heading">
          <div><p className="eyebrow">IMPORT</p><h2>Add a knowledge base</h2><span>{preview ? file?.name : 'Upload and configure a PDF source'}</span></div>
          <button className="workspace-toggle" onClick={toggleImportWorkspace} aria-expanded={importExpanded} aria-controls="import-workspace">
            {importExpanded ? 'Collapse' : 'Expand'} {importExpanded ? <ChevronUp size={16}/> : <ChevronDown size={16}/>}
          </button>
        </div>
        {success && <div className="import-success"><Check size={18}/><span><b>{success.name} was imported</b><small>{success.chunk_count} searchable chunks created. The form is ready for another document.</small></span><button onClick={() => setSuccess(null)} aria-label="Dismiss"><X size={15}/></button></div>}
        {importExpanded && <div id="import-workspace">
        <div className="stepper"><span className="current"><i>1</i>Upload</span><b/><span className={preview ? 'current' : ''}><i>2</i>Configure</span><b/><span className={success ? 'current' : ''}><i>3</i>Import</span></div>
        {!preview ? <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={e => {e.preventDefault(); setDragging(true)}} onDragLeave={() => setDragging(false)} onDrop={e => {e.preventDefault(); setDragging(false); choose(e.dataTransfer.files[0])}}>
          <div className="upload-icon"><UploadCloud size={30}/></div><h2>{busy ? 'Reading your document…' : 'Bring in a PDF'}</h2><p>Drag and drop it here, or choose a file from your device.</p><button className="primary" disabled={busy} onClick={() => input.current?.click()}>Choose PDF <ArrowRight size={17}/></button><input ref={input} hidden type="file" accept="application/pdf" onChange={e => choose(e.target.files?.[0])}/>{error && <div className="upload-error" role="alert"><AlertTriangle size={16}/><span>{error}</span></div>}<small>PDF up to 40 MB · Text-based documents work best</small>
        </div> : <div className="configure-grid">
          <div className="preview-card card">
            <div className="card-title"><div><p className="eyebrow">DOCUMENT</p><h2>Page preview</h2></div><button className="icon-button" onClick={() => reset()}><X size={17}/></button></div>
            <div className="file-chip"><div className="pdf-icon"><FileText size={20}/></div><div><b>{file?.name}</b><small>{preview.total_pages} pages · {file ? (file.size / 1048576).toFixed(1) : 0} MB</small></div><Check className="check" size={18}/></div>
            <div className="pages">{preview.pages.map(page => { const included = selected.includes(page.page_number); return <article className={included ? 'included' : 'excluded'} key={page.page_number}><span>{page.page_number}</span><div><b>Page {page.page_number}</b><small>{included ? 'Included' : 'Excluded'}</small><p>{page.excerpt || 'No extractable text on this page.'}</p></div></article>; })}</div>
          </div>
          <div className="settings-card card">
            <div className="card-title"><div><p className="eyebrow">CONFIGURATION</p><h2>Shape your knowledge</h2></div><Settings2 size={20}/></div>
            <label>Knowledge base name<input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Product handbook"/></label>
            <div className="two-col"><label>Include only <input value={include} disabled={Boolean(exclude.trim())} onChange={e => setInclude(e.target.value)} placeholder={exclude ? 'Using exclusions' : 'All pages'}/><small>e.g. 1-4, 8, 12-15</small></label><label>Exclude pages <input value={exclude} disabled={Boolean(include.trim())} onChange={e => setExclude(e.target.value)} placeholder={include ? 'Using inclusions' : 'None'}/><small>{selected.length} of {preview.total_pages} selected</small></label></div>
            <label>Chunking strategy</label><div className="strategy-grid">{strategies.map(item => <button key={item.id} onClick={() => setStrategy(item.id)} className={strategy === item.id ? 'selected' : ''}><span>{item.name}</span><small>{item.caption}</small>{strategy === item.id && <Check size={15}/>}</button>)}</div>
            <div className="parameter-box"><div className="two-col"><label>Chunk size <div className="unit-input"><input type="number" value={chunkSize} min="100" max="4000" onChange={e => setChunkSize(+e.target.value)}/><span>chars</span></div></label><label>Overlap <div className="unit-input"><input type="number" value={overlap} min="0" max="1000" onChange={e => setOverlap(+e.target.value)}/><span>chars</span></div></label></div>{strategy === 'semantic' && <label>Similarity threshold <input type="range" min="0" max="1" step=".01" value={threshold} onChange={e => setThreshold(+e.target.value)}/><small>{threshold.toFixed(2)}</small></label>}{strategy === 'hierarchical' && <label>Parent chunk size <div className="unit-input"><input type="number" value={parentSize} onChange={e => setParentSize(+e.target.value)}/><span>chars</span></div></label>}</div>
            {error && <div className="notice error">{error}</div>}
            <button className="primary import" disabled={busy || selected.length === 0 || !name.trim()} onClick={ingest}>{busy ? 'Building index…' : 'Import knowledge base'} {!busy && <ArrowRight size={17}/>}</button>
          </div>
        </div>}
        </div>}
      </section>

      <section className="recent">
        <div className="section-heading">
          <div><p className="eyebrow">LIBRARY</p><h2>Knowledge bases</h2><span>{total} {total === 1 ? 'source' : 'sources'} · newest first</span></div>
          <div className="search"><Search size={16}/><input value={search} onChange={event => { setSearch(event.target.value); setPage(1); }} placeholder="Search library" aria-label="Search knowledge bases"/></div>
        </div>
        {libraryError && <div className="notice error library-error">{libraryError}</div>}
        {selectedItems.length > 0 && <div className="bulk-toolbar" role="toolbar" aria-label="Selected knowledge-base actions"><span><b>{selectedItems.length}</b> selected</span><div><button disabled={selectedDisabledItems.length === 0 || Boolean(rowBusy)} onClick={enableSelectedKnowledgeBases}><CirclePlay size={15}/>Enable</button><button disabled={selectedEnabledItems.length === 0 || Boolean(rowBusy)} onClick={() => setConfirmation({type: 'disable', items: selectedEnabledItems})}><CirclePause size={15}/>Disable</button><button className="bulk-delete" disabled={Boolean(rowBusy)} onClick={() => setConfirmation({type: 'delete', items: selectedItems})}><Trash2 size={14}/>Delete</button><button className="clear-selection" disabled={Boolean(rowBusy)} onClick={() => setSelectedIds(new Set())}>Clear</button></div></div>}
        <div className="kb-list" aria-busy={listLoading}>
          <div className="kb-table-head"><label className="kb-select"><input ref={selectAllInput} type="checkbox" checked={allItemsSelected} onChange={toggleAllItems} disabled={items.length === 0 || listLoading} aria-label="Select all knowledge bases on this page"/><span/></label><span/><span>Knowledge base</span><span>Created</span><span>Strategy</span><span>Pages</span><span>Chunks</span><span>Status</span></div>
          {listLoading ? <div className="library-empty">Loading knowledge bases…</div> : items.length === 0 ? <div className="library-empty"><Library size={22}/><b>{!activeCompany ? 'Select a company' : search ? 'No matching knowledge bases' : 'No knowledge bases yet'}</b><span>{!activeCompany ? 'Choose a company above to view its knowledge bases.' : search ? 'Try a different name or filename.' : 'Import a PDF to create your first source.'}</span></div> : items.map(item => <div className={`kb-row ${item.status === 'disabled' ? 'disabled' : ''} ${selectedIds.has(item.id) ? 'selected' : ''}`} key={item.id}>
            <label className="kb-select"><input type="checkbox" checked={selectedIds.has(item.id)} onChange={() => toggleSelected(item.id)} disabled={Boolean(rowBusy)} aria-label={`Select ${item.name}`}/><span/></label>
            <div className={`kb-icon ${item.mime_type === 'application/pdf' ? 'pdf' : ''}`} title={item.mime_type} aria-label={item.mime_type === 'application/pdf' ? 'PDF document' : 'Document'}><FileText size={19}/>{item.mime_type === 'application/pdf' && <small>PDF</small>}</div>
            <div className="kb-main"><b>{item.name}</b><span>{item.filename}</span></div>
            <time className="created-at" dateTime={item.created_at}>{formatCreatedAt(item.created_at)}</time>
            <div className="kb-detail"><b>{strategyNames[item.chunking.strategy]}</b><span>{item.chunking.chunk_size} chars</span></div>
            <span>{item.selected_pages}</span>
            <span>{item.chunk_count}</span>
            <span className={`status ${item.status}`}><i/>{item.status === 'enabled' ? 'Enabled' : 'Disabled'}</span>
          </div>)}
        </div>
        {totalPages > 1 && <div className="pagination" role="navigation" aria-label="Knowledge-base pages"><span>Page {page} of {totalPages}</span><div><button disabled={page === 1 || listLoading} onClick={() => setPage(current => current - 1)}><ArrowLeft size={15}/>Previous</button><button disabled={page >= totalPages || listLoading} onClick={() => setPage(current => current + 1)}>Next<ArrowRight size={15}/></button></div></div>}
      </section>
      </> : view === 'sessions' ? <>
      {activeCompany ? <section className="company-context-bar">
        <div className="company-context-identity"><div className="company-context-logo">{activeCompany.has_logo ? <img src={companyLogoUrl(activeCompany.id)} alt=""/> : <Building2 size={24}/>}</div><div><p className="eyebrow">ACTIVE COMPANY</p><b>{activeCompany.name}</b><span>Conversation history for {activeCompany.bot_alias}</span></div></div>
        <div className="company-context-controls"><label>Switch workspace<select value={activeCompanyId} onChange={event => setActiveCompanyId(event.target.value)} aria-label="Active company">{companies.map(company => <option key={company.id} value={company.id}>{company.name}</option>)}</select></label><button onClick={() => launchChatbot(activeCompany.id)}><Bot size={16}/>Try ChatBot</button></div>
      </section> : companies.length > 0 ? <section className="company-context-bar empty"><div className="company-context-identity"><div className="company-context-logo"><Building2 size={24}/></div><div><p className="eyebrow">CHOOSE A WORKSPACE</p><b>Select a company</b><span>Chat sessions are scoped to one company.</span></div></div><div className="company-context-controls"><label>Company<select value="" onChange={event => setActiveCompanyId(event.target.value)} aria-label="Select company"><option value="" disabled>Choose company…</option>{companies.map(company => <option key={company.id} value={company.id}>{company.name}</option>)}</select></label><button onClick={() => setView('companies')}><Settings2 size={16}/>Manage companies</button></div></section> : <section className="company-context-bar empty"><div className="company-context-identity"><div className="company-context-logo"><Building2 size={24}/></div><div><p className="eyebrow">COMPANY REQUIRED</p><b>Create your first company</b><span>Chat sessions belong to a company.</span></div></div><button onClick={() => setView('companies')}><Building2 size={16}/>Create company</button></section>}
      {selectedChatSession ? <section className="session-review card">
        <div className="session-review-header"><button className="session-back" onClick={returnToChatSessions}><ArrowLeft size={16}/>Back to sessions</button><div><span className={`session-status ${selectedChatSession.session.status}`}>{selectedChatSession.session.status}</span><b>{formatCreatedAt(selectedChatSession.session.created_at)}</b><small>{selectedChatSession.session.ip_address || 'IP unavailable'} · {selectedChatSession.session.message_count} messages</small></div></div>
        <div className="session-transcript" aria-label="Read-only chat transcript">{selectedChatSession.messages.map(message => <article className={`session-message ${message.owner}`} key={message.id}>{message.owner === 'bot' && <div className={`session-message-avatar avatar-${activeCompany?.bot_avatar || 'bot'}`}>{activeCompany ? avatarIcon(activeCompany.bot_avatar, 15) : <Bot size={15}/>}</div>}<div className="session-message-body"><div className="session-message-meta"><b>{message.owner === 'bot' ? activeCompany?.bot_alias || 'Bot' : 'User'}</b><time dateTime={message.created_at}>{formatClock(message.created_at)}</time></div><p>{message.content}</p></div>{message.owner === 'user' && <div className="session-message-avatar user"><UserRound size={15}/></div>}</article>)}</div>
      </section> : <section className="session-library card">
        <div className="session-library-heading"><div><p className="eyebrow">CONVERSATIONS</p><h2>Chat sessions</h2><span>{chatSessionTotal} {chatSessionTotal === 1 ? 'session' : 'sessions'} · newest first</span></div><div className="session-date-filter"><label><span><CalendarDays size={14}/>From</span><input required type="date" value={chatSessionFrom} min={shiftDateInput(chatSessionTo, -(MAX_CHAT_SESSION_DATE_RANGE_DAYS - 1))} max={chatSessionTo} onChange={event => { const value = event.target.value; setChatSessionFrom(value); if (value && chatSessionTo > shiftDateInput(value, MAX_CHAT_SESSION_DATE_RANGE_DAYS - 1)) setChatSessionTo(shiftDateInput(value, MAX_CHAT_SESSION_DATE_RANGE_DAYS - 1)); setChatSessionPage(1); }}/></label><label><span>To</span><input required type="date" value={chatSessionTo} min={chatSessionFrom} max={shiftDateInput(chatSessionFrom, MAX_CHAT_SESSION_DATE_RANGE_DAYS - 1)} onChange={event => { const value = event.target.value; setChatSessionTo(value); if (value && chatSessionFrom < shiftDateInput(value, -(MAX_CHAT_SESSION_DATE_RANGE_DAYS - 1))) setChatSessionFrom(shiftDateInput(value, -(MAX_CHAT_SESSION_DATE_RANGE_DAYS - 1))); setChatSessionPage(1); }}/></label><label className="session-page-size"><span>Rows</span><select value={chatSessionPageSize} onChange={event => { setChatSessionPageSize(Number(event.target.value)); setChatSessionPage(1); }} aria-label="Chat sessions per page">{CHAT_SESSION_PAGE_SIZES.map(size => <option key={size} value={size}>{size}</option>)}</select></label><button className={`session-refresh ${chatSessionsLoading ? 'loading' : ''}`} disabled={chatSessionsLoading || !activeCompanyId || !chatSessionFrom || !chatSessionTo} onClick={() => void loadChatSessionPage(chatSessionPage, activeCompanyId, chatSessionFrom, chatSessionTo)} aria-label="Refresh chat sessions" title="Refresh chat sessions"><RefreshCw size={16}/></button></div></div>
        {chatSessionError && <div className="notice error">{chatSessionError}</div>}
        <div className="session-table" aria-busy={chatSessionsLoading}><div className="session-table-head"><span>Started</span><span>Status</span><span>Messages</span><span>Duration</span><span>IP address</span><span/></div>{chatSessionsLoading ? <div className="session-empty">Loading chat sessions…</div> : chatSessions.length === 0 ? <div className="session-empty"><MessageSquareText size={24}/><b>{activeCompany ? 'No sessions in this date range' : 'Select a company'}</b><span>{activeCompany ? 'Try another date range or launch the chatbot to begin a conversation.' : 'Choose a company above to review its conversations.'}</span></div> : chatSessions.map(session => <div className="session-row" key={session.id}><time dateTime={session.created_at}>{formatCreatedAt(session.created_at)}</time><span className={`session-status ${session.status}`}>{session.status}</span><span>{session.message_count}</span><span title={session.ended_at ? `Ended ${formatCreatedAt(session.ended_at)}` : undefined}>{formatSessionDuration(session.created_at, session.ended_at)}</span><span>{session.ip_address || 'Unavailable'}</span><button onClick={() => reviewChatSession(session)} aria-label={`View chat session from ${formatCreatedAt(session.created_at)}`}><Eye size={15}/>View chat</button></div>)}</div>
        {chatSessionTotalPages > 1 && <div className="pagination" role="navigation" aria-label="Chat-session pages"><span>Page {chatSessionPage} of {chatSessionTotalPages}</span><div><button disabled={chatSessionPage === 1 || chatSessionsLoading} onClick={() => setChatSessionPage(current => current - 1)}><ArrowLeft size={15}/>Previous</button><button disabled={chatSessionPage >= chatSessionTotalPages || chatSessionsLoading} onClick={() => setChatSessionPage(current => current + 1)}>Next<ArrowRight size={15}/></button></div></div>}
      </section>}
      </> : <section className="company-workspace">
        <form className="company-form card" onSubmit={submitCompany}>
          <div className="card-title"><div><p className="eyebrow">{companyDraft.id ? 'EDIT COMPANY' : 'NEW COMPANY'}</p><h2>{companyDraft.id ? companyDraft.name : 'Company profile'}</h2></div>{companyDraft.id && <button type="button" className="icon-button" onClick={resetCompanyEditor} aria-label="Cancel editing"><X size={17}/></button>}</div>
          <div className="company-identity-fields"><div className="company-logo-upload"><div className="company-logo-picker-wrap"><button type="button" className="company-logo-picker" onClick={() => companyLogoInput.current?.click()} aria-label="Choose company logo"><span className="company-logo-preview">{companyLogoPreview ? <img src={companyLogoPreview} alt="Selected company logo"/> : companyDraft.has_logo && !removeLogo && companyDraft.id ? <img src={companyLogoUrl(companyDraft.id)} alt="Current company logo"/> : <Building2 size={31}/>}<i><ImagePlus size={15}/></i></span></button>{companyDraft.has_logo && !companyLogo && <button type="button" className={`logo-remove-action ${removeLogo ? 'undo' : ''}`} onClick={() => setRemoveLogo(current => !current)} aria-label={removeLogo ? 'Keep existing logo' : 'Remove existing logo'} title={removeLogo ? 'Keep existing logo' : 'Remove existing logo'}>{removeLogo ? <RotateCcw size={14}/> : <Trash2 size={14}/>}</button>}</div><span>{companyLogo ? companyLogo.name : removeLogo ? 'Logo will be removed' : 'Choose logo'}</span><input ref={companyLogoInput} hidden type="file" accept="image/png,image/svg+xml,image/jpeg" onChange={event => chooseCompanyLogo(event.target.files?.[0])}/></div><label className="company-name-field">Company name<input required minLength={2} value={companyDraft.name} onChange={event => setCompanyDraft(current => ({...current, name:event.target.value}))}/><small>Logo: PNG, SVG or JPEG · max 512 KB</small></label></div>
          <label>About<textarea maxLength={1000} value={companyDraft.about || ''} onChange={event => setCompanyDraft(current => ({...current, about:event.target.value}))} placeholder="What should the assistant know about this company?"/><small>{companyDraft.about?.length || 0}/1000</small></label>
          <div className="two-col"><label><Phone size={13}/> Phone<input value={companyDraft.phone || ''} onChange={event => setCompanyDraft(current => ({...current, phone:event.target.value}))}/></label><label><Mail size={13}/> Email<input type="email" value={companyDraft.email || ''} onChange={event => setCompanyDraft(current => ({...current, email:event.target.value}))}/></label></div>
          <label><MapPin size={13}/> Address<textarea value={companyDraft.address || ''} onChange={event => setCompanyDraft(current => ({...current, address:event.target.value}))}/></label>
          <label>Maps URL<div className="map-url-field"><input value={companyDraft.maps_url || ''} onChange={event => setCompanyDraft(current => ({...current, maps_url:event.target.value}))} placeholder="https://maps.google.com/…"/>{mapsLink ? <a href={mapsLink} target="_blank" rel="noreferrer" aria-label="Open location in a new tab" title="Open location in a new tab"><MapPinned size={17}/></a> : <button type="button" disabled aria-label="Enter a valid Maps URL to open location" title="Enter a valid Maps URL"><MapPinned size={17}/></button>}</div></label>
          <div className="two-col"><label>Bot alias<input required minLength={2} value={companyDraft.bot_alias || 'AIBot'} onChange={event => setCompanyDraft(current => ({...current, bot_alias:event.target.value}))}/></label><fieldset><legend>Bot avatar</legend><div className="avatar-options">{avatarOptions.map(avatar => <button type="button" key={avatar.id} className={`avatar-${avatar.id} ${companyDraft.bot_avatar === avatar.id ? 'selected' : ''}`} onClick={() => setCompanyDraft(current => ({...current, bot_avatar:avatar.id}))} aria-label={avatar.label} title={avatar.label}>{avatarIcon(avatar.id)}</button>)}</div></fieldset></div>
          <label>Bot greeting message<textarea maxLength={400} value={companyDraft.bot_greet_message || ''} onChange={event => setCompanyDraft(current => ({...current, bot_greet_message:event.target.value}))} placeholder={DEFAULT_BOT_GREET_MESSAGE}/><small>{companyDraft.bot_greet_message?.length || 0}/400 · Only {'{{bot_alias}}'} and {'{{company_name}}'} placeholders are allowed</small></label>
          {companyError && <div className="notice error">{companyError}</div>}
          <button className="primary" disabled={companyBusy}>{companyBusy ? 'Saving…' : companyDraft.id ? 'Update company' : 'Create company'}<ArrowRight size={16}/></button>
        </form>
        <div className="company-list"><div className="section-heading"><div><p className="eyebrow">COMPANIES</p><h2>{companies.length} configured</h2></div></div>{companies.length === 0 ? <div className="company-empty"><Building2 size={25}/><b>No companies yet</b><span>Create one to begin importing knowledge.</span></div> : companies.map(company => <article className="company-card" key={company.id}><div className="company-logo">{company.has_logo ? <img src={companyLogoUrl(company.id)} alt=""/> : <Building2 size={22}/>}</div><div className="company-card-main"><div><b>{company.name}</b><span>{company.about || 'No company description yet.'}</span></div><div className="company-meta"><span>{avatarIcon(company.bot_avatar, 14)} {company.bot_alias}</span>{company.email && <span><Mail size={13}/>{company.email}</span>}{company.phone && <span><Phone size={13}/>{company.phone}</span>}</div></div><div className="company-actions"><button onClick={() => editCompany(company)} aria-label={`Edit ${company.name}`}><Pencil size={15}/></button><button className="delete" onClick={() => setCompanyToDelete(company)} aria-label={`Delete ${company.name}`}><Trash2 size={15}/></button></div></article>)}</div>
      </section>}
    </main>
    {confirmation && <div className="confirmation-backdrop" onMouseDown={() => !rowBusy && setConfirmation(null)}><section className={`confirmation-dialog ${confirmation.type}`} role="dialog" aria-modal="true" aria-labelledby="confirmation-title" onMouseDown={event => event.stopPropagation()}><div className="confirmation-icon">{confirmation.type === 'delete' ? <AlertTriangle size={22}/> : <CirclePause size={22}/>}</div><div className="confirmation-copy"><p className="eyebrow">{confirmation.type === 'delete' ? 'PERMANENT ACTION' : 'CHANGE AVAILABILITY'}</p><h2 id="confirmation-title">{confirmation.type === 'delete' ? `Delete ${confirmation.items.length === 1 ? 'knowledge base' : `${confirmation.items.length} knowledge bases`}?` : `Disable ${confirmation.items.length === 1 ? 'knowledge base' : `${confirmation.items.length} knowledge bases`}?`}</h2><p>{confirmation.type === 'delete' ? 'This permanently removes the selected knowledge-base records and all associated vector chunks. This action cannot be undone.' : 'The selected sources will disappear from the chatbot and will not be used for retrieval. You can enable them again at any time.'}</p><div className="confirmation-target"><FileText size={17}/><span><b>{confirmation.items.length === 1 ? confirmation.items[0].name : `${confirmation.items.length} knowledge bases selected`}</b><small>{confirmation.items.length === 1 ? `${confirmation.items[0].chunk_count} chunks · ${confirmation.items[0].selected_pages} pages` : confirmation.items.map(item => item.name).join(', ')}</small></span></div></div><div className="confirmation-actions"><button className="cancel" autoFocus disabled={Boolean(rowBusy)} onClick={() => setConfirmation(null)}>Keep {confirmation.items.length === 1 ? 'knowledge base' : 'knowledge bases'}</button><button className={confirmation.type === 'delete' ? 'confirm-delete' : 'confirm-disable'} disabled={Boolean(rowBusy)} onClick={confirmKnowledgeBaseAction}>{rowBusy ? 'Working…' : confirmation.type === 'delete' ? `Delete ${confirmation.items.length === 1 ? 'permanently' : 'selected'}` : `Disable ${confirmation.items.length === 1 ? 'knowledge base' : 'selected'}`}</button></div></section></div>}
    {companyToDelete && <div className="confirmation-backdrop" onMouseDown={() => !companyBusy && setCompanyToDelete(null)}><section className="confirmation-dialog delete" role="dialog" aria-modal="true" aria-labelledby="company-delete-title" onMouseDown={event => event.stopPropagation()}><div className="confirmation-icon"><AlertTriangle size={22}/></div><div className="confirmation-copy"><p className="eyebrow">PERMANENT ACTION</p><h2 id="company-delete-title">Delete company?</h2><p>This permanently removes the company, every associated knowledge base, and all related Qdrant chunks.</p><div className="confirmation-target"><Building2 size={17}/><span><b>{companyToDelete.name}</b><small>This action cannot be undone.</small></span></div></div><div className="confirmation-actions"><button className="cancel" autoFocus disabled={companyBusy} onClick={() => setCompanyToDelete(null)}>Keep company</button><button className="confirm-delete" disabled={companyBusy} onClick={confirmDeleteCompany}>{companyBusy ? 'Deleting…' : 'Delete company'}</button></div></section></div>}
  </div>;
}

export { rangeToPages };
