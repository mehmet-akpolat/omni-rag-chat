import { describe, expect, it } from 'vitest';
import { formatMessageTime, formatSessionTime, messageSegments, messagesForModel, persistedMessages, positiveNumber, sessionIdFromSearch } from './App';

describe('sessionIdFromSearch', () => {
  it('reads and decodes the session-specific launch parameter', () => {
    expect(sessionIdFromSearch('?session_id=session%20one&ref=website')).toBe('session one');
  });

  it('does not invent a session when the parameter is absent', () => {
    expect(sessionIdFromSearch('?ref=website')).toBe('');
  });
});

describe('chat timing helpers', () => {
  it('formats session countdowns and message times', () => {
    expect(formatSessionTime(601)).toBe('10:01');
    expect(formatSessionTime(-1)).toBe('00:00');
    expect(formatMessageTime('2026-09-17T12:34:56Z')).toMatch(/^\d{2}:\d{2}:\d{2}$/);
  });

  it('uses positive configuration values or their defaults', () => {
    expect(positiveNumber('15', 10)).toBe(15);
    expect(positiveNumber('0', 10)).toBe(10);
    expect(positiveNumber('invalid', 60)).toBe(60);
  });
});

describe('persistedMessages', () => {
  it('maps the backend transcript and excludes its greeting from model history', () => {
    const [greeting, user] = persistedMessages([
      {id:'bot-1', session_id:'s1', owner:'bot', content:'Welcome to Acme Corp.', created_at:'2026-09-17T10:00:00Z'},
      {id:'user-1', session_id:'s1', owner:'user', content:'Hello', created_at:'2026-09-17T10:00:01Z'},
    ]);
    expect(greeting).toMatchObject({role: 'assistant', kind: 'greeting', content: 'Welcome to Acme Corp.'});
    expect(greeting.created_at).toBeTruthy();
    expect(messagesForModel([greeting, user])).toEqual([user]);
  });
});

describe('messageSegments', () => {
  it('separates safe web links from surrounding text and punctuation', () => {
    expect(messageSegments('Visit https://example.com/help, then http://docs.example.com.')).toEqual([
      {text: 'Visit '},
      {text: 'https://example.com/help', href: 'https://example.com/help'},
      {text: ', then '},
      {text: 'http://docs.example.com', href: 'http://docs.example.com'},
      {text: '.'},
    ]);
  });
});
