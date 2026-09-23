import { describe, expect, it } from 'vitest';
import { chatbotLaunchUrl, dateInputValue, formatClock, formatCreatedAt, formatMetric, formatSessionDuration, isUrlMime, maskApiKey, providerLabel, rangeToPages, shiftDateInput, shouldAutoExpandImport, supportsUrlImport, validateGreetingTemplate } from './App';
describe('chatbotLaunchUrl', () => { it('creates a session-specific chat URL', () => { expect(chatbotLaunchUrl('session/a', 'https://chat.example.test')).toBe('https://chat.example.test?session_id=session%2Fa'); }); });
describe('rangeToPages', () => { it('parses and clamps page ranges', () => { expect(rangeToPages('1-3, 7, 12-15, nope', 12)).toEqual([1,2,3,7,12]); }); });
describe('formatCreatedAt', () => { it('formats a stored creation timestamp', () => { expect(formatCreatedAt('2026-09-16T10:30:00Z')).toContain('2026'); }); });
describe('chat session dates', () => {
  it('formats date inputs and message clocks', () => {
    expect(dateInputValue(new Date('2026-09-17T10:30:00Z'))).toMatch(/^2026-09-17$/);
    expect(formatClock('2026-09-17T10:30:45Z')).toMatch(/:30:45/);
    expect(shiftDateInput('2026-09-17', -13)).toBe('2026-09-04');
    expect(shiftDateInput('2026-09-17', 13)).toBe('2026-09-30');
    expect(formatSessionDuration('2026-09-17T10:00:00Z', '2026-09-17T10:00:45Z')).toBe('45 secs');
    expect(formatSessionDuration('2026-09-17T10:00:00Z', '2026-09-17T10:00:01Z')).toBe('1 sec');
    expect(formatSessionDuration('2026-09-17T10:00:00Z', '2026-09-17T10:02:01Z')).toBe('2 mins 1 sec');
    expect(formatSessionDuration('2026-09-17T10:00:00Z')).toBe('-');
  });
});
describe('analytics metrics', () => {
  it('formats whole and fractional metric values', () => {
    expect(formatMetric(1200)).toMatch(/1[,.]200/);
    expect(formatMetric(2.345, 2)).toMatch(/2[,.]35/);
  });
});
describe('shouldAutoExpandImport', () => {
  it('opens only for an empty, unfiltered knowledge-base list', () => {
    expect(shouldAutoExpandImport(0, '')).toBe(true);
    expect(shouldAutoExpandImport(1, '')).toBe(false);
    expect(shouldAutoExpandImport(0, 'missing')).toBe(false);
  });
});

describe('supportsUrlImport', () => {
  it('allows only structure-aware URL strategies', () => {
    expect(supportsUrlImport('recursive')).toBe(true);
    expect(supportsUrlImport('hierarchical')).toBe(true);
    expect(supportsUrlImport('fixed')).toBe(false);
    expect(supportsUrlImport('semantic')).toBe(false);
  });
});
describe('isUrlMime', () => {
  it('derives URL sources from MIME type', () => {
    expect(isUrlMime('text/html')).toBe(true);
    expect(isUrlMime('application/xhtml+xml')).toBe(true);
    expect(isUrlMime('application/pdf')).toBe(false);
  });
});
describe('LLM presentation', () => {
  it('uses provider branding and masks all but the first and last two key characters', () => {
    expect(providerLabel('openai')).toBe('OpenAI');
    expect(providerLabel('huggingface')).toBe('HuggingFace');
    expect(maskApiKey('company-openai-key')).toBe('co••••••••ey');
    expect(maskApiKey('abc')).toBe('•••');
  });
});
describe('validateGreetingTemplate', () => {
  it('accepts only supported placeholders and plain text', () => {
    expect(validateGreetingTemplate('Hello {{bot_alias}} from {{company_name}}.')).toBeUndefined();
    expect(validateGreetingTemplate('Hello there.')).toBeUndefined();
    expect(validateGreetingTemplate('Hello {{user_name}}.')).toContain('Unsupported greeting placeholder');
    expect(validateGreetingTemplate('Hello {{company_name}.')).toContain('malformed placeholder syntax');
  });
});
