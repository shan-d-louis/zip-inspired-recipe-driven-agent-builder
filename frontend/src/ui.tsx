// Small shared pieces: logo, labels, buttons, badges.

import type { ReactNode } from "react";
import type { PurchaseRequest } from "./api";

/** Friendly names for the tool blocks (the code name is shown too). */
export const TOOL_LABELS: Record<string, string> = {
  document_retrieval: "Document search",
  api_data: "Request data",
  company_context: "Company policies",
};

export function toolLabel(name: string): string {
  return TOOL_LABELS[name] ?? name;
}

export function formatAmount(request: PurchaseRequest): string {
  return new Intl.NumberFormat("en-CA", {
    style: "currency",
    currency: request.currency,
    maximumFractionDigits: 0,
  }).format(request.amount);
}

export function Logo() {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden="true">
      <rect x="3" y="3" width="12" height="12" rx="2" className="fill-ink" />
      <rect x="17" y="3" width="12" height="12" rx="2" className="fill-accent" />
      <rect x="3" y="17" width="12" height="12" rx="2" className="fill-accent" />
      <rect x="17" y="17" width="12" height="12" rx="2" className="fill-ink" />
    </svg>
  );
}

const BUTTON_BASE =
  "inline-flex min-h-11 items-center justify-center gap-2 rounded-xl px-4 text-[15px] font-semibold " +
  "transition active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-45 " +
  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

export const buttonStyles = {
  primary: `${BUTTON_BASE} bg-ink text-paper hover:bg-black`,
  accent: `${BUTTON_BASE} bg-accent text-white hover:bg-accent-ink`,
  secondary: `${BUTTON_BASE} border border-line bg-card text-ink hover:border-ink/40`,
  ghost: `${BUTTON_BASE} text-ink hover:bg-ink/5`,
};

const BADGE_TONES = {
  neutral: "bg-info-soft text-info",
  accent: "bg-accent-soft text-accent-ink",
  ok: "bg-ok-soft text-ok",
  high: "bg-high-soft text-high",
  medium: "bg-medium-soft text-medium",
  low: "bg-low-soft text-low",
  info: "bg-info-soft text-info",
};

export function Badge({ tone = "neutral", children }: { tone?: keyof typeof BADGE_TONES; children: ReactNode }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ${BADGE_TONES[tone]}`}>
      {children}
    </span>
  );
}

export function SectionTitle({ number, children }: { number?: number; children: ReactNode }) {
  return (
    <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-muted">
      {number !== undefined && (
        <span className="grid h-6 w-6 place-items-center rounded-md bg-ink font-mono text-xs text-paper">{number}</span>
      )}
      {children}
    </h2>
  );
}

export function Spinner() {
  return (
    <span
      className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-current border-r-transparent"
      aria-hidden="true"
    />
  );
}
