// Screen 2: what the agent did (trace) and what it found (result).

import { useState } from "react";
import Markdown from "react-markdown";
import type { Finding, RunResult } from "./api";
import { Badge, SectionTitle, Spinner, buttonStyles, toolLabel } from "./ui";

export type RunState =
  | { status: "running" }
  | { status: "done"; result: RunResult }
  | { status: "failed"; error: string };

interface Props {
  run: RunState;
  agentName: string;
  onBack: () => void;
  onEdit: () => void;
  onRetry: () => void;
}

export function Result({ run, agentName, onBack, onEdit, onRetry }: Props) {
  return (
    <div className="mx-auto max-w-xl px-4 pb-12 pt-3">
      <div className="mb-4 flex items-center justify-between gap-2">
        <button type="button" onClick={onBack} className={buttonStyles.ghost + " -ml-3"}>
          ← Back
        </button>
        <span className="truncate text-sm font-semibold text-muted">{agentName}</span>
      </div>

      {run.status === "running" && <Running />}
      {run.status === "failed" && <Failure message={run.error} onRetry={onRetry} onBack={onBack} />}
      {run.status === "done" && !run.result.ok && (
        <Failure message={run.result.error ?? "The agent could not finish."} onRetry={onRetry} onBack={onBack} />
      )}
      {run.status === "done" && run.result.ok && <Success result={run.result} agentName={agentName} onEdit={onEdit} />}
    </div>
  );
}

function Running() {
  return (
    <div className="rounded-2xl border border-line bg-card p-5" role="status" aria-live="polite">
      <div className="flex items-center gap-3 font-semibold">
        <Spinner /> Running... can take up to 20s
      </div>
      <p className="mt-1 text-sm text-muted">The agent is choosing tools, reading documents and checking policies.</p>
      <div className="mt-4 space-y-2" aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <div key={i} className="pulse-bar h-10 rounded-lg bg-info-soft" style={{ animationDelay: `${i * 0.2}s` }} />
        ))}
      </div>
    </div>
  );
}

function Failure({ message, onRetry, onBack }: { message: string; onRetry: () => void; onBack: () => void }) {
  return (
    <div className="rounded-2xl border border-high/30 bg-high-soft p-5" role="alert">
      <p className="font-semibold text-high">That didn't work</p>
      <p className="mt-1 text-[15px] leading-snug">{message}</p>
      <div className="mt-4 grid grid-cols-2 gap-2.5">
        <button type="button" onClick={onRetry} className={buttonStyles.primary}>
          Try again
        </button>
        <button type="button" onClick={onBack} className={buttonStyles.secondary}>
          Back
        </button>
      </div>
    </div>
  );
}

function Success({ result, agentName, onEdit }: { result: RunResult; agentName: string; onEdit: () => void }) {
  return (
    <>
      <StatusLine result={result} />
      <Trace result={result} />
      <section className="mb-6">
        <SectionTitle>What it found</SectionTitle>
        {result.findings ? <FindingCards findings={result.findings.findings} summary={result.findings.summary} /> : <Report result={result} />}
      </section>
      <Actions result={result} agentName={agentName} onEdit={onEdit} />
    </>
  );
}

function StatusLine({ result }: { result: RunResult }) {
  return (
    <div className="mb-5 flex flex-wrap gap-2">
      <Badge tone="ok">✓ Done</Badge>
      {result.provider && <Badge>Model: {result.provider}</Badge>}
      {result.cached && <Badge tone="accent">Cached run</Badge>}
      {result.mode === "fake" && <Badge tone="medium">Demo mode (scripted model)</Badge>}
    </div>
  );
}

function Trace({ result }: { result: RunResult }) {
  return (
    <section className="mb-6">
      <SectionTitle>What the agent did</SectionTitle>
      {result.trace.length === 0 ? (
        <p className="text-sm text-muted">It answered without calling any tools.</p>
      ) : (
        <ol className="relative space-y-3 border-l-2 border-dashed border-line pl-5">
          {result.trace.map((step) => (
            <li key={step.step} className="relative">
              <span className="absolute -left-[33px] top-2 grid h-6 w-6 place-items-center rounded-md bg-ink font-mono text-xs text-paper">
                {step.step}
              </span>
              <div className="rounded-xl border border-line bg-card p-3">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-semibold">{toolLabel(step.tool)}</span>
                  <span className="shrink-0 font-mono text-xs text-muted">{step.durationMs} ms</span>
                </div>
                <p className="mt-1 font-mono text-xs leading-relaxed text-muted [overflow-wrap:anywhere]">{step.summary}</p>
              </div>
            </li>
          ))}
          <li className="relative">
            <span className="absolute -left-[33px] top-1.5 grid h-6 w-6 place-items-center rounded-md bg-accent text-xs text-white">
              ✎
            </span>
            <p className="py-1.5 text-sm font-semibold">Wrote the answer from what it gathered</p>
          </li>
        </ol>
      )}
    </section>
  );
}

const SEVERITY_TONE = { high: "high", medium: "medium", low: "low", info: "info" } as const;

function FindingCards({ findings, summary }: { findings: Finding[]; summary: string }) {
  return (
    <div className="space-y-3">
      <p className="text-[15px] leading-snug [overflow-wrap:anywhere]">{summary}</p>
      {findings.map((finding, i) => (
        <article key={i} className="rounded-2xl border border-line bg-card p-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={SEVERITY_TONE[finding.severity]}>{finding.severity.toUpperCase()}</Badge>
            <h3 className="font-semibold leading-tight [overflow-wrap:anywhere]">{finding.title}</h3>
          </div>
          <p className="mt-2 text-[15px] leading-snug [overflow-wrap:anywhere]">{finding.detail}</p>
          {finding.quote && <Quote quote={finding.quote} source={finding.source} verified={finding.verified} />}
        </article>
      ))}
    </div>
  );
}

function Quote({ quote, source, verified }: { quote: string; source: string | null; verified: boolean | null }) {
  return (
    <figure className="mt-3 rounded-lg border-l-3 border-accent bg-paper px-3 py-2">
      <blockquote className="text-sm leading-snug [overflow-wrap:anywhere]">“{quote}”</blockquote>
      <figcaption className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
        {source && <code className="font-mono text-muted">{source}</code>}
        {verified ? (
          <span className="font-semibold text-ok">✓ Verified in source</span>
        ) : (
          <span className="font-semibold text-high">✗ Not found in source</span>
        )}
      </figcaption>
    </figure>
  );
}

function Report({ result }: { result: RunResult }) {
  return (
    <div className="rounded-2xl border border-line bg-card p-4">
      {/* react-markdown without rehype-raw: raw HTML from the model is never rendered. */}
      <div className="md">
        <Markdown skipHtml>{result.markdown ?? ""}</Markdown>
      </div>
      {result.citations.length > 0 && (
        <div className="mt-4 border-t border-line pt-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted">Citations</p>
          {result.citations.map((c, i) => (
            <Quote key={i} quote={c.quote} source={c.source} verified={c.verified} />
          ))}
        </div>
      )}
    </div>
  );
}

// --- Sharing: copy and mailto (no backend needed) --------------------------

export function resultAsText(result: RunResult, agentName: string): string {
  const lines = [`${agentName}: agent result`, ""];
  if (result.findings) {
    lines.push(result.findings.summary, "");
    for (const f of result.findings.findings) {
      lines.push(`[${f.severity.toUpperCase()}] ${f.title}`, f.detail);
      if (f.quote) lines.push(`"${f.quote}" (${f.source ?? "unknown"}, ${f.verified ? "verified" : "not verified"})`);
      lines.push("");
    }
  } else {
    lines.push(result.markdown ?? "", "");
  }
  lines.push("From the AgentBlocks demo. All data is fictional.");
  return lines.join("\n");
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Older browsers or non-secure pages: fall back to a hidden textarea.
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  }
}

function Actions({ result, agentName, onEdit }: { result: RunResult; agentName: string; onEdit: () => void }) {
  const [copied, setCopied] = useState<"idle" | "copied" | "failed">("idle");
  const text = resultAsText(result, agentName);
  // Keep mailto links short: some mail apps reject very long URLs.
  const body = text.length > 1500 ? `${text.slice(0, 1500)}\n...(shortened)` : text;
  const mailto = `mailto:?subject=${encodeURIComponent(`Agent result: ${agentName}`.slice(0, 120))}&body=${encodeURIComponent(body)}`;

  const copy = async () => {
    setCopied((await copyText(text)) ? "copied" : "failed");
    setTimeout(() => setCopied("idle"), 2000);
  };

  return (
    <section className="space-y-2.5">
      <div className="grid grid-cols-2 gap-2.5">
        <button type="button" onClick={copy} className={buttonStyles.secondary}>
          {copied === "copied" ? "✓ Copied" : copied === "failed" ? "Copy failed" : "Copy result"}
        </button>
        <a href={mailto} className={buttonStyles.secondary}>
          Open in my email app
        </a>
      </div>
      <button type="button" onClick={onEdit} className={`${buttonStyles.primary} w-full`}>
        Edit this agent
      </button>
    </section>
  );
}
