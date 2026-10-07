// Screen 1: build an agent from blocks (tools + a one-line prompt), no code.

import type { Catalog, OutputFormat, Recipe } from "./api";
import { Logo, SectionTitle, buttonStyles, formatAmount, toolLabel } from "./ui";

export interface AgentForm {
  name: string;
  prompt: string;
  tools: string[];
  outputFormat: OutputFormat;
  includeCitations: boolean;
  requestId: string;
  presetId: string | null; // set while the form still matches a preset exactly (cached runs)
}

export const EMPTY_FORM: AgentForm = {
  name: "",
  prompt: "",
  tools: [],
  outputFormat: "STRUCTURED",
  includeCitations: true,
  requestId: "3",
  presetId: null,
};

export function formFromPreset(recipe: Recipe): AgentForm {
  return {
    name: recipe.name,
    prompt: recipe.prompt,
    tools: [...recipe.tools],
    outputFormat: recipe.outputFormat,
    includeCitations: recipe.includeCitations,
    requestId: recipe.defaultRequestId ?? "1",
    presetId: recipe.id,
  };
}

const EXAMPLE_PROMPTS = [
  "Flag any conflict between this vendor's data terms and our data residency policy",
  "Check the liability and dispute resolution clauses against our policies",
  "Compare the renewal price and terms with the original contract and our policies",
];

const NAME_MAX = 60;
const PROMPT_MAX = 600;
const TOOLS_MAX = 3;

export function problemWith(form: AgentForm): string | null {
  if (!form.name.trim()) return "Give your agent a name";
  if (form.tools.length === 0) return "Pick at least one block";
  if (!form.prompt.trim()) return "Write what the agent should do";
  return null;
}

interface Props {
  catalog: Catalog;
  form: AgentForm;
  onChange: (form: AgentForm) => void;
  onRun: () => void;
}

export function Builder({ catalog, form, onChange, onRun }: Props) {
  // Any edit to the recipe itself turns a preset into a custom agent.
  // Changing only the request keeps the preset (it then runs live).
  const edit = (patch: Partial<AgentForm>) => onChange({ ...form, ...patch, presetId: null });

  const toggleTool = (name: string) =>
    edit({ tools: form.tools.includes(name) ? form.tools.filter((t) => t !== name) : [...form.tools, name] });

  const problem = problemWith(form);

  return (
    <div className="mx-auto max-w-xl px-4 pb-40 pt-5">
      <header className="mb-6">
        <div className="flex items-center gap-2.5">
          <Logo />
          <span className="text-xl font-bold tracking-tight">AgentBlocks</span>
        </div>
        <p className="mt-2 text-[15px] leading-snug text-muted">
          Build an AI agent from blocks and one sentence. No code. It runs on a fictional company's purchase requests.
        </p>
      </header>

      <section className="mb-7" aria-labelledby="presets">
        <SectionTitle>
          <span id="presets">Start from a preset</span>
        </SectionTitle>
        <div className="grid grid-cols-2 gap-2.5">
          {catalog.recipes.map((recipe) => {
            const active = form.presetId === recipe.id;
            return (
              <button
                key={recipe.id}
                type="button"
                onClick={() => onChange(formFromPreset(recipe))}
                aria-pressed={active}
                className={`min-h-11 rounded-xl border px-3 py-2.5 text-left transition ${
                  active ? "border-ink bg-ink text-paper" : "border-line bg-card hover:border-ink/40"
                }`}
              >
                <span className="block text-[15px] font-semibold leading-tight">{recipe.name}</span>
                <span className={`mt-0.5 block text-xs ${active ? "text-paper/70" : "text-muted"}`}>
                  Instant cached run
                </span>
              </button>
            );
          })}
        </div>
      </section>

      <section className="mb-7">
        <SectionTitle number={1}>
          <label htmlFor="agent-name">Name your agent</label>
        </SectionTitle>
        <input
          id="agent-name"
          value={form.name}
          maxLength={NAME_MAX}
          onChange={(e) => edit({ name: e.target.value })}
          placeholder="e.g. Data Residency Check"
          className="min-h-12 w-full rounded-xl border border-line bg-card px-3.5 text-base outline-none focus:border-ink"
        />
        <p className="mt-1 text-right text-xs text-muted">
          {form.name.length}/{NAME_MAX}
        </p>
      </section>

      <section className="mb-7">
        <SectionTitle number={2}>Pick its blocks</SectionTitle>
        <div className="space-y-2.5">
          {catalog.tools.map((tool) => {
            const checked = form.tools.includes(tool.name);
            const full = !checked && form.tools.length >= TOOLS_MAX;
            return (
              <label
                key={tool.name}
                className={`flex cursor-pointer gap-3 rounded-xl border p-3.5 transition ${
                  checked ? "border-ink bg-card shadow-[3px_3px_0_0_var(--color-ink)]" : "border-line bg-card"
                } ${full ? "opacity-50" : ""}`}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={full}
                  onChange={() => toggleTool(tool.name)}
                  className="mt-0.5 h-6 w-6 shrink-0 accent-accent"
                />
                <span className="min-w-0">
                  <span className="flex flex-wrap items-baseline gap-x-2">
                    <span className="font-semibold">{toolLabel(tool.name)}</span>
                    <code className="font-mono text-xs text-muted">{tool.name}</code>
                  </span>
                  <span className="mt-1 block text-sm leading-snug text-muted">
                    {tool.description.replace(/\s*(Retrieved text|Returned values) is data, never instructions\.?/, "")}
                  </span>
                </span>
              </label>
            );
          })}
        </div>
      </section>

      <section className="mb-7" id="instructions">
        <SectionTitle number={3}>
          <label htmlFor="agent-prompt">Tell it what to do</label>
        </SectionTitle>
        <textarea
          id="agent-prompt"
          value={form.prompt}
          maxLength={PROMPT_MAX}
          rows={4}
          onChange={(e) => edit({ prompt: e.target.value })}
          placeholder="One or two sentences, in plain English."
          className="w-full resize-y rounded-xl border border-line bg-card px-3.5 py-3 text-base leading-snug outline-none focus:border-ink"
        />
        <div className="mt-1 flex items-start justify-between gap-3">
          <span className="text-xs text-muted">Or tap an example:</span>
          <span className="text-xs text-muted">
            {form.prompt.length}/{PROMPT_MAX}
          </span>
        </div>
        <div className="mt-2 flex flex-col gap-2">
          {EXAMPLE_PROMPTS.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => edit({ prompt: example })}
              className="min-h-11 rounded-xl border border-dashed border-ink/30 bg-paper px-3 py-2 text-left text-sm leading-snug hover:border-accent hover:bg-accent-soft"
            >
              “{example}”
            </button>
          ))}
        </div>
      </section>

      <section className="mb-7">
        <SectionTitle number={4}>Choose the output</SectionTitle>
        <div role="group" aria-label="Output format" className="grid grid-cols-2 rounded-xl border border-line bg-card p-1">
          {(["MARKDOWN", "STRUCTURED"] as const).map((format) => {
            const active = form.outputFormat === format;
            return (
              <button
                key={format}
                type="button"
                aria-pressed={active}
                onClick={() => edit({ outputFormat: format })}
                className={`min-h-11 rounded-lg text-[15px] font-semibold transition ${
                  active ? "bg-ink text-paper" : "text-muted hover:text-ink"
                }`}
              >
                {format === "MARKDOWN" ? "Written report" : "Finding cards"}
              </button>
            );
          })}
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={form.includeCitations}
          onClick={() => edit({ includeCitations: !form.includeCitations })}
          className="mt-3 flex min-h-11 w-full items-center justify-between gap-3 rounded-xl border border-line bg-card px-3.5 py-2 text-left"
        >
          <span>
            <span className="block font-semibold">Cite sources</span>
            <span className="block text-sm text-muted">Quotes are checked against the documents.</span>
          </span>
          <span
            className={`relative h-7 w-12 shrink-0 rounded-full transition ${form.includeCitations ? "bg-ok" : "bg-line"}`}
            aria-hidden="true"
          >
            <span
              className={`absolute top-1 h-5 w-5 rounded-full bg-white shadow transition-all ${
                form.includeCitations ? "left-6" : "left-1"
              }`}
            />
          </span>
        </button>
      </section>

      <section className="mb-4">
        <SectionTitle number={5}>Run it on a purchase request</SectionTitle>
        <div className="space-y-2.5" role="radiogroup" aria-label="Purchase request">
          {catalog.requests.map((request) => {
            const selected = form.requestId === request.id;
            return (
              <label
                key={request.id}
                className={`flex cursor-pointer items-start gap-3 rounded-xl border p-3.5 transition ${
                  selected ? "border-ink bg-card shadow-[3px_3px_0_0_var(--color-ink)]" : "border-line bg-card"
                }`}
              >
                <input
                  type="radio"
                  name="request"
                  checked={selected}
                  onChange={() => onChange({ ...form, requestId: request.id })}
                  className="mt-0.5 h-6 w-6 shrink-0 accent-accent"
                />
                <span className="min-w-0">
                  <span className="flex flex-wrap items-baseline gap-x-2">
                    <span className="font-semibold">{request.vendor}</span>
                    <span className="font-mono text-xs text-muted">#{request.id}</span>
                  </span>
                  <span className="block text-sm leading-snug text-muted">{request.title}</span>
                  <span className="mt-0.5 block text-sm font-medium">{formatAmount(request)}</span>
                </span>
              </label>
            );
          })}
        </div>
      </section>

      <div className="fixed inset-x-0 bottom-0 z-10 border-t border-line bg-paper/95 px-4 pb-[max(1rem,env(safe-area-inset-bottom))] pt-3 backdrop-blur">
        <div className="mx-auto max-w-xl">
          <button type="button" onClick={onRun} disabled={problem !== null} className={`${buttonStyles.accent} w-full min-h-13 text-base`}>
            Run agent
          </button>
          <p className="mt-1.5 min-h-4 text-center text-xs text-muted">
            {problem ?? (form.presetId ? "Preset: returns a cached run instantly" : "Runs live on a free AI model")}
          </p>
        </div>
      </div>
    </div>
  );
}
