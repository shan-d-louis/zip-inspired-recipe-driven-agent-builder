// Two screens, one piece of state each: the agent form and the current run.

import { useEffect, useState } from "react";
import { ApiError, loadCatalog, recipeIdFor, runRecipe, type Catalog } from "./api";
import { Builder, EMPTY_FORM, type AgentForm } from "./Builder";
import { Result, type RunState } from "./Result";
import { Logo, Spinner, buttonStyles } from "./ui";

type Screen = "builder" | "result";

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [form, setForm] = useState<AgentForm>(EMPTY_FORM);
  const [screen, setScreen] = useState<Screen>("builder");
  const [run, setRun] = useState<RunState>({ status: "running" });
  const [scrollTo, setScrollTo] = useState<string | null>(null);

  const fetchCatalog = () => {
    setCatalogError(null);
    loadCatalog()
      .then(setCatalog)
      .catch((e: unknown) => setCatalogError(e instanceof ApiError ? e.message : "Couldn't load the app."));
  };
  useEffect(fetchCatalog, []);

  // Each screen change starts at the top (or at a requested section).
  useEffect(() => {
    if (scrollTo) {
      document.getElementById(scrollTo)?.scrollIntoView({ block: "start" });
      setScrollTo(null);
    } else {
      window.scrollTo(0, 0);
    }
  }, [screen]);

  const runAgent = async () => {
    setScreen("result");
    setRun({ status: "running" });
    try {
      const { presetId, requestId, ...input } = form;
      const recipeId = presetId ?? (await recipeIdFor({ ...input, name: input.name.trim(), prompt: input.prompt.trim() }));
      setRun({ status: "done", result: await runRecipe(recipeId, requestId) });
    } catch (e) {
      setRun({ status: "failed", error: e instanceof ApiError ? e.message : "Something went wrong. Please try again." });
    }
  };

  if (!catalog) {
    return (
      <div className="grid min-h-dvh place-items-center px-6 text-center">
        <div>
          <div className="mb-3 flex justify-center">
            <Logo />
          </div>
          {catalogError ? (
            <>
              <p className="mb-4 text-[15px]">{catalogError}</p>
              <button type="button" onClick={fetchCatalog} className={buttonStyles.primary}>
                Try again
              </button>
            </>
          ) : (
            <p className="flex items-center gap-2 text-muted">
              <Spinner /> Loading (the free server may take a moment to wake up)...
            </p>
          )}
        </div>
      </div>
    );
  }

  if (screen === "result") {
    return (
      <Result
        run={run}
        agentName={form.name.trim() || "Agent"}
        onBack={() => setScreen("builder")}
        onEdit={() => {
          setScrollTo("instructions");
          setScreen("builder");
        }}
        onRetry={runAgent}
      />
    );
  }

  return <Builder catalog={catalog} form={form} onChange={setForm} onRun={runAgent} />;
}
