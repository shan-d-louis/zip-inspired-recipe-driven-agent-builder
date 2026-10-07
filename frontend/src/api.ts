// All server calls: plain fetch to /graphql. No GraphQL client library needed.

export type OutputFormat = "MARKDOWN" | "STRUCTURED";

export interface ToolInfo {
  name: string;
  description: string;
}

export interface Recipe {
  id: string;
  name: string;
  prompt: string;
  tools: string[];
  outputFormat: OutputFormat;
  includeCitations: boolean;
  isPreset: boolean;
  defaultRequestId: string | null;
}

export interface PurchaseRequest {
  id: string;
  title: string;
  type: string;
  vendor: string;
  amount: number;
  currency: string;
}

export interface RecipeInput {
  name: string;
  prompt: string;
  tools: string[];
  outputFormat: OutputFormat;
  includeCitations: boolean;
}

export interface Finding {
  severity: "high" | "medium" | "low" | "info";
  title: string;
  detail: string;
  source: string | null;
  quote: string | null;
  verified: boolean | null;
}

export interface Citation {
  source: string;
  quote: string;
  verified: boolean;
}

export interface TraceStep {
  step: number;
  tool: string;
  summary: string;
  durationMs: number;
}

export interface RunResult {
  runId: string | null;
  ok: boolean;
  cached: boolean;
  provider: string | null;
  mode: string;
  recipeName: string;
  markdown: string | null;
  findings: { summary: string; findings: Finding[] } | null;
  citations: Citation[];
  trace: TraceStep[];
  error: string | null;
}

/** An error whose message is safe and friendly to show to the visitor. */
export class ApiError extends Error {}

async function gql<T>(query: string, variables: Record<string, unknown> = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch("/graphql", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, variables }),
    });
  } catch {
    throw new ApiError("Couldn't reach the server. Check your connection and try again.");
  }
  if (!response.ok) {
    throw new ApiError(`The server had a problem (HTTP ${response.status}). Please try again.`);
  }
  const body = (await response.json()) as { data?: T; errors?: { message: string }[] };
  if (body.errors?.length) {
    throw new ApiError(body.errors[0].message); // the server only sends friendly messages
  }
  return body.data as T;
}

const RECIPE_FIELDS = "id name prompt tools outputFormat includeCitations isPreset defaultRequestId";

export interface Catalog {
  tools: ToolInfo[];
  recipes: Recipe[];
  requests: PurchaseRequest[];
}

export function loadCatalog(): Promise<Catalog> {
  return gql<Catalog>(`{
    tools { name description }
    recipes { ${RECIPE_FIELDS} }
    requests { id title type vendor amount currency }
  }`);
}

export async function runRecipe(recipeId: string, requestId: string): Promise<RunResult> {
  const data = await gql<{ runRecipe: RunResult }>(
    `mutation ($recipeId: ID!, $requestId: ID!) {
      runRecipe(recipeId: $recipeId, requestId: $requestId) {
        runId ok cached provider mode recipeName markdown error
        findings { summary findings { severity title detail source quote verified } }
        citations { source quote verified }
        trace { step tool summary durationMs }
      }
    }`,
    { recipeId, requestId },
  );
  return data.runRecipe;
}

// --- Custom recipe ids ---------------------------------------------------
// The server only lists presets, so this tab remembers the ids of recipes it
// created (keyed by their exact settings) to avoid creating duplicates.
// Kept in memory, mirrored to sessionStorage when the browser allows it.

const STORAGE_KEY = "agentblocks.recipeIds";
const knownIds: Record<string, string> = readStoredIds();

function readStoredIds(): Record<string, string> {
  try {
    return JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? "{}") as Record<string, string>;
  } catch {
    return {};
  }
}

function storeIds(): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(knownIds));
  } catch {
    // Private mode or blocked storage: the in-memory copy still works.
  }
}

async function recipeExists(id: string): Promise<boolean> {
  const data = await gql<{ recipe: { id: string } | null }>(`query ($id: ID!) { recipe(id: $id) { id } }`, { id });
  return data.recipe !== null;
}

/** The id of a recipe with exactly these settings, creating it if needed. */
export async function recipeIdFor(input: RecipeInput): Promise<string> {
  const key = JSON.stringify(input);
  const known = knownIds[key];
  // The server keeps recipes in memory, so they vanish on restart: check first.
  if (known && (await recipeExists(known))) return known;
  const data = await gql<{ createRecipe: { id: string } }>(
    `mutation ($input: RecipeInput!) { createRecipe(input: $input) { id } }`,
    { input },
  );
  knownIds[key] = data.createRecipe.id;
  storeIds();
  return data.createRecipe.id;
}
