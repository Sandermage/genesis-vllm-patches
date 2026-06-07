// SPDX-License-Identifier: Apache-2.0
// Layer editor — load a model/hardware/profile/preset definition, edit its
// curated + discovered fields with a live YAML mirror, and save an operator-local
// copy. Extracted from App.tsx (modularization) with no behavior change.
 
import { useEffect, useMemo, useState } from "react";
import { AlertCircle, CheckCircle2, Code2, PackageCheck } from "lucide-react";
import { api, type V2LayerApplyResult } from "../api";
import { type ElementKind, ELEMENT_FIELDS, discoverExtraFields, groupFields, ElementField } from "./element-fields";
import { getIn, setIn, objToYaml } from "../lib/config-utils";
import { CodeBlock } from "../components/code-block";

export function LayerEditor({ kind, layerId }: { kind: ElementKind; layerId: string }) {
  const [edited, setEdited] = useState<Record<string, any> | null>(null);
  const [source, setSource] = useState("");
  const [state, setState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [applyResult, setApplyResult] = useState<V2LayerApplyResult | null>(null);
  const [applying, setApplying] = useState(false);

  useEffect(() => {
    if (!layerId) { setEdited(null); return; }
    const controller = new AbortController();
    setState("loading");
    setError(null);
    setApplyResult(null);
    api.v2Layer(kind, layerId, controller.signal)
      .then((layer) => {
        if (controller.signal.aborted) return;
        setEdited(layer.definition as Record<string, any>);
        setSource(layer.source);
        setState("ready");
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setEdited(null);
        setState("error");
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => controller.abort();
  }, [kind, layerId]);

  const fields = useMemo(() => {
    const curated = ELEMENT_FIELDS[kind];
    if (!edited) return curated;
    const known = new Set(curated.map((spec) => spec.path));
    return [...curated, ...discoverExtraFields(edited, known)];
  }, [kind, edited]);
  const yaml = edited ? objToYaml(edited) : [`# Loading ${kind}…`];

  async function runSave() {
    if (!edited) return;
    setApplying(true);
    try {
      const result = await api.v2LayerApply({ kind, layer_id: layerId, yaml_text: yaml.join("\n") + "\n" });
      setApplyResult(result);
    } catch (err) {
      setApplyResult({
        kind, layer_id: layerId, target_path: "", action: "create", written: false,
        bytes_written: 0, status: "blocked", message: err instanceof Error ? err.message : String(err), blocked_reasons: []
      });
    } finally {
      setApplying(false);
    }
  }

  return (
    <div className="preset-editor">
      <p className="element-source">{source || (state === "loading" ? "loading…" : "")}</p>
      {error && <div className="config-plan-error"><AlertCircle size={15} /><span>{error}</span></div>}
      {edited ? (
        <div className="preset-editor-cols">
          <div className="element-groups">
            {groupFields(fields).map(([group, list]) => (
              <div className="element-group" key={group}>
                {group && <div className="element-group-head">{group}</div>}
                <div className="element-fields">
                  {list.map((spec) => (
                    <ElementField
                      key={spec.path}
                      spec={spec}
                      value={getIn(edited, spec.path)}
                      onChange={(value) => setEdited((current) => (current ? setIn(current, spec.path, value) : current))}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="preset-editor-yaml">
            <div className="config-panel-title"><Code2 size={16} /><strong>{kind}.yaml</strong><span>{yaml.length} lines</span></div>
            <CodeBlock lines={yaml} />
          </div>
        </div>
      ) : (
        <p className="muted">Select a {kind} to edit.</p>
      )}
      {applyResult && (
        <div className={`element-apply ${applyResult.status}`}>
          <span className="finding-icon">
            {applyResult.status === "applied" ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
          </span>
          <div>
            <strong>{applyResult.status === "applied" ? "Saved to user dir" : applyResult.status}</strong>
            <small>{applyResult.target_path || applyResult.message}</small>
          </div>
        </div>
      )}
      <div className="config-actions">
        <span className="config-actions-note">Edits write an operator-local {kind} copy (never the builtin)</span>
        <button className="primary-action" onClick={() => void runSave()} disabled={!edited || applying}>
          <PackageCheck size={14} /> {applying ? "Saving…" : "Save to user dir"}
        </button>
      </div>
    </div>
  );
}
