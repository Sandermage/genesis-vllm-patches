// SPDX-License-Identifier: Apache-2.0
// Generic labelled form-field primitives shared by the config editors: text,
// number, boolean toggle (aria-pressed) and select.

export function TextField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="param-field">
      <span>{label}</span>
      <input value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

export function NumberField({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return (
    <label className="param-field">
      <span>{label}</span>
      <input type="number" value={value} onChange={(event) => { const n = Number(event.target.value); if (Number.isFinite(n)) onChange(n); }} />
    </label>
  );
}

export function BoolField({ label, value, onChange }: { label: string; value: boolean; onChange: (value: boolean) => void }) {
  return (
    <button className="bool-field" onClick={() => onChange(!value)} aria-pressed={value}>
      <span>{label}</span>
      <i className={value ? "active" : ""} />
    </button>
  );
}

export function SelectField({
  label,
  value,
  options,
  onChange
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="param-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </label>
  );
}
