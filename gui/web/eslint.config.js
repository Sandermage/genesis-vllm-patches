import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import jsxA11y from "eslint-plugin-jsx-a11y";
import globals from "globals";

// Flat ESLint config for the SNDR Control Center web app. Pragmatic rule set: a
// real CI gate for correctness (hooks, unused vars, obvious mistakes) without
// fighting the existing codebase's deliberate `any` on API-shaped records.
export default tseslint.config(
  { ignores: ["dist/**", "node_modules/**", "src/api/schema.gen.ts"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  jsxA11y.flatConfigs.recommended,
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      globals: { ...globals.browser },
    },
    plugins: { "react-hooks": reactHooks },
    rules: {
      // Classic, high-signal hook rules. The v6 "recommended" set adds opinionated
      // rules (e.g. set-state-in-effect) that flag the legitimate
      // `useEffect(() => { void load() }, [])` data-loading pattern, so we enable
      // the two that catch real bugs and keep exhaustive-deps advisory.
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      // The API exposes card/composed as Record<string, any>; typing them
      // fully is out of scope and not worth blocking the build over.
      "@typescript-eslint/no-explicit-any": "off",
      // Caught at build time by tsconfig noUnusedLocals; keep as a warning here
      // and allow the `_`-prefix convention for intentionally-unused args.
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_", caughtErrors: "none" },
      ],
      "no-empty": ["warn", { allowEmptyCatch: true }],
      // Pre-existing a11y interaction patterns (click handlers on non-interactive
      // elements, autofocus in modals) are surfaced as warnings to fix
      // incrementally rather than blocking the build on the whole existing
      // codebase. New genuinely-broken markup still shows up in lint output.
      "jsx-a11y/click-events-have-key-events": "warn",
      "jsx-a11y/no-static-element-interactions": "warn",
      "jsx-a11y/no-noninteractive-element-interactions": "warn",
      "jsx-a11y/interactive-supports-focus": "warn",
      "jsx-a11y/label-has-associated-control": "warn",
      // autoFocus is used deliberately and only inside modals / the login form
      // (focus the first field on open) — a recognised good-UX exception to the
      // generic rule, and awkward to disable per-attribute on multi-line inputs.
      "jsx-a11y/no-autofocus": "off",
    },
  },
  {
    // Tests may use non-null assertions and looser typing freely.
    files: ["**/*.{test,spec}.{ts,tsx}", "e2e/**/*.ts"],
    rules: {
      "@typescript-eslint/no-non-null-assertion": "off",
    },
  },
  {
    // Node dev scripts (screenshot sweep, etc.) — Node globals, not browser.
    files: ["scripts/**/*.{mjs,js,ts}"],
    languageOptions: { globals: { ...globals.node } },
  },
);
