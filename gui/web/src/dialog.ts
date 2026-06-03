// Shared modal-dialog accessibility hooks, used by both App and ContainersPanel
// so every role=dialog surface gets the same keyboard semantics without App and
// Containers importing each other (which would be circular).
import { useEffect, type RefObject } from "react";

// Traps Tab focus inside the dialog while it is open and restores focus to the
// previously-focused element (the trigger) on close. Respects an element that
// already holds focus inside the dialog (e.g. an autofocused input), so it never
// fights an explicit autoFocus. `active` lets callers that render the dialog
// inline (`{open && <section/>}`) re-arm the trap when it opens; components that
// mount/unmount can omit it.
export function useDialogFocus<T extends HTMLElement>(ref: RefObject<T | null>, active: boolean = true) {
  useEffect(() => {
    if (!active) return;
    const node = ref.current;
    if (!node) return;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const selector = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
    const focusables = () => Array.from(node.querySelectorAll<HTMLElement>(selector)).filter((el) => el.offsetParent !== null || el === document.activeElement);
    if (!node.contains(document.activeElement)) focusables()[0]?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const items = focusables();
      if (items.length === 0) { event.preventDefault(); return; }
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    node.addEventListener("keydown", onKeyDown);
    return () => {
      node.removeEventListener("keydown", onKeyDown);
      previouslyFocused?.focus?.();
    };
  }, [ref, active]);
}

// Calls `onClose` on Escape while the dialog is mounted. Pair with useDialogFocus
// for a fully keyboard-operable modal.
export function useEscapeKey(onClose: () => void, active: boolean = true) {
  useEffect(() => {
    if (!active) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, active]);
}
