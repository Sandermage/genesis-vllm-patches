# SPDX-License-Identifier: Apache-2.0
"""Package / pin / override policy dataclasses.

Hosts PackageSource / PackageSources / PackageVersions /
UpstreamPinPolicy / OverridesPolicy. All classes relocated from
``model_configs/schema.py`` in M.5.1; bodies unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ._base import SchemaError


@dataclass
class PackageSource:
    """Y2 (UNIFIED_CONFIG plan 2026-05-09): one channel/source declaration.

    Operators declare WHERE each runtime dependency comes from
    (distro repo / pip channel / source build / docker image / NVIDIA
    repo). Drives `sndr deps install` policy: refuse to `curl|bash`
    unless explicitly opted in, prefer official distro repos by default.

    Fields:
      - name: 'docker' | 'nvidia_container_toolkit' | 'python' | 'vllm' | ...
      - kind: 'distro_repo' | 'pip' | 'docker_image' | 'nvidia_repo' |
              'github_release' | 'source_build' | 'curl_pipe_bash'
      - channel: free-form (e.g. 'stable', 'nightly', 'main')
      - allow_third_party: True if non-official upstream sources are OK
        (e.g. unofficial Docker repo, pre-release pip indices)
      - notes: free-form
    """
    name: str
    kind: str
    channel: str = "stable"
    allow_third_party: bool = False
    notes: str = ""

    _VALID_KINDS = (
        "distro_repo", "pip", "docker_image", "nvidia_repo",
        "github_release", "source_build", "curl_pipe_bash",
    )

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SchemaError("PackageSource.name must be non-empty string")
        if self.kind not in self._VALID_KINDS:
            raise SchemaError(
                f"PackageSource.kind must be one of {self._VALID_KINDS} "
                f"(got {self.kind!r})"
            )
        # SAFETY: curl|bash is opt-in via allow_third_party=True and
        # never default. Document the risk for the operator.
        if self.kind == "curl_pipe_bash" and not self.allow_third_party:
            raise SchemaError(
                f"PackageSource.kind='curl_pipe_bash' for {self.name!r} "
                f"requires allow_third_party=True (explicit opt-in to "
                f"running an arbitrary upstream script as root)"
            )


@dataclass
class PackageSources:
    """Y2 container — list of declared sources, indexed by name."""
    sources: list[PackageSource] = field(default_factory=list)

    def validate(self) -> None:
        if not isinstance(self.sources, list):
            raise SchemaError("PackageSources.sources must be list")
        names = []
        for s in self.sources:
            s.validate()
            if s.name in names:
                raise SchemaError(
                    f"PackageSources.sources duplicate name {s.name!r}"
                )
            names.append(s.name)

    def get(self, name: str) -> Optional[PackageSource]:
        for s in self.sources:
            if s.name == name:
                return s
        return None


@dataclass
class UpstreamPinPolicy:
    """Y11 (UNIFIED_CONFIG plan 2026-05-09): per-config vLLM pin policy.

    Operators can declare which vLLM pins this config has been
    validated against (`allowed_pins`) and which pins are known to
    break it (`blocked_pins`). The launcher consults this BEFORE
    starting vllm; a blocked pin aborts with a precise error
    pointing at the relevant `notes` entry.

    `required_pin` is the strict equivalent of `vllm_pin_required` at
    the top level — when set, only that exact pin is allowed (subset
    of `allowed_pins`). Use this for stable / community-prod configs.

    Empty allowlist + empty blocklist = legacy "warn-only" behavior;
    KNOWN_GOOD_VLLM_PINS still enforces project-wide allowlist.
    """
    required_pin: Optional[str] = None
    allowed_pins: list[str] = field(default_factory=list)
    blocked_pins: list[str] = field(default_factory=list)
    notes: str = ""

    def validate(self) -> None:
        for name, lst in (("allowed_pins", self.allowed_pins),
                          ("blocked_pins", self.blocked_pins)):
            if not isinstance(lst, list):
                raise SchemaError(
                    f"UpstreamPinPolicy.{name} must be list[str]"
                )
            for p in lst:
                if not isinstance(p, str) or not p.strip():
                    raise SchemaError(
                        f"UpstreamPinPolicy.{name} entries must be "
                        f"non-empty strings (got {p!r})"
                    )
        # Cross-check: required_pin can't be in blocked_pins.
        if self.required_pin and self.required_pin in self.blocked_pins:
            raise SchemaError(
                f"UpstreamPinPolicy.required_pin {self.required_pin!r} "
                f"is also listed in blocked_pins"
            )
        # Overlap: allowed ∩ blocked must be empty.
        overlap = set(self.allowed_pins) & set(self.blocked_pins)
        if overlap:
            raise SchemaError(
                f"UpstreamPinPolicy: pins in both allowed_pins and "
                f"blocked_pins: {sorted(overlap)}"
            )

    def check(self, running_pin: Optional[str]) -> Optional[str]:
        """Returns a violation message string if `running_pin` is rejected.

        Returns None if the pin is allowed (or no policy is declared).
        Order of decision:
          1. blocked_pins → reject
          2. required_pin set → must equal it
          3. allowed_pins set → must be in the list
          4. otherwise → allow (defer to KNOWN_GOOD_VLLM_PINS)
        """
        if not running_pin:
            return None
        if running_pin in self.blocked_pins:
            note = f" — {self.notes}" if self.notes else ""
            return (
                f"vllm pin {running_pin!r} is in this config's "
                f"upstream.blocked_pins{note}"
            )
        if self.required_pin and running_pin != self.required_pin:
            return (
                f"vllm pin {running_pin!r} != upstream.required_pin "
                f"{self.required_pin!r}"
            )
        if self.allowed_pins and running_pin not in self.allowed_pins:
            return (
                f"vllm pin {running_pin!r} not in upstream.allowed_pins "
                f"{sorted(self.allowed_pins)}"
            )
        return None


@dataclass
class OverridesPolicy:
    """Y12 (UNIFIED_CONFIG plan 2026-05-09): runtime override safety.

    Operators can declare which env vars are safe to override at
    `sndr launch --override KEY=VAL` time, and what numeric ranges
    are acceptable. Safety: prevents an operator from setting
    `GENESIS_P67_NUM_KV_SPLITS=999` and silently destroying TPS, or
    from setting `GENESIS_PN16_TOOL_THINK_BUDGET=-1` and crashing
    the request middleware.

    `allow_env` is a list of env-var keys (regex-free literal match)
    that may be overridden. Vars not in the list are rejected.

    `safe_ranges` maps env-var key → (min_str, max_str). The launcher
    parses the override value as int OR float and rejects out-of-range.
    Strings in min/max so YAML parses naturally; coerced lazily.
    """
    allow_env: list[str] = field(default_factory=list)
    safe_ranges: dict[str, list[str]] = field(default_factory=dict)
    notes: str = ""

    def validate(self) -> None:
        if not isinstance(self.allow_env, list):
            raise SchemaError("OverridesPolicy.allow_env must be list[str]")
        for k in self.allow_env:
            if not isinstance(k, str) or not k.strip():
                raise SchemaError(
                    f"OverridesPolicy.allow_env entries must be non-empty "
                    f"strings (got {k!r})"
                )
        if not isinstance(self.safe_ranges, dict):
            raise SchemaError("OverridesPolicy.safe_ranges must be dict")
        for k, rng in self.safe_ranges.items():
            if not isinstance(rng, list) or len(rng) != 2:
                raise SchemaError(
                    f"OverridesPolicy.safe_ranges[{k!r}] must be a "
                    f"[min, max] 2-list (got {rng!r})"
                )
            for v in rng:
                try:
                    float(v)
                except (TypeError, ValueError) as e:
                    raise SchemaError(
                        f"OverridesPolicy.safe_ranges[{k!r}] bound "
                        f"{v!r} is not numeric"
                    ) from e

    def check(self, key: str, value: str) -> Optional[str]:
        """Returns violation msg if (key,value) override is rejected.

        Returns None if accepted. Order:
          1. allow_env is empty → reject (no overrides allowed)
          2. key not in allow_env → reject
          3. key in safe_ranges → value must parse as number AND lie in [min,max]
          4. key in allow_env but no range → accept (string-only override)
        """
        if not self.allow_env:
            return "overrides not enabled (allow_env is empty)"
        if key not in self.allow_env:
            return (
                f"override key {key!r} not in allow_env "
                f"(allowed: {sorted(self.allow_env)})"
            )
        if key in self.safe_ranges:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return (
                    f"override {key}={value!r} not numeric — range "
                    f"{self.safe_ranges[key]} requires a number"
                )
            lo, hi = float(self.safe_ranges[key][0]), float(self.safe_ranges[key][1])
            if not (lo <= v <= hi):
                return (
                    f"override {key}={value!r} outside safe range "
                    f"[{lo}, {hi}]"
                )
        return None


@dataclass
class PackageVersions:
    """Y1 (UNIFIED_CONFIG plan 2026-05-09): in-container package pins.

    Operators declare the runtime python packages the container needs
    (alongside vLLM itself). The renderer honors `python_packages`
    when SNDR_DEV_INSTALL_RUNTIME_DEPS=1 is set inside the container,
    rather than the renderer hardcoding versions in a string literal
    (B6 fix: previous renderer baked `pandas==2.2.3 scipy==1.14.1
    xxhash==3.5.0` into every config).

    All fields optional. If `python_packages` is empty/None, the
    renderer falls back to the legacy hardcoded baseline so existing
    YAML configs that don't declare this block keep working.

    Future blocks (planned per UNIFIED_CONFIG_AUTOMATION_PLAN §Y1):
      - vllm:           {channel: stable|tested|nightly|local, version}
      - torch:          {version}
      - flashinfer:     {version, source}
      - triton:         {version}
      - transformers:   {version}
    """
    python_packages: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def validate(self) -> None:
        for name, pin in self.python_packages.items():
            if not isinstance(name, str) or not name.strip():
                raise SchemaError(
                    f"PackageVersions.python_packages keys must be non-empty "
                    f"strings (got {name!r})"
                )
            if not isinstance(pin, str) or not pin.strip():
                raise SchemaError(
                    f"PackageVersions.python_packages[{name!r}] must be a "
                    f"non-empty version string (got {pin!r})"
                )
            # Operators must pin exactly — supply chain integrity.
            # Allow `==`, `===`, or bare version (we add `==` if bare).
            # Reject bare ranges like `>=2.0`.
            stripped = pin.strip()
            if any(stripped.startswith(op) for op in (">=", "<=", ">", "<", "~=")):
                raise SchemaError(
                    f"PackageVersions.python_packages[{name!r}]={pin!r} "
                    f"must be an exact pin (use 'X.Y.Z' or '==X.Y.Z'); "
                    f"version ranges are not allowed in production configs."
                )

    def to_pip_args(self) -> str:
        """Render as space-joined `name==version` arguments for `pip install`.

        Empty dict → empty string (renderer treats as "fallback to legacy").
        Bare version values get `==` prefix; explicit `==X.Y` passed through.
        """
        if not self.python_packages:
            return ""
        parts: list[str] = []
        for name, pin in self.python_packages.items():
            stripped = pin.strip()
            if stripped.startswith("=="):
                parts.append(f"{name}{stripped}")
            else:
                parts.append(f"{name}=={stripped}")
        return " ".join(parts)
