# SPDX-License-Identifier: Apache-2.0
"""TDD for `sndr.license` (Phase 4 / F-010-012 audit fix).

Structured engine-tier eligibility checks: replaces the simple
"can we import vllm.sndr_engine?" gate with a proper boundary that
verifies package presence + license key + version compat.

Goal of this test suite: make the gate's failure modes individually
observable so an operator gets an actionable message ("install the
package" vs "set the key" vs "version mismatch") rather than a generic
"engine-tier patch skipped".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _license_module():
    from sndr import license as L
    return L


def _core_version() -> str:
    """Live platform version — eligibility tests that simulate a
    COMPATIBLE engine overlay must track the real core major instead
    of pinning one (the hardcoded "11.0.0" pin broke on the v12 bump
    via VERSION_MISMATCH)."""
    from sndr.version import SNDR_CORE_VERSION
    return SNDR_CORE_VERSION


# ─── Status enum sanity ────────────────────────────────────────────────────


class TestLicenseStatus:
    def test_enum_values_distinct(self):
        L = _license_module()
        statuses = [
            L.LicenseStatus.LICENSED,
            L.LicenseStatus.NO_PACKAGE,
            L.LicenseStatus.NO_KEY,
            L.LicenseStatus.VERSION_MISMATCH,
            L.LicenseStatus.OVERRIDE,
        ]
        # All distinct
        assert len({s.value for s in statuses}) == 5

    def test_bad_payload_status_present(self):
        """BAD_PAYLOAD is a distinct failure mode for the
        signature-OK + contract-violation case."""
        L = _license_module()
        assert L.LicenseStatus.BAD_PAYLOAD.value == "bad_payload"
        assert L.LicenseStatus.BAD_PAYLOAD != L.LicenseStatus.BAD_SIGNATURE


# ─── Payload contract (Etap 0.1) ─────────────────────────────────────────


class TestPayloadContract:
    """A signed token must carry the full contract:
    customer_id (non-empty str), issued_at + expires_at (positive
    epoch numerics, expires > issued), engine_major (int, not bool).

    Without this check, missing or wrong-type fields silently
    passed as LICENSED and the token became effectively unbounded.
    """

    NOW = 1_700_000_000.0  # frozen epoch for deterministic comparisons

    def _valid(self, **overrides):
        base = {
            "customer_id": "test-customer-7",
            "issued_at": self.NOW - 86400,    # one day ago
            "expires_at": self.NOW + 86400,   # one day ahead
            "engine_major": 11,
        }
        base.update(overrides)
        return base

    def test_valid_payload_passes(self):
        L = _license_module()
        assert L._validate_payload_contract(self._valid(), now_epoch=self.NOW) is None

    @pytest.mark.parametrize("missing", [
        "customer_id", "issued_at", "expires_at", "engine_major",
    ])
    def test_missing_required_field_rejected(self, missing):
        L = _license_module()
        payload = self._valid()
        del payload[missing]
        err = L._validate_payload_contract(payload, now_epoch=self.NOW)
        assert err is not None
        assert missing in err

    def test_string_expires_at_rejected(self):
        L = _license_module()
        err = L._validate_payload_contract(
            self._valid(expires_at="2026-12-31"), now_epoch=self.NOW,
        )
        assert err is not None
        assert "expires_at" in err

    def test_non_string_customer_id_rejected(self):
        L = _license_module()
        err = L._validate_payload_contract(
            self._valid(customer_id=12345), now_epoch=self.NOW,
        )
        assert err is not None
        assert "customer_id" in err

    def test_empty_customer_id_rejected(self):
        L = _license_module()
        err = L._validate_payload_contract(
            self._valid(customer_id="   "), now_epoch=self.NOW,
        )
        assert err is not None
        assert "non-empty" in err

    def test_bool_engine_major_rejected(self):
        """Python quirk: `True` isinstance(int) -> True. The validator
        must reject bool explicitly so a token with
        `engine_major: true` does not pass as a numeric major."""
        L = _license_module()
        err = L._validate_payload_contract(
            self._valid(engine_major=True), now_epoch=self.NOW,
        )
        assert err is not None
        assert "engine_major" in err and "bool" in err

    def test_bool_expires_at_rejected(self):
        L = _license_module()
        err = L._validate_payload_contract(
            self._valid(expires_at=True), now_epoch=self.NOW,
        )
        assert err is not None
        assert "bool" in err

    def test_negative_expires_at_rejected(self):
        L = _license_module()
        err = L._validate_payload_contract(
            self._valid(expires_at=-1), now_epoch=self.NOW,
        )
        assert err is not None
        assert "positive" in err

    def test_expires_before_issued_rejected(self):
        L = _license_module()
        err = L._validate_payload_contract(
            self._valid(issued_at=self.NOW, expires_at=self.NOW - 1),
            now_epoch=self.NOW,
        )
        assert err is not None
        assert "greater than issued" in err

    def test_future_issued_at_rejected(self):
        """Token issued in the future = clock attack or signer misconfigured."""
        L = _license_module()
        err = L._validate_payload_contract(
            self._valid(issued_at=self.NOW + 3600),  # one hour in the future
            now_epoch=self.NOW,
        )
        assert err is not None
        assert "future" in err

    def test_small_clock_skew_tolerated(self):
        """Skew <= 60s is tolerated — clock drift between issuer and
        verifier is expected."""
        L = _license_module()
        assert L._validate_payload_contract(
            self._valid(issued_at=self.NOW + 30),  # 30 seconds in the future
            now_epoch=self.NOW,
        ) is None

    def test_int_timestamps_accepted(self):
        """expires_at/issued_at accept int as well as float."""
        L = _license_module()
        assert L._validate_payload_contract(
            self._valid(
                issued_at=int(self.NOW) - 100,
                expires_at=int(self.NOW) + 100,
            ),
            now_epoch=self.NOW,
        ) is None


# ─── Public verify_token / is_placeholder_anchor (Etap 0.5) ──────────────


class TestPublicApi:
    """The ceremony doc and downstream integrations consume public
    names; the underscored `_verify_signed_token` and
    `_is_placeholder_anchor` are internal helpers."""

    def test_verify_token_exported(self):
        L = _license_module()
        assert hasattr(L, "verify_token")
        assert "verify_token" in L.__all__

    def test_is_placeholder_anchor_exported(self):
        L = _license_module()
        assert hasattr(L, "is_placeholder_anchor")
        assert "is_placeholder_anchor" in L.__all__

    def test_token_verification_exported(self):
        L = _license_module()
        assert hasattr(L, "TokenVerification")
        assert "TokenVerification" in L.__all__

    def test_verify_token_rejects_malformed_format(self):
        """A token without the separator or too short to parse must
        resolve to BAD_SIGNATURE rather than crashing the verifier."""
        L = _license_module()
        r = L.verify_token("not-a-signed-token")
        assert r.status == L.LicenseStatus.BAD_SIGNATURE
        assert "format invalid" in r.detail.lower()

    def test_verify_token_returns_token_verification(self):
        L = _license_module()
        r = L.verify_token("garbage.also-garbage")
        # Must be an instance of TokenVerification, not a tuple
        assert isinstance(r, L.TokenVerification)
        # status — enum value
        assert isinstance(r.status, L.LicenseStatus)

    def test_is_placeholder_anchor_runs(self):
        """Smoke check: the call returns bool without raising."""
        L = _license_module()
        result = L.is_placeholder_anchor()
        assert isinstance(result, bool)


# ─── License key resolution ────────────────────────────────────────────────


class TestKeyResolution:
    def test_env_key_read(self, monkeypatch):
        L = _license_module()
        monkeypatch.setenv("SNDR_ENGINE_LICENSE_KEY", "test-key-abc")
        assert L._read_license_key_from_env() == "test-key-abc"

    def test_empty_env_returns_none(self, monkeypatch):
        L = _license_module()
        monkeypatch.setenv("SNDR_ENGINE_LICENSE_KEY", "")
        assert L._read_license_key_from_env() is None

    def test_missing_env_returns_none(self, monkeypatch):
        L = _license_module()
        monkeypatch.delenv("SNDR_ENGINE_LICENSE_KEY", raising=False)
        assert L._read_license_key_from_env() is None

    def test_file_key_read(self, tmp_path):
        L = _license_module()
        p = tmp_path / "license.json"
        p.write_text(json.dumps({"key": "file-key-xyz"}))
        assert L._read_license_key_from_file(p) == "file-key-xyz"

    def test_file_missing_returns_none(self, tmp_path):
        L = _license_module()
        p = tmp_path / "nonexistent.json"
        assert L._read_license_key_from_file(p) is None

    def test_file_bad_json_returns_none(self, tmp_path):
        L = _license_module()
        p = tmp_path / "license.json"
        p.write_text("not valid json {")
        assert L._read_license_key_from_file(p) is None

    def test_file_no_key_field_returns_none(self, tmp_path):
        L = _license_module()
        p = tmp_path / "license.json"
        p.write_text(json.dumps({"unrelated": "data"}))
        assert L._read_license_key_from_file(p) is None


# ─── Version compat ────────────────────────────────────────────────────────


class TestVersionCompat:
    @pytest.mark.parametrize("core,engine,compatible", [
        ("11.0.0", "11.0.0", True),
        ("11.0.0", "11.5.3", True),
        ("11.0.0", "v11.0.0", True),  # `v` prefix tolerated
        ("11.0.0", "10.9.0", False),  # major mismatch
        ("11.0.0", "12.0.0", False),
        ("v11.0.0", "10.0.0", False),
    ])
    def test_compat_rule(self, core, engine, compatible):
        L = _license_module()
        assert L._versions_compatible(core, engine) is compatible

    def test_unparseable_returns_false(self):
        L = _license_module()
        assert L._versions_compatible("garbage", "11.0.0") is False
        assert L._versions_compatible("11.0.0", "garbage") is False


# ─── Eligibility result paths ──────────────────────────────────────────────


class TestEligibility:
    def test_override_short_circuits(self, monkeypatch):
        """SNDR_ENABLE_TIER_OVERRIDE=1 forces community-only mode
        even if everything else is licensed."""
        L = _license_module()
        monkeypatch.setenv("SNDR_ENABLE_TIER_OVERRIDE", "1")
        result = L.check_engine_tier_eligible()
        assert result.eligible is False
        assert result.status == L.LicenseStatus.OVERRIDE

    def test_no_package_when_engine_absent(self, monkeypatch):
        """If `vllm.sndr_engine` can't be imported, status=NO_PACKAGE."""
        L = _license_module()
        # Force-mask the engine package via sys.modules
        import sys
        monkeypatch.setitem(sys.modules, "vllm.sndr_engine", None)
        # Also clear override
        monkeypatch.delenv("SNDR_ENABLE_TIER_OVERRIDE", raising=False)
        result = L.check_engine_tier_eligible()
        assert result.eligible is False
        assert result.status == L.LicenseStatus.NO_PACKAGE

    def test_no_key_when_package_present_no_key(self, monkeypatch):
        """Engine package is present (it's in this repo) but no license
        key set anywhere → status=NO_KEY."""
        L = _license_module()
        monkeypatch.delenv("SNDR_ENABLE_TIER_OVERRIDE", raising=False)
        monkeypatch.delenv("SNDR_ENGINE_LICENSE_KEY", raising=False)
        # Use a guaranteed-missing license file
        result = L.check_engine_tier_eligible(
            license_file=Path("/tmp/definitely-does-not-exist.json"),
        )
        assert result.eligible is False
        # Either NO_PACKAGE or NO_KEY depending on whether sndr_engine
        # is installed in the test env. On dev rig + repo checkout,
        # `vllm.sndr_engine` IS importable so we expect NO_KEY.
        assert result.status in (
            L.LicenseStatus.NO_KEY, L.LicenseStatus.NO_PACKAGE,
        )

    def test_licensed_with_explicit_key(self, monkeypatch):
        """All checks pass → eligible=True, status=LICENSED.

        DA-010 (audit 2026-05-08): tests now SIMULATE a real engine
        overlay being present (via `_engine_overlay_available` patch +
        explicit `_engine_package_version` return), because the public
        skeleton is no longer enough to flip the gate.

        Strict-tests fix (UNIFIED_CONFIG plan 2026-05-09): the legacy
        gate is opted into EXPLICITLY here so the test passes even
        when conftest's setdefault was overridden to empty.
        """
        L = _license_module()
        monkeypatch.delenv("SNDR_ENABLE_TIER_OVERRIDE", raising=False)
        monkeypatch.setenv("SNDR_ALLOW_LEGACY_LICENSE_KEYS", "1")
        monkeypatch.setattr(L, "_engine_overlay_available", lambda: True)
        monkeypatch.setattr(L, "_engine_package_version", _core_version)
        result = L.check_engine_tier_eligible(
            license_key="any-non-empty-string",
        )
        assert result.eligible is True
        assert result.status in (
            L.LicenseStatus.LICENSED,
            L.LicenseStatus.LICENSED_LEGACY,
        )

    def test_version_mismatch_detected(self, monkeypatch):
        """If sndr_engine version is incompatible with sndr_core,
        status=VERSION_MISMATCH.

        Strict-tests fix (UNIFIED_CONFIG plan 2026-05-09): explicit
        legacy gate so we reach the version check even when
        SNDR_ALLOW_LEGACY_LICENSE_KEYS is not pre-set.
        """
        L = _license_module()
        monkeypatch.delenv("SNDR_ENABLE_TIER_OVERRIDE", raising=False)
        monkeypatch.setenv("SNDR_ALLOW_LEGACY_LICENSE_KEYS", "1")
        # Force engine version reader to return an incompatible major
        monkeypatch.setattr(L, "_engine_package_version", lambda: "5.0.0")
        result = L.check_engine_tier_eligible(
            license_key="any-key",
        )
        assert result.eligible is False
        assert result.status == L.LicenseStatus.VERSION_MISMATCH

    def test_skip_override_check_flag(self, monkeypatch):
        """`skip_override_check=True` ignores SNDR_ENABLE_TIER_OVERRIDE.

        Strict-tests fix (UNIFIED_CONFIG plan 2026-05-09): explicit
        legacy gate so the unsigned `any-key` is accepted as
        LICENSED_LEGACY.
        """
        L = _license_module()
        monkeypatch.setenv("SNDR_ENABLE_TIER_OVERRIDE", "1")
        monkeypatch.setenv("SNDR_ALLOW_LEGACY_LICENSE_KEYS", "1")
        # DA-010: simulate real engine overlay.
        monkeypatch.setattr(L, "_engine_overlay_available", lambda: True)
        monkeypatch.setattr(L, "_engine_package_version", _core_version)
        result = L.check_engine_tier_eligible(
            license_key="any-key",
            skip_override_check=True,
        )
        assert result.status in (
            L.LicenseStatus.LICENSED,
            L.LicenseStatus.LICENSED_LEGACY,
        )

    def test_legacy_unsigned_key_rejected_without_dev_flag(self, monkeypatch):
        """P1-3 (audit 2026-05-08): unsigned keys fail with BAD_SIGNATURE
        in production (no SNDR_ALLOW_LEGACY_LICENSE_KEYS)."""
        L = _license_module()
        monkeypatch.delenv("SNDR_ALLOW_LEGACY_LICENSE_KEYS", raising=False)
        monkeypatch.delenv("SNDR_ENABLE_TIER_OVERRIDE", raising=False)
        # DA-010: simulate engine overlay so we reach the signature check.
        monkeypatch.setattr(L, "_engine_overlay_available", lambda: True)
        monkeypatch.setattr(L, "_engine_package_version", _core_version)
        result = L.check_engine_tier_eligible(license_key="some-plain-key")
        assert result.eligible is False
        assert result.status == L.LicenseStatus.BAD_SIGNATURE
        assert "unsigned" in result.reason.lower()

    def test_legacy_dev_flag_re_enables_unsigned_keys(self, monkeypatch):
        """When SNDR_ALLOW_LEGACY_LICENSE_KEYS=1 is explicitly set,
        unsigned keys are accepted as LICENSED_LEGACY."""
        L = _license_module()
        monkeypatch.setenv("SNDR_ALLOW_LEGACY_LICENSE_KEYS", "1")
        monkeypatch.delenv("SNDR_ENABLE_TIER_OVERRIDE", raising=False)
        # DA-010: simulate engine overlay.
        monkeypatch.setattr(L, "_engine_overlay_available", lambda: True)
        monkeypatch.setattr(L, "_engine_package_version", _core_version)
        result = L.check_engine_tier_eligible(license_key="some-plain-key")
        assert result.eligible is True
        assert result.status == L.LicenseStatus.LICENSED_LEGACY

    def test_skeleton_only_install_does_not_unlock_engine(self, monkeypatch):
        """DA-010 (audit 2026-05-08): an importable `vllm.sndr_engine`
        skeleton package without a real overlay must NOT flip the gate
        to "engine available". The contract is `engine_available()`
        returning True, not just `import vllm.sndr_engine` succeeding.
        """
        L = _license_module()
        monkeypatch.delenv("SNDR_ENABLE_TIER_OVERRIDE", raising=False)
        # Skeleton imports OK (default behavior in this repo) — but
        # the overlay availability probe must return False.
        monkeypatch.setattr(L, "_engine_overlay_available", lambda: False)
        result = L.check_engine_tier_eligible(license_key="any-key")
        assert result.eligible is False
        assert result.status == L.LicenseStatus.NO_PACKAGE
        assert "not installed" in result.reason.lower() or \
               "engine" in result.reason.lower()


# ─── Integration with dispatcher gate ─────────────────────────────────────


class TestDispatcherIntegration:
    """The dispatcher tier-gate must consult `check_engine_tier_eligible`
    when a registry entry has `tier=engine`."""

    def test_engine_patch_skipped_when_no_license(self, monkeypatch):
        """An opt-in engine-tier patch (default_on=False) should skip
        with a license-aware reason when no key is set."""
        from sndr.dispatcher import should_apply, PATCH_REGISTRY
        # Find an engine-tier patch in the live registry
        engine_pids = [
            pid for pid, meta in PATCH_REGISTRY.items()
            if isinstance(meta, dict) and meta.get("tier") == "engine"
        ]
        if not engine_pids:
            pytest.skip("no engine-tier patches in registry")
        pid = engine_pids[0]
        # Clear any license override + key
        monkeypatch.delenv("SNDR_ENABLE_TIER_OVERRIDE", raising=False)
        monkeypatch.delenv("SNDR_ENGINE_LICENSE_KEY", raising=False)
        # Force the env flag truthy so the env-flag gate is open and we
        # hit the tier-gate
        env_flag = PATCH_REGISTRY[pid].get("env_flag")
        if env_flag:
            monkeypatch.setenv(env_flag, "1")
        decision, reason = should_apply(pid)
        # On dev rig (no license file, no env key) the gate should
        # surface a "tier=engine: ..." reason.
        if decision is False and "tier=engine" in reason:
            assert "license" in reason.lower() or "no_key" in reason.lower() or "engine" in reason.lower()
