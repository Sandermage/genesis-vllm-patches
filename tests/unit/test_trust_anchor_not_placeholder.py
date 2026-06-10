# SPDX-License-Identifier: Apache-2.0
"""Production trust-anchor gate.

Guards against shipping `sndr.license` with the
all-zeros placeholder Ed25519 public key. The placeholder mode
rejects every signed token as BAD_SIGNATURE, which silently
disables the engine-tier license entirely.
"""
from __future__ import annotations


def test_trust_anchor_is_not_placeholder():
    """The active pubkey must not be the 32-zero placeholder."""
    from sndr.license import _is_placeholder_anchor

    assert not _is_placeholder_anchor(), (
        "Trust anchor placeholder detected. Run the offline "
        "trust-anchor ceremony, then update "
        "`_TRUST_ANCHOR_PUBKEY_B64URL` in `vllm/sndr_core/license.py`."
    )


def test_trust_anchor_pubkey_shape():
    """Pubkey must be 43-char base64url-encoded (32 raw bytes)."""
    import base64
    from sndr.license import _TRUST_ANCHOR_PUBKEY_B64URL

    assert isinstance(_TRUST_ANCHOR_PUBKEY_B64URL, str), \
        "anchor must be str"
    assert len(_TRUST_ANCHOR_PUBKEY_B64URL) == 43, (
        f"Ed25519 pubkey base64url (no padding) is 43 chars; "
        f"got {len(_TRUST_ANCHOR_PUBKEY_B64URL)}"
    )
    pad = "=" * (-len(_TRUST_ANCHOR_PUBKEY_B64URL) % 4)
    raw = base64.urlsafe_b64decode(_TRUST_ANCHOR_PUBKEY_B64URL + pad)
    assert len(raw) == 32, f"raw key must be 32 bytes, got {len(raw)}"
