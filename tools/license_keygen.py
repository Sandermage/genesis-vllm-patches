#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SNDR Engine license tooling — generate keypair + sign tokens.

The public Genesis distribution ships a development-only Ed25519
anchor in `vllm.sndr_core.license` so nothing in the public repo
verifies real customer tokens. This tool is the canonical workflow
to rotate the development anchor to a production anchor AND mint
signed tokens for license holders.

Two modes:

  1. `python tools/license_keygen.py generate-keypair`
     Generate a fresh Ed25519 keypair. Writes:
       - `private_key.pem`      (offline, NEVER ship in repo)
       - `public_key.b64url`    (32 raw bytes, base64url-encoded; this
                                 string is what gets pasted into
                                 `vllm/sndr_core/license.py:_TRUST_ANCHOR_PUBKEY_B64URL`)

  2. `python tools/license_keygen.py sign-token \\
         --private-key private_key.pem \\
         --customer-id sander-test \\
         --expires-in-days 365 \\
         --engine-major 11`
     Mint a signed token. Output is `<base64url-payload>.<base64url-signature>`
     ready to drop into `SNDR_ENGINE_LICENSE_KEY` env or
     `~/.sndr/license.json` `{"key": "..."}`.

Workflow when commercial overlay is ready to ship:

  1. Run `generate-keypair` ONCE on an offline machine.
  2. Commit `public_key.b64url` content into `license.py`.
  3. Tag the release; ship the wheel.
  4. For each license holder, mint a token via `sign-token` on the
     offline rig and deliver it out-of-band (signed email, PGP, etc.).
  5. Holder pastes token into `SNDR_ENGINE_LICENSE_KEY`.
  6. License gate verifies signature against the public key bundled
     in the wheel; expired / tampered / wrong-major tokens reject
     fail-closed.

Requires: `pip install cryptography`. Genesis itself does NOT depend
on `cryptography` at runtime — the optional verification path imports
it lazily. This tool always requires it because keygen IS the workflow.

Author: Sandermage(Sander)-Barzov Aleksandr.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _require_cryptography():
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        sys.stderr.write(
            "ERROR: this tool needs `cryptography`. Run:\n"
            "       pip install cryptography\n"
        )
        sys.exit(2)
    return Ed25519PrivateKey, Ed25519PublicKey, serialization


def cmd_generate_keypair(args: argparse.Namespace) -> int:
    Ed25519PrivateKey, _Pub, serialization = _require_cryptography()
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key()

    # Private key — PEM, encrypted optional. Default unencrypted; the
    # operator can pass --passphrase to enable PEM-level encryption.
    if args.passphrase:
        algo = serialization.BestAvailableEncryption(args.passphrase.encode())
    else:
        algo = serialization.NoEncryption()
    private_pem = sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=algo,
    )

    # Public key — raw 32 bytes, base64url. This is the format
    # `vllm.sndr_core.license` reads.
    public_raw = pk.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_b64url = _b64url_encode(public_raw)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "private_key.pem").write_bytes(private_pem)
    (out_dir / "public_key.b64url").write_text(public_b64url + "\n")

    print(f"✓ private key: {out_dir / 'private_key.pem'}")
    print(f"✓ public key:  {out_dir / 'public_key.b64url'}")
    print()
    print("public_key (paste into vllm/sndr_core/license.py):")
    print(f"  {public_b64url}")
    print()
    print("⚠ Keep private_key.pem OFFLINE. Never commit. Never push.")
    print("⚠ Add the directory to .gitignore if it lives in the repo.")
    return 0


def cmd_sign_token(args: argparse.Namespace) -> int:
    Ed25519PrivateKey, _Pub, serialization = _require_cryptography()
    pem = Path(args.private_key).read_bytes()
    pwd = args.passphrase.encode() if args.passphrase else None
    sk = serialization.load_pem_private_key(pem, password=pwd)

    now = int(time.time())
    payload = {
        "customer_id": args.customer_id,
        "issued_at": now,
        "expires_at": now + args.expires_in_days * 86400,
        "engine_major": args.engine_major,
    }
    if args.notes:
        payload["notes"] = args.notes

    payload_bytes = json.dumps(payload, sort_keys=True).encode("ascii")
    payload_b64 = _b64url_encode(payload_bytes)
    sig = sk.sign(payload_b64.encode("ascii"))
    sig_b64 = _b64url_encode(sig)

    token = f"{payload_b64}.{sig_b64}"
    print(token)
    sys.stderr.write(
        f"\n✓ token issued for customer_id={args.customer_id}\n"
        f"✓ expires {time.strftime('%Y-%m-%d', time.localtime(payload['expires_at']))} "
        f"(in {args.expires_in_days} days)\n"
        f"✓ engine major: {args.engine_major}\n"
    )
    return 0


def cmd_verify_token(args: argparse.Namespace) -> int:
    """Verify a token using the bundled trust anchor (or an explicit key).

    Useful for testing: round-trip generate→sign→verify before shipping.
    """
    from cryptography.exceptions import InvalidSignature
    _, Ed25519PublicKey, _ = _require_cryptography()

    if args.public_key:
        # Read base64url string from file or arg
        p = Path(args.public_key)
        b64 = p.read_text().strip() if p.is_file() else args.public_key
    else:
        from vllm.sndr_core.license import _TRUST_ANCHOR_PUBKEY_B64URL
        b64 = _TRUST_ANCHOR_PUBKEY_B64URL

    pad = "=" * (-len(b64) % 4)
    pubkey_bytes = base64.urlsafe_b64decode(b64 + pad)
    pubkey = Ed25519PublicKey.from_public_bytes(pubkey_bytes)

    payload_b64, sig_b64 = args.token.split(".", 1)
    sig = base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4))
    try:
        pubkey.verify(sig, payload_b64.encode("ascii"))
    except InvalidSignature:
        sys.stderr.write("✗ signature INVALID\n")
        return 1
    payload = json.loads(base64.urlsafe_b64decode(
        payload_b64 + "=" * (-len(payload_b64) % 4)
    ))
    print(json.dumps(payload, indent=2))
    sys.stderr.write("\n✓ signature valid\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate-keypair",
                       help="Generate a fresh Ed25519 keypair.")
    g.add_argument("--out-dir", default=".",
                   help="Directory to write key files into (default: cwd).")
    g.add_argument("--passphrase", default=None,
                   help="Optional passphrase to encrypt the private PEM.")
    g.set_defaults(func=cmd_generate_keypair)

    s = sub.add_parser("sign-token", help="Mint a signed license token.")
    s.add_argument("--private-key", required=True,
                   help="Path to PEM private key from generate-keypair.")
    s.add_argument("--passphrase", default=None,
                   help="Passphrase for an encrypted private PEM.")
    s.add_argument("--customer-id", required=True,
                   help="Customer identifier embedded in the token payload.")
    s.add_argument("--expires-in-days", type=int, default=365,
                   help="Token validity window in days (default: 365).")
    s.add_argument("--engine-major", type=int, required=True,
                   help="Engine major version this token is bound to "
                        "(e.g. 11 for sndr_engine 11.x).")
    s.add_argument("--notes", default=None,
                   help="Free-form note to embed in the payload.")
    s.set_defaults(func=cmd_sign_token)

    v = sub.add_parser("verify-token", help="Verify a token signature.")
    v.add_argument("token", help="The full <payload>.<signature> token string.")
    v.add_argument("--public-key", default=None,
                   help="Path to base64url public key file or the literal "
                        "string. Defaults to the bundled trust anchor.")
    v.set_defaults(func=cmd_verify_token)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
