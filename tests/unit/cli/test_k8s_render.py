# SPDX-License-Identifier: Apache-2.0
"""S3.3 closure (audit P3-3, 2026-05-12): tests for the k8s YAML renderer.

Cover:

  • Existing hostPath storage path (regression safety).
  • New `node_selector` block renders into the pod template spec.
  • New `pvc` block creates a `kind: PersistentVolumeClaim` + volume
    binding via persistentVolumeClaim.claimName.
  • `secret_mounts` creates a volume via secret.secretName.
  • All YAML documents are parseable (yaml.safe_load_all).
"""
from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from sndr.cli.legacy.k8s import _all_yaml
from sndr.model_configs.schema import (
    DockerConfig, HardwareSpec, KubernetesConfig, ModelConfig,
)


def _make_k8s_cfg(**k8s_overrides) -> ModelConfig:
    k8s_kwargs = dict(
        flavor="generic-single-node",
        namespace="test-ns",
        image="vllm/vllm-openai:nightly",
        gpu_count=2,
    )
    k8s_kwargs.update(k8s_overrides)
    return ModelConfig(
        key="test-k8s", title="K8s Test",
        description="d", schema_version=1, maintainer="x",
        model_path="/models/Test-7B",
        hardware=HardwareSpec(
            gpu_match_keys=["test"], n_gpus=2,
            min_vram_per_gpu_mib=24576,
        ),
        docker=DockerConfig(
            image="vllm/vllm-openai:nightly",
            container_name="vllm-test",
            port=8000,
        ),
        kubernetes=KubernetesConfig(**k8s_kwargs),
    )


class TestK8sRender:
    def test_baseline_yaml_parses(self):
        cfg = _make_k8s_cfg()
        docs = list(yaml.safe_load_all(_all_yaml(cfg)))
        # ConfigMap + Service + Deployment at minimum
        kinds = [d.get("kind") for d in docs if d]
        assert "ConfigMap" in kinds
        assert "Service" in kinds
        assert "Deployment" in kinds

    def test_hostpath_storage_renders(self):
        cfg = _make_k8s_cfg(storage={"models": "/srv/models"})
        out = _all_yaml(cfg)
        assert "hostPath" in out
        assert "/srv/models" in out

    def test_node_selector_renders(self):
        cfg = _make_k8s_cfg(node_selector={
            "gpu-class": "a5000",
            "nvidia.com/gpu.present": "true",
        })
        out = _all_yaml(cfg)
        assert "nodeSelector:" in out
        assert "gpu-class: a5000" in out
        # Parseable
        docs = list(yaml.safe_load_all(out))
        deploy = next(d for d in docs if d and d.get("kind") == "Deployment")
        node_sel = deploy["spec"]["template"]["spec"]["nodeSelector"]
        assert node_sel["gpu-class"] == "a5000"

    def test_node_selector_omitted_when_empty(self):
        cfg = _make_k8s_cfg()  # no node_selector
        out = _all_yaml(cfg)
        assert "nodeSelector:" not in out

    def test_pvc_creates_persistent_volume_claim(self):
        cfg = _make_k8s_cfg(
            pvc={"genesis-models": "/models"},
            pvc_size_gib={"genesis-models": 500},
            pvc_storage_class="local-path",
        )
        out = _all_yaml(cfg)
        docs = list(yaml.safe_load_all(out))
        kinds = [d.get("kind") for d in docs if d]
        assert "PersistentVolumeClaim" in kinds
        pvc = next(d for d in docs
                    if d and d.get("kind") == "PersistentVolumeClaim")
        assert pvc["metadata"]["name"] == "genesis-models"
        assert pvc["spec"]["resources"]["requests"]["storage"] == "500Gi"
        assert pvc["spec"]["storageClassName"] == "local-path"

    def test_pvc_default_size_when_unspecified(self):
        cfg = _make_k8s_cfg(pvc={"hf-cache": "/root/.cache/huggingface"})
        out = _all_yaml(cfg)
        docs = list(yaml.safe_load_all(out))
        pvc = next(d for d in docs
                    if d and d.get("kind") == "PersistentVolumeClaim")
        assert pvc["spec"]["resources"]["requests"]["storage"] == "100Gi"

    def test_pvc_mounted_in_deployment(self):
        cfg = _make_k8s_cfg(pvc={"models-vol": "/models"})
        docs = list(yaml.safe_load_all(_all_yaml(cfg)))
        deploy = next(d for d in docs if d and d.get("kind") == "Deployment")
        container = deploy["spec"]["template"]["spec"]["containers"][0]
        mount = next(m for m in container["volumeMounts"]
                      if m["name"] == "models-vol")
        assert mount["mountPath"] == "/models"
        # Volume binding
        vol = next(v for v in deploy["spec"]["template"]["spec"]["volumes"]
                    if v["name"] == "models-vol")
        assert vol["persistentVolumeClaim"]["claimName"] == "models-vol"

    def test_secret_mount_renders(self):
        cfg = _make_k8s_cfg(secret_mounts={"hf-token": "/etc/hf-token"})
        docs = list(yaml.safe_load_all(_all_yaml(cfg)))
        deploy = next(d for d in docs if d and d.get("kind") == "Deployment")
        vol = next(v for v in deploy["spec"]["template"]["spec"]["volumes"]
                    if v["name"] == "hf-token")
        assert vol["secret"]["secretName"] == "hf-token"

    def test_no_pvc_when_empty(self):
        cfg = _make_k8s_cfg()  # no pvc / no secret
        out = _all_yaml(cfg)
        docs = list(yaml.safe_load_all(out))
        kinds = [d.get("kind") for d in docs if d]
        assert "PersistentVolumeClaim" not in kinds


# ─── Etap 2.4/2.5 — schema validation at render time ────────────────────


class TestRenderValidation:
    """Etap 2.4/2.5 (audit 2026-05-12): YAML is built from dicts via
    `yaml.safe_dump_all` and each name / path / size is validated at
    render time. Misconfigurations raise ValueError instead of reaching
    `kubectl apply` and failing with cryptic server-side errors."""

    def test_invalid_pvc_name_rejected(self):
        cfg = _make_k8s_cfg(pvc={"BadName_With_Underscores": "/data"})
        with pytest.raises(ValueError, match="DNS-1123"):
            _all_yaml(cfg)

    def test_relative_pvc_mount_rejected(self):
        cfg = _make_k8s_cfg(pvc={"models": "relative/path"})
        with pytest.raises(ValueError, match="must be an absolute path"):
            _all_yaml(cfg)

    def test_zero_pvc_size_rejected(self):
        cfg = _make_k8s_cfg(pvc={"x": "/x"}, pvc_size_gib={"x": 0})
        with pytest.raises(ValueError, match="must be > 0"):
            _all_yaml(cfg)

    def test_negative_pvc_size_rejected(self):
        cfg = _make_k8s_cfg(pvc={"x": "/x"}, pvc_size_gib={"x": -10})
        with pytest.raises(ValueError, match="must be > 0"):
            _all_yaml(cfg)

    def test_duplicate_mount_path_rejected(self):
        cfg = _make_k8s_cfg(
            pvc={"models": "/data"},
            secret_mounts={"token": "/data"},  # collides
        )
        with pytest.raises(ValueError, match="collides"):
            _all_yaml(cfg)

    def test_invalid_secret_name_rejected(self):
        cfg = _make_k8s_cfg(secret_mounts={"Bad_Secret!": "/etc/x"})
        with pytest.raises(ValueError, match="DNS-1123"):
            _all_yaml(cfg)

    def test_label_with_prefix_accepted(self):
        """K8s labels allow `prefix/name` format."""
        cfg = _make_k8s_cfg(node_selector={
            "nvidia.com/gpu.present": "true",
            "gpu-class": "a5000",
        })
        # No exception
        docs = list(yaml.safe_load_all(_all_yaml(cfg)))
        deploy = next(d for d in docs if d and d.get("kind") == "Deployment")
        sel = deploy["spec"]["template"]["spec"]["nodeSelector"]
        assert sel["nvidia.com/gpu.present"] == "true"
        assert sel["gpu-class"] == "a5000"

    def test_invalid_label_prefix_rejected(self):
        cfg = _make_k8s_cfg(node_selector={"Bad.Prefix/x": "v"})
        with pytest.raises(ValueError, match="DNS-1123 subdomain"):
            _all_yaml(cfg)

    def test_yaml_output_is_well_formed(self):
        """safe_dump_all output must parse as a multi-doc YAML stream."""
        cfg = _make_k8s_cfg(
            pvc={"models-vol": "/models"},
            secret_mounts={"hf-token": "/secrets/hf"},
            node_selector={"gpu-class": "a5000"},
        )
        out = _all_yaml(cfg)
        docs = list(yaml.safe_load_all(out))
        # Header is a comment-only preamble; safe_load_all yields one
        # implicit `None` doc for it plus the four real manifests
        real = [d for d in docs if isinstance(d, dict)]
        # ConfigMap + Service + 1 PVC + Deployment
        assert len(real) == 4
        kinds = {d["kind"] for d in real}
        assert kinds == {"ConfigMap", "Service", "PersistentVolumeClaim",
                          "Deployment"}


# ─── v12 — SNDR identity stamped on every Deployment ────────────────────


class TestSndrIdentity:
    """v12 (research-driven 2026-06-08): the k8s tab felt like a generic
    dashboard because rendered Deployments were anonymous (`app: <name>`
    only). Stamping the SNDR identity — preset key, pin, enabled-patch
    count + names — as labels/annotations is the keystone: the panel can
    then map a live pod back to the preset/pin/patches that defined it,
    and every later feature (drift badge, rolling pin upgrade, autoscale
    bounds) keys off this identity. Selector stays on `app=` (immutable)."""

    def test_deployment_carries_sndr_identity_labels(self):
        cfg = _make_k8s_cfg()
        cfg.genesis_env.update({
            "GENESIS_ENABLE_P82": "1",
            "GENESIS_ENABLE_PN90": "1",
            "GENESIS_ENABLE_P71": "0",  # disabled -> not counted
        })
        docs = list(yaml.safe_load_all(_all_yaml(cfg)))
        deploy = next(d for d in docs if d and d.get("kind") == "Deployment")
        labels = deploy["metadata"]["labels"]
        assert labels["app.kubernetes.io/managed-by"] == "sndr"
        assert labels["sndr.io/preset"] == "test-k8s"
        assert labels["sndr.io/patch-count"] == "2"
        # The pod template carries the same identity so pod queries resolve it.
        tmpl_labels = deploy["spec"]["template"]["metadata"]["labels"]
        assert tmpl_labels["sndr.io/preset"] == "test-k8s"
        assert tmpl_labels["sndr.io/patch-count"] == "2"
        # Selector stays on app= only (immutable, must remain a subset).
        assert deploy["spec"]["selector"]["matchLabels"] == {"app": "sndr-test-k8s"}

    def test_deployment_annotations_carry_pin_and_patch_names(self):
        cfg = _make_k8s_cfg(image="vllm/vllm-openai:nightly-abc123")
        cfg.genesis_env.update({"GENESIS_ENABLE_P82": "1", "GENESIS_ENABLE_PN90": "1"})
        docs = list(yaml.safe_load_all(_all_yaml(cfg)))
        deploy = next(d for d in docs if d and d.get("kind") == "Deployment")
        ann = deploy["metadata"]["annotations"]
        assert ann["sndr.io/pin"] == "nightly-abc123"
        assert "P82" in ann["sndr.io/patches"]
        assert "PN90" in ann["sndr.io/patches"]

    def test_identity_labels_are_k8s_valid(self):
        # patch-count is a string scalar; managed-by/preset are label-safe.
        cfg = _make_k8s_cfg()
        docs = list(yaml.safe_load_all(_all_yaml(cfg)))
        deploy = next(d for d in docs if d and d.get("kind") == "Deployment")
        labels = deploy["metadata"]["labels"]
        assert labels["sndr.io/patch-count"] == "0"  # no patches enabled
        # No annotations crash when there are zero enabled patches.
        ann = deploy["metadata"].get("annotations", {})
        assert ann.get("sndr.io/patches", "") == ""


# ─── Etap 2.6 — delete --delete-pvc opt-in ──────────────────────────────


class TestDeletePvcFlag:
    """Etap 2.6 (audit 2026-05-12): PVCs are preserved by default
    on `sndr k8s delete`. Operator must pass --delete-pvc to drop them."""

    def test_argparser_registers_delete_pvc(self):
        """The flag is registered on the `delete` subcommand only."""
        import argparse
        from sndr.cli.legacy.k8s import add_argparser
        parser = argparse.ArgumentParser(prog="sndr")
        subparsers = parser.add_subparsers()
        add_argparser(subparsers)
        # Parsing should accept --delete-pvc on `k8s delete`
        ns = parser.parse_args(["k8s", "delete", "some-preset",
                                  "--delete-pvc"])
        assert getattr(ns, "delete_pvc", False) is True

    def test_argparser_flag_absent_on_other_subcommands(self):
        """The flag must NOT be accepted on render/apply/status/logs."""
        import argparse
        from sndr.cli.legacy.k8s import add_argparser
        parser = argparse.ArgumentParser(prog="sndr")
        subparsers = parser.add_subparsers()
        add_argparser(subparsers)
        with pytest.raises(SystemExit):
            parser.parse_args(["k8s", "render", "x", "--delete-pvc"])
