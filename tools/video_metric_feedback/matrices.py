"""Export low-dimensional video-feedback parameters to the existing FFN runtime.

The runtime uses row-stored tokens, ``U @ A``, and a complete high/low matrix
manifest.  Selected sites use the CURRENT_Q1 kit, not the historical Q1r2
matrix at the same site.  Their unrounded family is

    A(theta) = A_q1 + theta * (P64 / gamma_L) @ C_L @ Q32.T.

Qb is FP64 -> FP32 -> BF16 round-to-nearest-even.  At exactly theta=0 or 1,
the archived BF16 endpoint is authoritative.  CPU reconstruction differences
are reported, since CPU and original GPU FP64 GEMMs need not be bit-identical.
Other sites retain the supplied base manifest.  No model is loaded or run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


MODE = "kit_current_q1"
RECIPE = "A_q1 + theta*(P64/gamma_L)@C_L@Q32.T; FP64->FP32->BF16_RNE"
_SHA = re.compile(r"^[0-9a-fA-F]{64}$")
_CORE_KEYS = ("A_q1", "P64", "gamma_L", "C_L", "Q32")
_ENDPOINT_KEYS = ("A_Q1_BF16", "A_L_BF16")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    """Decode raw BF16 words without treating them as integer values."""
    words = np.asarray(bits)
    if words.dtype != np.dtype("uint16"):
        raise ValueError("BF16 storage must be uint16 raw words")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def float32_to_bf16_rne(values: np.ndarray) -> np.ndarray:
    """Apply the repository's FP32 -> BF16 RNE contract to finite values."""
    with np.errstate(over="ignore", invalid="ignore"):
        value32 = np.asarray(values, dtype=np.float32)
    if not np.isfinite(value32).all():
        raise ValueError("Nonfinite FP32 operand in BF16 export")
    words = np.ascontiguousarray(value32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    result = ((words + rounding) >> np.uint32(16)).astype(np.uint16)
    _check_finite_bits(result, "rounded matrix")
    return result


def _check_finite_bits(bits: np.ndarray, name: str) -> None:
    if np.any((bits & np.uint16(0x7F80)) == np.uint16(0x7F80)):
        raise ValueError(f"Nonfinite BF16 value in {name}")


def _dimensions(expected_width: int, expected_layers: int) -> tuple[int, int]:
    if (isinstance(expected_width, bool) or not isinstance(expected_width, int)
            or expected_width <= 0):
        raise ValueError("expected_width must be a positive integer")
    if (isinstance(expected_layers, bool) or not isinstance(expected_layers, int)
            or expected_layers <= 0):
        raise ValueError("expected_layers must be a positive integer")
    return expected_width, expected_layers


def _read_manifest(path: Path, width: int, layers: int) -> dict[str, dict[str, Any]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    expected = {f"{expert}/{layer}" for expert in ("high", "low") for layer in range(layers)}
    if not isinstance(entries, dict) or set(entries) != expected:
        raise ValueError(f"Consumer manifest must contain exactly {2 * layers} high/low sites")
    for site, item in entries.items():
        if not isinstance(item, dict) or item.get("shape") != [width, width]:
            raise ValueError(f"Invalid matrix shape in manifest at {site}")
        if not isinstance(item.get("sha256"), str) or not _SHA.fullmatch(item["sha256"]):
            raise ValueError(f"Invalid matrix SHA256 at {site}")
        if item.get("dtype", "BF16_RAW_UINT16") != "BF16_RAW_UINT16":
            raise ValueError(f"Invalid matrix dtype at {site}")
        if not isinstance(item.get("path"), str) or not item["path"]:
            raise ValueError(f"Missing matrix path at {site}")
        matrix_path = Path(item["path"]).expanduser()
        if not matrix_path.is_absolute():
            matrix_path = path.parent / matrix_path
        item["path"] = str(matrix_path.resolve())
        item["sha256"] = item["sha256"].lower()
    return entries


def _read_matrix(item: dict[str, Any], width: int, *, verify_hash: bool) -> np.ndarray:
    path = Path(item["path"])
    if verify_hash and sha256_file(path) != item["sha256"]:
        raise ValueError(f"Matrix SHA256 mismatch: {path}")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if not isinstance(value, np.ndarray):
        if hasattr(value, "close"):
            value.close()
        raise ValueError(f"Expected one .npy raw BF16 matrix: {path}")
    if value.dtype != np.dtype("uint16") or value.shape != (width, width):
        raise ValueError(f"Invalid raw BF16 matrix payload: {path}")
    if verify_hash:
        _check_finite_bits(value, str(path))
    return value


def preflight_manifest(
    manifest_path: str | Path,
    *,
    expected_width: int = 5120,
    expected_layers: int = 40,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate a complete consumer manifest before native generation.

    Full mode reads and hashes each distinct payload once.  Header-only mode
    checks availability/shape/dtype and explicitly does not verify payload SHA
    or finiteness.  Export itself hashes only the overwritten base sites.
    """
    width, layers = _dimensions(expected_width, expected_layers)
    path = Path(manifest_path).resolve()
    entries = _read_manifest(path, width, layers)
    checked: dict[str, str] = {}
    payload_bytes = 0
    for item in entries.values():
        old_sha = checked.get(item["path"])
        if old_sha is not None:
            if old_sha != item["sha256"]:
                raise ValueError("One payload path has conflicting declared SHA256 values")
            continue
        value = _read_matrix(item, width, verify_hash=verify_hashes)
        payload_bytes += value.nbytes
        del value
        checked[item["path"]] = item["sha256"]
    return {
        "manifest_path": str(path), "manifest_sha256": sha256_file(path),
        "sites": len(entries), "distinct_payloads": len(checked),
        "distinct_payload_bytes": payload_bytes,
        "payload_sha256_and_finite_verified": bool(verify_hashes),
    }


def _validate_shapes(arrays: dict[str, Any], width: int) -> None:
    if any(key not in arrays for key in _CORE_KEYS + _ENDPOINT_KEYS):
        raise ValueError("Kit must contain the linear factors and both archived BF16 endpoints")
    a, p, gamma, c, q = (arrays[key] for key in _CORE_KEYS)
    if tuple(a.shape) != (width, width):
        raise ValueError("A_q1 shape mismatch")
    if len(p.shape) != 2 or p.shape[0] != width or p.shape[1] <= 0:
        raise ValueError("P64 shape mismatch")
    if len(q.shape) != 2 or q.shape[0] != width or q.shape[1] <= 0:
        raise ValueError("Q32 shape mismatch")
    if tuple(gamma.shape) != (p.shape[1],) or tuple(c.shape) != (p.shape[1], q.shape[1]):
        raise ValueError("Readout/gamma shape mismatch")
    for key in _ENDPOINT_KEYS:
        if tuple(arrays[key].shape) != (width, width):
            raise ValueError(f"{key} shape mismatch")


def _endpoint_result(
    current: np.ndarray, learned: np.ndarray,
    reconstructed_current: np.ndarray, reconstructed_learned: np.ndarray,
    interior: np.ndarray | None, theta: float, backend: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    for name, bits in (("A_Q1_BF16", current), ("A_L_BF16", learned)):
        if bits.dtype != np.dtype("uint16"):
            raise ValueError(f"{name} must contain raw BF16 uint16 words")
        _check_finite_bits(bits, name)
    q1_difference = int(np.count_nonzero(current != reconstructed_current))
    l_difference = int(np.count_nonzero(learned != reconstructed_learned))
    if theta == 0.0:
        selected, route = current.copy(), "archived_A_Q1_BF16"
    elif theta == 1.0:
        selected, route = learned.copy(), "archived_A_L_BF16"
    else:
        assert interior is not None
        selected, route = interior, "cpu_fp64_reconstruction_then_fp32_bf16"
    return selected, current, {
        "backend": backend, "selected_route": route,
        "endpoint_authority": "archived_bf16_bits",
        "reconstructed_q1_mismatch_count": q1_difference,
        "reconstructed_l_mismatch_count": l_difference,
        "reconstructed_q1_matches_archived": q1_difference == 0,
        "reconstructed_l_matches_archived": l_difference == 0,
        "original_gpu_intermediate_bit_equivalence": "NOT_ASSERTED",
    }


def _fold_npz(path: Path, theta: float, width: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        kit = {key: archive[key] for key in _CORE_KEYS + _ENDPOINT_KEYS if key in archive}
    _validate_shapes(kit, width)
    for key in _CORE_KEYS:
        if kit[key].dtype.kind != "f" or not np.isfinite(kit[key]).all():
            raise ValueError(f"Nonfinite or non-floating kit tensor: {key}")
        kit[key] = kit[key].astype(np.float64, copy=False)
    if np.any(kit["gamma_L"] <= 0):
        raise ValueError("gamma_L must be strictly positive")
    delta = ((kit["P64"] / kit["gamma_L"][None, :]) @ kit["C_L"]) @ kit["Q32"].T
    current = float32_to_bf16_rne(kit["A_q1"])
    learned = float32_to_bf16_rne(kit["A_q1"] + delta)
    interior = (None if theta in (0.0, 1.0) else
                float32_to_bf16_rne(kit["A_q1"] + theta * delta))
    return _endpoint_result(kit["A_Q1_BF16"], kit["A_L_BF16"], current, learned,
                            interior, theta, f"numpy/{np.__version__}")


def _fold_pt(path: Path, theta: float, width: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import torch
    except ImportError as error:
        raise ImportError("Reading production .pt kits requires PyTorch; .npz fixtures require only NumPy") from error
    kit = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(kit, dict):
        raise ValueError("Expected the A_Q1_RESIDUAL tensor dictionary")
    for key in _CORE_KEYS + _ENDPOINT_KEYS:
        if key not in kit or not isinstance(kit[key], torch.Tensor):
            raise ValueError(f"Missing kit tensor: {key}")
    _validate_shapes(kit, width)
    for key in _CORE_KEYS:
        if not kit[key].is_floating_point() or not bool(torch.isfinite(kit[key]).all()):
            raise ValueError(f"Nonfinite or non-floating kit tensor: {key}")
        kit[key] = kit[key].double()
    if bool((kit["gamma_L"] <= 0).any()):
        raise ValueError("gamma_L must be strictly positive")
    for key in _ENDPOINT_KEYS:
        if kit[key].dtype != torch.bfloat16 or not bool(torch.isfinite(kit[key]).all()):
            raise ValueError(f"Expected finite BF16 endpoint: {key}")
    delta = ((kit["P64"] / kit["gamma_L"][None, :]) @ kit["C_L"]) @ kit["Q32"].T

    def qbits(value: Any) -> np.ndarray:
        rounded = value.float().bfloat16().contiguous()
        if not bool(torch.isfinite(rounded).all()):
            raise ValueError("Nonfinite BF16 value in reconstructed matrix")
        return rounded.view(torch.uint16).numpy()

    current, learned = qbits(kit["A_q1"]), qbits(kit["A_q1"] + delta)
    interior = None if theta in (0.0, 1.0) else qbits(kit["A_q1"] + theta * delta)
    return _endpoint_result(
        kit["A_Q1_BF16"].contiguous().view(torch.uint16).numpy(),
        kit["A_L_BF16"].contiguous().view(torch.uint16).numpy(),
        current, learned, interior, theta, f"torch/{torch.__version__}/cpu",
    )


def _publish_temp(temp_path: Path, destination: Path) -> str:
    digest = sha256_file(temp_path)
    try:
        os.link(temp_path, destination)
    except FileExistsError:
        if sha256_file(destination) != digest:
            raise FileExistsError(f"Refusing to replace a different candidate artifact: {destination}")
    finally:
        temp_path.unlink(missing_ok=True)
    return digest


def _save_matrix(path: Path, bits: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".matrix-", suffix=".npy", delete=False) as stream:
        temp_path = Path(stream.name)
        try:
            np.save(stream, bits, allow_pickle=False)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    return _publish_temp(temp_path, path)


def _json_bytes(data: Any) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _save_json(path: Path, value: Any) -> None:
    payload = _json_bytes(value)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".metadata-", delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write(payload)
    _publish_temp(temp_path, path)


def export_candidate(
    base_manifest_path: str | Path,
    sites: list[dict[str, Any]],
    theta: dict[str, float],
    output_dir: str | Path,
    expected_width: int = 5120,
    expected_layers: int = 40,
) -> dict[str, Any]:
    """Export a candidate for the unmodified ResidentMatrices consumer.

    Site specs contain ``site``, ``parameter``, ``kit_path`` and optionally
    ``kit_sha256``.  Relative payload/kit paths are resolved against the base
    manifest directory.  Each kit is loaded, folded and released separately.
    Existing identical artifacts are reusable; different content is never
    overwritten.  Call ``preflight_manifest`` before actual model generation.
    """
    width, layers = _dimensions(expected_width, expected_layers)
    base_path = Path(base_manifest_path).resolve()
    entries = _read_manifest(base_path, width, layers)
    if not isinstance(sites, list) or not sites:
        raise ValueError("At least one selected site is required")
    if not isinstance(theta, dict):
        raise ValueError("theta must be a parameter-to-coefficient dictionary")
    selected: dict[str, dict[str, Any]] = {}
    for item in sites:
        if not isinstance(item, dict):
            raise ValueError("Each selected site must be an object")
        site, parameter = item.get("site"), item.get("parameter")
        if not isinstance(site, str) or site not in entries or site in selected:
            raise ValueError("Unknown or duplicate selected site")
        if not isinstance(parameter, str) or not parameter:
            raise ValueError("Each selected site needs a parameter name")
        if not isinstance(item.get("kit_path"), str) or not item["kit_path"]:
            raise ValueError("Each selected site needs kit_path")
        expected_sha = item.get("kit_sha256")
        if expected_sha is not None and (not isinstance(expected_sha, str) or not _SHA.fullmatch(expected_sha)):
            raise ValueError("Invalid kit_sha256")
        selected[site] = item
    if set(theta) != {item["parameter"] for item in selected.values()}:
        raise ValueError("theta keys must exactly match the selected parameter groups")
    coefficients: dict[str, float] = {}
    for key, value in theta.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"Coefficient must be a finite number: {key}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Coefficient must be finite: {key}")
        coefficients[key] = 0.0 if number == 0.0 else number

    # Header checks are cheap; full old-payload hashes are deferred to preflight
    # except for selected sites, which are needed for actual changed fractions.
    for item in entries.values():
        header = _read_matrix(item, width, verify_hash=False)
        del header
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    details: list[dict[str, Any]] = []
    for site, specification in sorted(selected.items()):
        parameter = specification["parameter"]
        coefficient = coefficients[parameter]
        kit_path = Path(specification["kit_path"]).expanduser()
        if not kit_path.is_absolute():
            kit_path = base_path.parent / kit_path
        kit_path = kit_path.resolve()
        kit_sha = sha256_file(kit_path)
        if specification.get("kit_sha256") and kit_sha != specification["kit_sha256"].lower():
            raise ValueError(f"Kit SHA256 mismatch at {site}")
        if kit_path.suffix.lower() == ".npz":
            bits, current, checks = _fold_npz(kit_path, coefficient, width)
        elif kit_path.suffix.lower() == ".pt":
            bits, current, checks = _fold_pt(kit_path, coefficient, width)
        else:
            raise ValueError("Expected a production .pt kit or a NumPy .npz fixture")
        original_entry = dict(entries[site])
        original = _read_matrix(original_entry, width, verify_hash=True)
        changed_base = float(np.count_nonzero(bits != original) / bits.size)
        changed_current = float(np.count_nonzero(bits != current) / bits.size)
        del original, current
        expert, layer = site.split("/")
        matrix_path = output / "matrices" / f"{expert}_{int(layer):02d}.npy"
        matrix_sha = _save_matrix(matrix_path, bits)
        del bits
        candidate_entry = {
            "path": str(matrix_path), "sha256": matrix_sha, "shape": [width, width],
            "dtype": "BF16_RAW_UINT16", "recipe": RECIPE,
            "provenance": {"mode": MODE, "parameter": parameter, "coefficient": coefficient,
                           "kit_sha256": kit_sha, "numerical_route": checks["selected_route"]},
        }
        entries[site] = candidate_entry
        details.append({
            "site": site, "parameter": parameter, "coefficient": coefficient,
            "kit_path": str(kit_path), "kit_sha256": kit_sha,
            "base_entry": original_entry, "candidate_entry": candidate_entry,
            "changed_fraction_from_base": changed_base,
            "changed_fraction_from_current_q1": changed_current,
            "endpoint_check": checks,
        })
    identity = {
        "mode": MODE, "recipe": RECIPE, "width": width, "layers_per_expert": layers,
        "base_manifest_sha256": sha256_file(base_path), "coefficients": coefficients,
        "selected_sites": [{"site": d["site"], "parameter": d["parameter"],
                            "kit_sha256": d["kit_sha256"],
                            "payload_sha256": d["candidate_entry"]["sha256"]} for d in details],
        "all_payload_sha256": {site: item["sha256"] for site, item in entries.items()},
    }
    # The oracle sees compiled BF16 matrices, not their unrounded coefficients.
    # Different theta values on the same quantization plateau must share an
    # execution identity.  Keep their distinct provenance in a separate hash.
    execution_identity = {
        "schema": "ffnds.video_metric_feedback.matrix_execution.v1",
        "width": width, "layers_per_expert": layers,
        "matrix_convention": "row-stored tokens U @ A; no transpose",
        "sites": {site: {"payload_sha256": item["sha256"],
                          "shape": item["shape"], "dtype": "BF16_RAW_UINT16"}
                  for site, item in sorted(entries.items())},
    }
    fingerprint = hashlib.sha256(_json_bytes(execution_identity)).hexdigest()
    provenance_fingerprint = hashlib.sha256(_json_bytes(identity)).hexdigest()
    manifest_path = output / "matrix_manifest.json"
    metadata_path = output / "matrix_metadata.json"
    metadata = {
        "schema": "ffnds.video_metric_feedback.matrices.v1", "fingerprint": fingerprint,
        "execution_identity": execution_identity,
        "provenance_fingerprint": provenance_fingerprint,
        "identity": identity, "base_manifest_path": str(base_path),
        "consumer_matrix_convention": "row-stored tokens U @ A; no transpose",
        "baseline_identity": "CURRENT_Q1 at selected kit sites; supplied base matrices elsewhere; not historical full Q1r2",
        "endpoint_policy": "theta 0/1 use archived BF16 bits; other values use CPU FP64 reconstruction then FP32/BF16",
        "cache_lifecycle": "Every candidate starts a complete new HHLL generation with fresh anchor caches",
        "base_payload_sha_verification": "selected sites verified during export; all sites require preflight_manifest before generation",
        "selected_sites": details, "untouched_sites": sorted(set(entries) - set(selected)),
        "matrix_manifest_sha256": hashlib.sha256(_json_bytes(entries)).hexdigest(),
    }
    _save_json(metadata_path, metadata)
    _save_json(manifest_path, entries)
    return {
        "manifest_path": str(manifest_path), "metadata_path": str(metadata_path),
        "fingerprint": fingerprint, "provenance_fingerprint": provenance_fingerprint,
        "mode": MODE, "selected_site_count": len(details),
        "total_site_count": len(entries), "coefficients": coefficients,
        "matrix_manifest_sha256": metadata["matrix_manifest_sha256"],
        "requires_full_payload_preflight": True,
    }
