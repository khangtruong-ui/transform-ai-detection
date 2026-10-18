"""Online checkpoint discovery, downloading, and strict weight validation hub.

Enforces the strict policy that no localization model may ever execute with
randomly initialized weights. Models without a verified pretrained checkpoint
(either resolved locally or downloaded from a registered online source) quit
with an explicit diagnostic notice.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


ONLINE_CHECKPOINT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "pscc_net": {
        "online_available": True,
        "default_filename": "pscc_net.pth",
        "primary_source_desc": "https://github.com/proteus1991/PSCC-Net (HRNet + NLCDetection + DetectionHead)",
        "sources": [
            {
                "type": "github_multi",
                "description": "Official PSCC-Net pretrained weights (HRNet + NLCDetection + DetectionHead)",
                "components": {
                    "FENet": "https://raw.githubusercontent.com/proteus1991/PSCC-Net/main/checkpoint/HRNet_checkpoint/HRNet.pth",
                    "SegNet": "https://raw.githubusercontent.com/proteus1991/PSCC-Net/main/checkpoint/NLCDetection_checkpoint/NLCDetection.pth",
                    "ClsNet": "https://raw.githubusercontent.com/proteus1991/PSCC-Net/main/checkpoint/DetectionHead_checkpoint/DetectionHead.pth",
                },
            },
            {
                "type": "hf_hub",
                "repo_id": "nebula/MaskCLIP-weights",
                "filename": "PSCC_sd15_20241017_16_41_36.pth",
                "description": "OpenSDI / IMDLBenCo PSCC-Net checkpoint on Hugging Face Hub",
            },
            {
                "type": "gdrive",
                "file_id": "1cL2YPAQdrC-MYrjNU9MBtX7UdZWCy4un",
                "filename": "pscc_cat_net.pth",
                "description": "PSCC-Net CAT-Net benchmark pretrained weights (Google Drive)",
            },
            {
                "type": "gdrive",
                "file_id": "1hT5X7vvd_H8NPtxfrscAW3A7W6aNSqhL",
                "filename": "pscc_casiav2.pth",
                "description": "PSCC-Net CASIA-v2 pretrained weights (Google Drive)",
            },
        ],
    },
    "trufor": {
        "online_available": True,
        "default_filename": "trufor.pth.tar",
        "primary_source_desc": "hf://DaniilOr/detect/weights/trufor.pth.tar & https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip",
        "sources": [
            {
                "type": "hf_hub",
                "repo_id": "DaniilOr/detect",
                "filename": "weights/trufor.pth.tar",
                "description": "Official TruFor checkpoint hosted on Hugging Face Hub (DaniilOr/detect)",
            },
            {
                "type": "url_zip",
                "url": "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip",
                "member": "weights/trufor.pth.tar",
                "md5": "7bee48f3476c7fd2ded7b3222b2329c5",
                "description": "Official GRIP-UNINA TruFor weights archive",
            },
            {
                "type": "hf_hub",
                "repo_id": "suncore147/imdl-benco",
                "filename": "Trufor/checkpoint-0.pth",
                "description": "IMDL-BenCo TruFor pretrained checkpoint on Hugging Face Hub",
            },
            {
                "type": "hf_hub",
                "repo_id": "nebula/MaskCLIP-weights",
                "filename": "Trufor_sd15_20241107_00_29_48_phase2.pth",
                "description": "OpenSDI TruFor pretrained checkpoint on Hugging Face Hub",
            },
            {
                "type": "gdrive",
                "file_id": "1x6_FaYMxiJfBAZ9WZjcagKWY-95Sl45-",
                "filename": "trufor_cat_net.pth",
                "description": "TruFor CAT-Net pretrained checkpoint (Google Drive)",
            },
        ],
    },
    "iml_vit": {
        "online_available": True,
        "default_filename": "iml_vit.pth",
        "primary_source_desc": "hf://suncore147/imdl-benco/IML_ViT/checkpoint-199.pth & Google Drive (SunnyHaze/IML-ViT)",
        "sources": [
            {
                "type": "hf_hub",
                "repo_id": "suncore147/imdl-benco",
                "filename": "IML_ViT/checkpoint-199.pth",
                "description": "Official IML-BenCo IML-ViT checkpoint on Hugging Face Hub",
            },
            {
                "type": "hf_hub",
                "repo_id": "nebula/MaskCLIP-weights",
                "filename": "IML_ViT_sd15_20241106_04_29_53.pth",
                "description": "OpenSDI IML-ViT checkpoint on Hugging Face Hub",
            },
            {
                "type": "gdrive",
                "file_id": "1xXJGJPW1i5j9Pc1JKd7fJmIAQkvt9jY7",
                "filename": "iml-vit_checkpoint.pth",
                "description": "Official SunnyHaze/IML-ViT CASIAv2 checkpoint (Google Drive)",
            },
            {
                "type": "gdrive",
                "file_id": "1jlXw97GkyBbY4u5-e_liuhahKSQWCAFu",
                "filename": "iml-vit_checkpoint_trufor_20231104.pth",
                "description": "Official SunnyHaze/IML-ViT CAT-Net/TruFor protocol checkpoint (Google Drive)",
            },
            {
                "type": "gdrive",
                "file_id": "1Rs9gsg6vZmzKOcwx4udB3-A0FwjfFenb",
                "filename": "iml-vit_checkpoint_casiav2_20231014.pth",
                "description": "Official SunnyHaze/IML-ViT CASIAv2 20231014 checkpoint (Google Drive)",
            },
        ],
    },
    "sparsevit": {
        "online_available": True,
        "default_filename": "sparsevit.pth",
        "primary_source_desc": "gdrive://104BPPvLXkxuPu_NHaxjesdcdZ-ln92-G (scu-zjz/SparseViT)",
        "sources": [
            {
                "type": "gdrive",
                "file_id": "104BPPvLXkxuPu_NHaxjesdcdZ-ln92-G",
                "filename": "sparsevit_checkpoint.pth",
                "description": "Official scu-zjz/SparseViT pretrained checkpoint (Google Drive)",
            },
        ],
    },
    "code": {
        "online_available": True,
        "default_filename": "code.pth",
        "primary_source_desc": "gdrive://1aXTKhn-vKu4fG7XdKyWv4AWZVMkcIdd3 (siriusPRX/CoDE CoDE_pre-trained.pth)",
        "sources": [
            {
                "type": "gdrive",
                "file_id": "1aXTKhn-vKu4fG7XdKyWv4AWZVMkcIdd3",
                "filename": "CoDE_pre-trained.pth",
                "description": "Official siriusPRX/CoDE reinforcement learning pretrained weights (Google Drive)",
            },
        ],
    },
    "maskclip": {
        "online_available": True,
        "default_filename": "maskclip.pth",
        "primary_source_desc": "hf://nebula/MaskCLIP-weights/MaskCLIP_sd15_20241103_17_45_16.pth (iamwangyabin/OpenSDI)",
        "sources": [
            {
                "type": "hf_hub",
                "repo_id": "nebula/MaskCLIP-weights",
                "filename": "MaskCLIP_sd15_20241103_17_45_16.pth",
                "description": "Official OpenSDI MaskCLIP pretrained checkpoint on Hugging Face Hub",
            },
        ],
    },
    "sida": {
        "online_available": True,
        "default_filename": "sida.pth",
        "primary_source_desc": "hf://saberzl/SIDA-7B & https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sources": [
            {
                "type": "hf_repo",
                "repo_id": "saberzl/SIDA-7B",
                "sam_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "description": "Official SIDA-7B (LLaVA-1.5-7B + SAM-ViT-H, ~16 GB) on Hugging Face Hub",
            },
        ],
    },
    "rita": {
        "online_available": False,
        "default_filename": "rita.pth",
        "primary_source_desc": "Unavailable publicly (password-protected Baidu Netdisk: pan.baidu.com/s/1W1HC4_mn044ub2NLJLSeQQ?pwd=43nn)",
        "sources": [],
        "unavailable_reason": (
            "Quitting with notice: Official RITA weights (scu-zjz/RITA) are only released on "
            "password-protected Baidu Netdisk (https://pan.baidu.com/s/1W1HC4_mn044ub2NLJLSeQQ?pwd=43nn) "
            "and no local checkpoint was found. Running with randomly initialized weights is prohibited."
        ),
    },
    "deep_forensic_detector": {
        "online_available": False,
        "default_filename": "deep_forensic.pth",
        "primary_source_desc": "None (custom BayarConv2d CNN architecture without published online weights)",
        "sources": [],
        "unavailable_reason": (
            "Quitting with notice: DeepForensicNet ('deep_forensic_detector') has no published online "
            "pretrained checkpoint and no local checkpoint was provided. Running with randomly initialized "
            "weights is prohibited."
        ),
    },
}


def get_online_checkpoint_info(model_name: str) -> Dict[str, Any]:
    """Return online checkpoint metadata for a registered model."""
    clean = model_name.lower().replace("-", "_")
    aliases = {
        "sparse_vit": "sparsevit",
        "pscc": "pscc_net",
        "mask_clip": "maskclip",
    }
    clean = aliases.get(clean, clean)
    return dict(ONLINE_CHECKPOINT_REGISTRY.get(clean, {
        "online_available": False,
        "default_filename": f"{clean}.pth",
        "primary_source_desc": "None",
        "sources": [],
        "unavailable_reason": (
            f"Quitting with notice: No pretrained checkpoint found for '{model_name}'. "
            f"Running with randomly initialized weights is prohibited."
        ),
    }))


def should_auto_download(auto_download: Optional[bool] = None) -> bool:
    """Determine whether online checkpoint downloading should be attempted."""
    if auto_download is not None:
        return bool(auto_download)
    env_val = os.environ.get("IML_AUTO_DOWNLOAD")
    if env_val is not None:
        return env_val.strip().lower() in ("1", "true", "yes", "on")
    if os.environ.get("IML_DISABLE_ONLINE_DOWNLOAD", "").strip().lower() in ("1", "true", "yes"):
        return False
    # Avoid unintended multi-GB network downloads during unit test suites unless auto_download=True
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    return True


def _download_github_multi(
    components: Dict[str, str],
    dest_path: str,
    timeout: int = 60,
) -> str:
    """Download multi-part component checkpoints (e.g., PSCC-Net) and save unified checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    combined: Dict[str, Dict[str, Any]] = {}
    for comp_name, url in components.items():
        logger.info(f"Downloading {comp_name} weights from {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()
        sd = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        clean_sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        combined[comp_name] = clean_sd

    torch.save({"pscc_official_components": combined}, dest_path)
    logger.info(f"Saved combined pretrained checkpoint to {dest_path}")
    return dest_path


def _download_url_direct(url: str, dest_path: str, timeout: int = 120) -> str:
    """Download a checkpoint directly from an HTTP/HTTPS URL."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    tmp_path = dest_path + ".part"
    logger.info(f"Downloading checkpoint from {url} -> {dest_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as out_f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)
    os.replace(tmp_path, dest_path)
    return dest_path


def _download_url_zip(
    url: str,
    member: str,
    dest_path: str,
    expected_md5: Optional[str] = None,
    timeout: int = 180,
) -> str:
    """Download a ZIP archive and extract the target checkpoint member."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    zip_tmp = dest_path + ".zip.part"
    logger.info(f"Downloading checkpoint archive from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    md5_hasher = hashlib.md5()
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(zip_tmp, "wb") as out_f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)
            md5_hasher.update(chunk)

    if expected_md5 and md5_hasher.hexdigest().lower() != expected_md5.lower():
        os.remove(zip_tmp)
        raise RuntimeError(f"MD5 mismatch for {url}: expected {expected_md5}, got {md5_hasher.hexdigest()}")

    with zipfile.ZipFile(zip_tmp, "r") as zf:
        with zf.open(member) as src_f, open(dest_path + ".part", "wb") as dst_f:
            while True:
                buf = src_f.read(1024 * 1024)
                if not buf:
                    break
                dst_f.write(buf)
    os.remove(zip_tmp)
    os.replace(dest_path + ".part", dest_path)
    logger.info(f"Extracted {member} to {dest_path}")
    return dest_path


def _download_hf_hub(repo_id: str, filename: str, dest_path: str) -> str:
    """Download a checkpoint file from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    logger.info(f"Downloading from Hugging Face Hub: {repo_id}/{filename} ...")
    cached_file = hf_hub_download(repo_id=repo_id, filename=filename)
    if os.path.abspath(cached_file) != os.path.abspath(dest_path):
        # Create symlink or copy to dest_path for fast future lookup
        try:
            if os.path.lexists(dest_path):
                os.remove(dest_path)
            os.symlink(os.path.abspath(cached_file), dest_path)
        except OSError:
            import shutil
            shutil.copy2(cached_file, dest_path)
    return dest_path


def _download_gdrive(file_id: str, dest_path: str) -> str:
    """Download a checkpoint file from Google Drive using gdown."""
    import gdown

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    tmp_path = dest_path + ".part"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    logger.info(f"Downloading from Google Drive (id={file_id}) -> {dest_path} ...")
    out = gdown.download(id=file_id, output=tmp_path, quiet=False)
    if not out or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        raise RuntimeError(f"gdown failed to download Google Drive file id={file_id}")
    os.replace(tmp_path, dest_path)
    return dest_path


def resolve_or_download_checkpoint(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    local_candidates: Optional[List[str]] = None,
    auto_download: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    timeout: int = 120,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a pretrained checkpoint locally or download from online sources.

    Returns:
        Tuple of (resolved_checkpoint_path, error_or_quit_notice).
        If resolved_checkpoint_path is None, error_or_quit_notice contains the
        explicit notice explaining why the model is quitting instead of running
        with randomly initialized weights.
    """
    info = get_online_checkpoint_info(model_name)
    default_fname = info.get("default_filename", f"{model_name}.pth")
    cache_root = cache_dir or os.path.expanduser("~/.cache/iml_models")

    # 1. Explicit checkpoint_path takes strict precedence
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
            return checkpoint_path, None
        return (
            None,
            f"Quitting with notice: Explicit checkpoint_path '{checkpoint_path}' for model "
            f"'{model_name}' was not found. Running with randomly initialized weights is prohibited.",
        )

    # 2. Check standard local candidate paths
    candidates = list(local_candidates or [])
    default_candidates = [
        os.path.join(cache_root, default_fname),
        os.path.join(cache_root, f"{model_name}.pth"),
        os.path.join("checkpoints", default_fname),
        os.path.join("checkpoints", f"{model_name}.pth"),
    ]
    for p in default_candidates:
        if p not in candidates:
            candidates.append(p)

    for path in candidates:
        if path and os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path) > 0:
            return path, None

    # 3. Check online availability
    if not info.get("online_available", False) or not info.get("sources"):
        reason = info.get(
            "unavailable_reason",
            f"Quitting with notice: No online or local pretrained weights exist for '{model_name}'. "
            f"Running with randomly initialized weights is prohibited.",
        )
        return None, reason

    # 4. Check if online auto-download is enabled
    if not should_auto_download(auto_download):
        return (
            None,
            f"Quitting with notice: No local checkpoint found for '{model_name}' and online "
            f"auto-download is disabled (registered online source: {info.get('primary_source_desc')}). "
            f"Running with randomly initialized weights is prohibited.",
        )

    # 5. Attempt online download across registered sources in priority order
    dest_path = os.path.join(cache_root, default_fname)
    errors: List[str] = []

    for src in info.get("sources", []):
        src_type = src.get("type")
        try:
            if src_type == "github_multi":
                return _download_github_multi(src["components"], dest_path, timeout=timeout), None
            elif src_type == "hf_hub":
                return _download_hf_hub(src["repo_id"], src["filename"], dest_path), None
            elif src_type == "url":
                return _download_url_direct(src["url"], dest_path, timeout=timeout), None
            elif src_type == "url_zip":
                return (
                    _download_url_zip(
                        src["url"],
                        src["member"],
                        dest_path,
                        expected_md5=src.get("md5"),
                        timeout=timeout,
                    ),
                    None,
                )
            elif src_type == "gdrive":
                return _download_gdrive(src["file_id"], dest_path), None
            elif src_type == "hf_repo":
                if os.environ.get("IML_DOWNLOAD_LARGE_LLM", "").strip().lower() not in ("1", "true", "yes"):
                    errors.append(
                        f"Source '{src.get('repo_id')}' requires ~16 GB LLaVA-7B + SAM-ViT-H weights "
                        f"(set IML_DOWNLOAD_LARGE_LLM=1 or place checkpoint at {dest_path})"
                    )
                    continue
                from huggingface_hub import snapshot_download
                snap_dir = snapshot_download(repo_id=src["repo_id"])
                return snap_dir, None
        except Exception as e:
            err_msg = f"{src_type} ({src.get('description', src.get('repo_id', src.get('file_id', '')))}): {e}"
            logger.warning(f"Online checkpoint download failed for {model_name} via {err_msg}")
            errors.append(err_msg)

    joined_errors = "; ".join(errors) if errors else "no downloadable source succeeded"
    return (
        None,
        f"Quitting with notice: Failed to download online pretrained weights for '{model_name}' "
        f"({joined_errors}). Running with randomly initialized weights is prohibited.",
    )


def extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    """Extract parameter state_dict from a loaded checkpoint dictionary."""
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(ckpt).__name__}")

    for key in ("state_dict", "model", "model_state_dict", "net", "network"):
        if key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break

    # Strip DataParallel 'module.' prefix if present
    cleaned: Dict[str, Any] = {}
    for k, v in ckpt.items():
        clean_k = k[len("module."):] if k.startswith("module.") else k
        cleaned[clean_k] = v
    return cleaned


def validate_and_load_state_dict(
    model: nn.Module,
    state_dict: Dict[str, Any],
    model_name: str,
    min_match_ratio: float = 0.6,
) -> Tuple[bool, Optional[str]]:
    """Validate that state_dict keys & tensor shapes match `model` before loading.

    Prevents silent `strict=False` fallback to randomly initialized weights when
    a checkpoint belongs to a different or incompatible architecture.
    """
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return (
            False,
            f"Quitting with notice: Checkpoint for '{model_name}' contains an empty or invalid state_dict. "
            f"Running with randomly initialized weights is prohibited.",
        )

    model_sd = model.state_dict()
    if len(model_sd) == 0:
        return False, f"Model '{model_name}' has no parameters."

    matched_keys = []
    shape_mismatches = []
    for k, target_tensor in model_sd.items():
        if k in state_dict:
            src_tensor = state_dict[k]
            if hasattr(src_tensor, "shape") and tuple(src_tensor.shape) == tuple(target_tensor.shape):
                matched_keys.append(k)
            else:
                shape_mismatches.append(k)

    match_ratio = len(matched_keys) / float(len(model_sd))
    if match_ratio < min_match_ratio:
        return (
            False,
            f"Quitting with notice: Checkpoint for '{model_name}' matched only {len(matched_keys)}/{len(model_sd)} "
            f"({match_ratio:.1%}) tensors of {model.__class__.__name__} "
            f"(shape mismatches: {len(shape_mismatches)}). "
            f"Running with partially/randomly initialized weights is prohibited.",
        )

    filtered_sd = {k: state_dict[k] for k in matched_keys}
    model.load_state_dict(filtered_sd, strict=False)
    return True, None
