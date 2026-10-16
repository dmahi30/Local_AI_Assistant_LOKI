"""Local hardware inspection utilities for LOKI.

Detects RAM, CPU cores, AVX2 instruction support, and dedicated/integrated GPUs
without requiring external network access or heavy dependencies.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import sys
from typing import Any

import psutil

from ..core.exceptions import HardwareDetectionError
from .models import GPUInfo, HardwareSpec

# Known discrete GPU naming patterns
DISCRETE_GPU_PATTERNS = [
    "geforce",
    "rtx",
    "gtx",
    "quadro",
    "tesla",
    "titan",
    "radeon rx",
    "radeon pro",
    "radeon vii",
    "arc a",
    "arc pro",
    "discrete",
]

# Known integrated GPU naming patterns
INTEGRATED_GPU_PATTERNS = [
    "intel(r) uhd",
    "intel(r) hd",
    "intel uhd",
    "intel hd",
    "iris",
    "radeon graphics",
    "integrated",
]


def detect_ram() -> tuple[float, float]:
    """Detect total and available physical RAM in gigabytes.

    Returns (total_ram_gb, available_ram_gb).
    """
    try:
        vm = psutil.virtual_memory()
        total = round(vm.total / (1024**3), 2)
        available = round(vm.available / (1024**3), 2)
        return total, available
    except Exception as e:
        # Fallback to Windows GlobalMemoryStatusEx if psutil fails on Windows
        if sys.platform == "win32":
            try:
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                total = round(stat.ullTotalPhys / (1024**3), 2)
                available = round(stat.ullAvailPhys / (1024**3), 2)
                return total, available
            except Exception as win_err:
                raise HardwareDetectionError(f"Failed to query system RAM: {win_err}") from win_err
        raise HardwareDetectionError(f"Failed to query system RAM: {e}") from e


def detect_cpu() -> tuple[int, int, str, str]:
    """Detect CPU physical cores, logical threads, architecture, and processor name.

    Returns (cores, threads, arch, model_name).
    """
    try:
        cores = psutil.cpu_count(logical=False) or 1
        threads = psutil.cpu_count(logical=True) or cores
        arch = platform.machine() or "x86_64"
        model = platform.processor() or ""
        return cores, threads, arch, model
    except Exception as e:
        return 1, 1, platform.machine() or "x86_64", ""


def detect_avx2() -> bool:
    """Check if the CPU supports AVX2 vector instructions.

    AVX2 is critical for performant CPU inference in llama.cpp / Ollama.
    """
    if sys.platform == "win32":
        try:
            # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40 (from winnt.h)
            result = ctypes.windll.kernel32.IsProcessorFeaturePresent(40)
            return bool(result)
        except Exception:
            pass

    # Linux check
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                content = f.read()
                return "avx2" in content.lower()
        except Exception:
            pass

    # macOS check
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.leaf7_features"], text=True)
            return "AVX2" in output
        except Exception:
            pass

    return False


def _is_dedicated_gpu(name: str, vram_gb: float) -> bool:
    """Heuristic helper to classify whether a display adapter is discrete/dedicated."""
    lower_name = name.lower()

    # Check explicit integrated matches first
    for pattern in INTEGRATED_GPU_PATTERNS:
        if pattern in lower_name and not any(d in lower_name for d in ["rtx", "gtx", "rx"]):
            return False

    # Check discrete matches
    for pattern in DISCRETE_GPU_PATTERNS:
        if pattern in lower_name:
            return True

    # If VRAM is reported and exceeds 1.5 GB, it's likely dedicated
    if vram_gb >= 1.5:
        return True

    return False


def _detect_gpus_winreg() -> list[GPUInfo]:
    """Inspect Windows Display Class Registry keys for video adapters and 64-bit VRAM."""
    if sys.platform != "win32":
        return []

    gpus: list[GPUInfo] = []
    try:
        import winreg

        key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as root_key:
            for i in range(32):
                try:
                    subkey_name = winreg.EnumKey(root_key, i)
                    if not subkey_name.isdigit():
                        continue
                    with winreg.OpenKey(root_key, subkey_name) as subkey:
                        try:
                            desc, _ = winreg.QueryValueEx(subkey, "DriverDesc")
                            if not desc or "Remote" in desc or "Virtual" in desc:
                                continue
                            try:
                                vram_bytes, _ = winreg.QueryValueEx(subkey, "HardwareInformation.qwMemorySize")
                                vram_gb = round(vram_bytes / (1024**3), 2)
                            except FileNotFoundError:
                                vram_gb = 0.0

                            is_dedicated = _is_dedicated_gpu(desc, vram_gb)
                            gpus.append(
                                GPUInfo(
                                    name=desc,
                                    vram_gb=vram_gb,
                                    is_dedicated=is_dedicated,
                                    detection_source="winreg",
                                    metadata={"index": i},
                                )
                            )
                        except FileNotFoundError:
                            continue
                except OSError:
                    break
    except Exception:
        pass

    return gpus


def _detect_gpus_powershell() -> list[GPUInfo]:
    """Query Win32_VideoController via PowerShell CIM as a fallback."""
    if sys.platform != "win32":
        return []

    gpus: list[GPUInfo] = []
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
        if res.returncode == 0 and res.stdout.strip():
            raw_data = json.loads(res.stdout)
            items = raw_data if isinstance(raw_data, list) else [raw_data]
            for item in items:
                name = item.get("Name", "Unknown Adapter")
                adapter_ram = item.get("AdapterRAM") or 0
                vram_gb = round(adapter_ram / (1024**3), 2)
                is_dedicated = _is_dedicated_gpu(name, vram_gb)
                gpus.append(
                    GPUInfo(
                        name=name,
                        vram_gb=vram_gb,
                        is_dedicated=is_dedicated,
                        detection_source="wmi_powershell",
                    )
                )
    except Exception:
        pass

    return gpus


def _detect_gpus_nvidia_smi() -> list[GPUInfo]:
    """Query nvidia-smi if available on the system."""
    gpus: list[GPUInfo] = []
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
        if res.returncode == 0 and res.stdout.strip():
            for line in res.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    name = parts[0]
                    vram_mb = float(parts[1])
                    vram_gb = round(vram_mb / 1024.0, 2)
                    gpus.append(
                        GPUInfo(
                            name=name,
                            vram_gb=vram_gb,
                            is_dedicated=True,
                            detection_source="nvidia_smi",
                        )
                    )
    except Exception:
        pass

    return gpus


def detect_gpus() -> list[GPUInfo]:
    """Detect all available GPUs/display adapters, prioritizing dedicated cards.

    Uses a multi-tier fallback:
    1. winreg (Windows 64-bit VRAM precision)
    2. PowerShell CIM / WMI
    3. nvidia-smi
    Never raises an exception; gracefully returns an empty list if no GPUs found.
    """
    gpus: list[GPUInfo] = []

    # Strategy 1: Windows Registry
    winreg_gpus = _detect_gpus_winreg()
    if winreg_gpus:
        gpus = winreg_gpus

    # Strategy 2: PowerShell CIM fallback (if winreg returned empty or non-Windows)
    if not gpus:
        gpus = _detect_gpus_powershell()

    # Strategy 3: nvidia-smi cross-check (can refine VRAM if winreg reported 0)
    nv_gpus = _detect_gpus_nvidia_smi()
    if nv_gpus:
        # If we didn't find any dedicated GPU yet, use nvidia-smi
        if not any(g.is_dedicated for g in gpus):
            gpus.extend(nv_gpus)
        else:
            # Update matching GPU VRAM if nvidia-smi has more precise info
            for nv in nv_gpus:
                for g in gpus:
                    if nv.name.lower() in g.name.lower() or g.name.lower() in nv.name.lower():
                        if nv.vram_gb > g.vram_gb:
                            g.vram_gb = nv.vram_gb
                            g.is_dedicated = True

    return gpus


def detect_hardware() -> HardwareSpec:
    """Perform full local hardware inspection and return a validated HardwareSpec."""
    total_ram, avail_ram = detect_ram()
    cores, threads, arch, model = detect_cpu()
    has_avx2 = detect_avx2()
    all_gpus = detect_gpus()

    # Select primary GPU: prefer discrete with highest VRAM, else integrated
    primary_gpu: GPUInfo | None = None
    dedicated_gpus = [g for g in all_gpus if g.is_dedicated]
    if dedicated_gpus:
        primary_gpu = max(dedicated_gpus, key=lambda g: g.vram_gb)
    elif all_gpus:
        primary_gpu = all_gpus[0]

    return HardwareSpec(
        total_ram_gb=total_ram,
        available_ram_gb=avail_ram,
        cpu_cores=cores,
        cpu_threads=threads,
        cpu_arch=arch,
        cpu_model=model,
        has_avx2=has_avx2,
        primary_gpu=primary_gpu,
        all_gpus=all_gpus,
    )
