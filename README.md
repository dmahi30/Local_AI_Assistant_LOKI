# LOKI: Local Offline Knowledge & Interface

LOKI is a hardware-adaptive, privacy-focused, offline-first local AI computer assistant for desktop/laptop computers.

## Phase 1: Foundation Architecture

This repository currently contains the **Phase 1 Foundation**:
- **Core Types & Enums (`loki/core/types.py`)**: Hardware tiers (`ECO`, `BALANCED_CPU`, `BALANCED_GPU`, `POWER`), Safety tiers (Safe, Notify, Dangerous), Intent types, and Tool envelopes.
- **Privacy-Safe Offline Logger (`loki/core/logging.py`)**: Local rotating logs in `%LOCALAPPDATA%\LOKI\logs\loki.log` with automatic username/home path sanitization and zero telemetry.
- **Domain Exceptions (`loki/core/exceptions.py`)**: Structured `LokiError` hierarchy for clean error handling and graceful fallbacks.
- **Configuration Manager (`loki/core/config.py`)**: Hardware-adaptive settings, model profiles, and JSON persistence (`%LOCALAPPDATA%\LOKI\config.json`).
- **Path Utilities (`loki/utils/paths.py`)**: Safe directory scaffolding and path resolution.

## Running Tests

```powershell
python -m pytest tests -v
```
