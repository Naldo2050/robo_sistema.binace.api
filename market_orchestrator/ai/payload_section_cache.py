# -*- coding: utf-8 -*-
"""
Cache leve por seção para reduzir reenvio de blocos estáveis no payload.
Evita dependências pesadas e usa escrita atômica para não corromper o arquivo.
"""

from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def canonical_ref(data: Dict[str, Any]) -> str:
    """Gera hash estável do conteúdo da seção."""
    try:
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = str(data)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_fresh(saved_at_ms: Optional[int], ttl_s: int, now_ms: int) -> bool:
    if saved_at_ms is None:
        return False
    if ttl_s <= 0:
        return False
    return (now_ms - saved_at_ms) <= ttl_s * 1000


class SectionCache:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            if not self.path.exists():
                return {}
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("SectionCache load failed: %s", e)
            return {}

    def _save(self) -> None:
        try:
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(self.path)
        except Exception as e:
            logger.warning("SectionCache save failed: %s", e)

    def get(self, section_key: str) -> Optional[Dict[str, Any]]:
        entry = self._data.get(section_key)
        return entry if isinstance(entry, dict) else None

    def set(self, section_key: str, ref: str, saved_at_ms: int, data: Dict[str, Any]) -> None:
        self._data[section_key] = {
            "ref": ref,
            "saved_at_ms": saved_at_ms,
            "data": data,
        }
        self._save()
