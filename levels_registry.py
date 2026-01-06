import json
import os
import uuid
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import math

# 🔹 IMPORTA TIME MANAGER
from time_manager import TimeManager

DEFAULT_ZONE_WIDTHS = {
    "POC": 0.00035,      # 0.035%
    "VAH": 0.00040,
    "VAL": 0.00040,
    "HVN": 0.00030,
    "LVN": 0.00025,
    "OB_WALL": 0.00015,  # paredes de orderbook
    "ABS_CLUSTER": 0.00020,
    "EXH_CLUSTER": 0.00020,
    "SWING": 0.00030
}

TIMEFRAME_WEIGHT = {"daily": 1.0, "weekly": 1.3, "monthly": 1.5}

@dataclass
class LevelZone:
    id: str
    kind: str                 # POC/VAH/VAL/HVN/LVN/OB_WALL/ABS_CLUSTER/EXH_CLUSTER/SWING
    timeframe: str            # daily/weekly/monthly/intraday
    anchor_price: float       # preço central da zona
    low: float
    high: float
    score: float
    confluence: List[str]     # fontes que compõem a zona
    created_at: str
    last_touched: Optional[str] = None
    touch_count: int = 0
    notes: Optional[str] = ""

    def to_dict(self):
        return asdict(self)

class LevelRegistry:
    def __init__(self, symbol: str, base_dir: str = "./memory", zone_widths: Dict[str, float] = None):
        self.symbol = symbol
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.path = os.path.join(base_dir, f"levels_{symbol}.json")
        self.zones: List[LevelZone] = []
        self.zone_widths = zone_widths or DEFAULT_ZONE_WIDTHS
        self._load()

        # 🔹 Inicializa TimeManager
        self.time_manager = TimeManager()

    def _load(self):
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self.zones = [LevelZone(**z) for z in raw]
            except Exception:
                self.zones = []

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump([z.to_dict() for z in self.zones], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _mk_zone(self, kind: str, timeframe: str, price: float, width_pct: float, confluence: List[str], notes: str = "") -> LevelZone:
        half = max(price * width_pct, 5.0)  # mínimo 5 USD de largura
        low, high = price - half, price + half
        score = 50.0 + 10.0 * len(confluence)
        score *= TIMEFRAME_WEIGHT.get(timeframe, 1.0)
        return LevelZone(
            id=str(uuid.uuid4()),
            kind=kind,
            timeframe=timeframe,
            anchor_price=float(round(price, 2)),
            low=float(round(low, 2)),
            high=float(round(high, 2)),
            score=float(round(score, 2)),
            confluence=confluence[:],
            # 🔹 USA TIME MANAGER
            created_at=self.time_manager.now_iso(),
            notes=notes,
        )

    def _similar(self, a: LevelZone, b: LevelZone) -> bool:
        # zonas com centros muito próximos se fundem
        tol = max(a.anchor_price * 0.0002, 8.0)
        return abs(a.anchor_price - b.anchor_price) <= tol

    def _merge(self, base: LevelZone, other: LevelZone):
        # merge conservador: expande limites e aumenta score/confluence
        base.low = float(round(min(base.low, other.low), 2))
        base.high = float(round(max(base.high, other.high), 2))
        base.anchor_price = float(round((base.anchor_price + other.anchor_price) / 2.0, 2))
        base.confluence = list(sorted(set(base.confluence + other.confluence)))
        base.score = float(round(base.score + 5.0, 2))
        if other.notes:
            base.notes = (base.notes + " | " + other.notes).strip(" | ")

    def add_or_merge(self, zone: LevelZone):
        for z in self.zones:
            if self._similar(z, zone) and z.kind == zone.kind:
                self._merge(z, zone)
                self._save()
                return z
        self.zones.append(zone)
        self._save()
        return zone

    # ---------- Atualizações a partir do contexto ----------
    def update_from_vp(self, historical_vp: Dict):
        if not historical_vp:
            return
        for tf in ["daily", "weekly", "monthly"]:
            vp = historical_vp.get(tf, {})
            if not vp:
                continue
            # POC/VAH/VAL
            for k in ["poc", "vah", "val"]:
                p = vp.get(k)
                if p:
                    z = self._mk_zone(k.upper(), tf, float(p), self.zone_widths[k.upper()], [f"{tf.upper()}_{k.upper()}"])
                    self.add_or_merge(z)
            # HVNs/LVNs
            for p in vp.get("hvns", [])[:30]:
                z = self._mk_zone("HVN", tf, float(p), self.zone_widths["HVN"], [f"{tf.upper()}_HVN"])
                self.add_or_merge(z)
            for p in vp.get("lvns", [])[:30]:
                z = self._mk_zone("LVN", tf, float(p), self.zone_widths["LVN"], [f"{tf.upper()}_LVN"])
                self.add_or_merge(z)

    def add_from_orderbook(self, ob_event: Dict):
        # paredes de compra/venda -> OB_WALL
        alerts = ob_event.get("alertas_liquidez") or []
        for a in alerts:
            if "PAREDE DE COMPRA" in a or "PAREDE DE VENDA" in a:
                # extrai preço no formato "... @ $115,458.21 ..."
                try:
                    price_str = a.split("@ $")[1].split(" ")[0].replace(",", "")
                    price = float(price_str)
                    z = self._mk_zone("OB_WALL", "intraday", price, self.zone_widths["OB_WALL"], ["OB_WALL"], notes=ob_event.get("resultado_da_batalha",""))
                    self.add_or_merge(z)
                except Exception:
                    continue

    def add_from_event(self, ev: Dict):
        # clusters de absorção/exaustão centralizados em poc_price/dwell ou no high/low
        kind = ev.get("tipo_evento","")
        if kind not in ["Absorção", "Exaustão"]:
            return
        price_candidates = []
        if "poc_price" in ev and ev["poc_price"]:
            price_candidates.append(float(ev["poc_price"]))
        if "dwell_price" in ev and ev["dwell_price"]:
            price_candidates.append(float(ev["dwell_price"]))
        # fallback: extremo da janela
        price_candidates += [float(ev.get("preco_minima", 0)), float(ev.get("preco_maxima", 0))]
        price_candidates = [p for p in price_candidates if p > 0]
        if not price_candidates:
            return
        price = sorted(price_candidates, key=lambda x: abs(x - float(ev.get("preco_fechamento", x))))[0]
        zone_kind = "ABS_CLUSTER" if kind == "Absorção" else "EXH_CLUSTER"
        z = self._mk_zone(zone_kind, "intraday", price, self.zone_widths[zone_kind], [kind, ev.get("resultado_da_batalha","")])
        self.add_or_merge(z)

    # ---------- Monitoramento ----------
    def check_price(self, price: float, debounce_seconds: int = 10) -> List[LevelZone]:
        """Retorna zonas tocadas pelo preço (low <= price <= high). Atualiza touches/last_touched."""
        touched = []
        # 🔹 USA TIME MANAGER
        now_iso = self.time_manager.now_iso()
        now_ts = self.time_manager.now() / 1000.0  # converte para segundos
        for z in self.zones:
            if z.low <= price <= z.high:
                # debounce: não contar múltiplos toques em janela curtíssima
                recent = 0
                if z.last_touched:
                    try:
                        recent = now_ts - datetime.fromisoformat(z.last_touched.replace("Z","")).timestamp()
                    except Exception:
                        recent = 999
                if recent < debounce_seconds:
                    continue
                z.touch_count += 1
                z.last_touched = now_iso
                z.score = float(round(z.score + 1.0, 2))
                touched.append(z)
        if touched:
            self._save()
        return touched

    def list_zones(self) -> List[Dict]:
        return [z.to_dict() for z in sorted(self.zones, key=lambda x: (-x.score, x.anchor_price))]