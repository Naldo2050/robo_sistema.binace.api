# institutional/confluence_shadow.py
"""
P2-A — Confluence Shadow v1: Evidence Reconciler (somente observação).

Pergunta respondida: "as evidências direcionais válidas observadas nesta janela
estão alinhadas, em direções mistas, ou insuficientes?"
NUNCA: que trade abrir, quando entrar, probabilidade de lucro, confidence, peso.

Contrato:
- Aceita SOMENTE instâncias de Evidence v1.0.0 com campo `source` preenchido.
- Evidências com validity != VALID são descartadas e listadas em discarded_non_valid.
- Composites (is_composite via taxonomy) nunca alteram status direcional.
- Redundância exata (P1-C) colapsa fields em grupos representativos.
- Overlapping lineage é registrada mas NUNCA decide "independent confirmation".
- DISJOINT_LINEAGE nunca é interpretada como "estatisticamente independente".
- independent_confirmation permanece False nesta versão.
- Nenhum threshold de famílias (1 evidência válida pode gerar ALIGNED_*).
- Nenhum peso, TTL, decay, confidence contínua, probability ou score numérico.
- Nenhuma alteração ao confluence_engine.py legado.

absorption.current:
- Label/direction canônica: Absorção de Compra => BEARISH, Absorção de Venda => BULLISH.
- Magnitude NÃO validada (P0-A2).
- Calibration = UNCALIBRATED_HEURISTIC.
- Lineage sobrepõe executed_flow (OVERLAPPING_LINEAGE).
- Dentro de whale.score é NON_VOTING_UNVALIDATED_MAGNITUDE.
- Pode aparecer como observação direcional no reconciler, com metadados
  de calibração e overlap preservados; NÃO pode ser promovida a independent
  confirmation de fluxo.

Folha (só stdlib + institutional.evidence + institutional.evidence_taxonomy +
institutional.evidence_independence). Sem ciclo. Sem tocar confluence_engine,
event_bridge, payload, prompt, weights, trading, LLM, risk, execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple

from institutional.evidence import (
    Evidence,
    EvidenceCalibration,
    EvidenceDirection,
    EvidenceFamily,
    EvidenceType,
    EvidenceValidity,
    FEATURE_CONTRACT_VERSION,
)
from institutional import evidence_taxonomy as tx
from institutional import evidence_independence as indep


# ── STATUS OBSERVACIONAIS ──────────────────────────────────────────────────────

class ReconcilerStatus(str, Enum):
    """Classificação observacional do alinhamento de evidências.

    Não implica ação, trade, ou probabilidade. Descreve somente o estado
    qualitativo do conjunto observado.
    """
    ALIGNED_BULLISH = "aligned_bullish"
    ALIGNED_BEARISH = "aligned_bearish"
    MIXED_DIRECTIONS = "mixed_directions"
    NEUTRAL_ONLY = "neutral_only"
    INSUFFICIENT_DATA = "insufficient_data"


# ── RESULTADO IMUTÁVEL ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConfluenceShadowResult:
    """Resultado determinístico do Evidence Reconciler v1.

    Sem confidence/probability/weight/trade/action/entry/score.
    independent_confirmation permanece False nesta versão.
    """
    # Identificação temporal e de escopo
    symbol: str
    observation_open_ms: int
    observation_close_ms: int
    causal_anchor_ms: int

    # Classificação observacional
    status: ReconcilerStatus

    # Evidências categorizadas
    valid_directional_evidence: Tuple[dict, ...] = ()
    neutral_evidence: Tuple[dict, ...] = ()
    discarded_non_valid: Tuple[dict, ...] = ()

    # Lineage e redundância (P1-C)
    exact_redundancy_groups: Tuple[dict, ...] = ()
    overlap_relations: Tuple[dict, ...] = ()
    disjoint_relations: Tuple[dict, ...] = ()
    unknown_lineage_relations: Tuple[dict, ...] = ()

    # Composites informativos (nunca alteram status)
    observed_composites: Tuple[dict, ...] = ()

    # Resumo estruturado
    directions_present: Tuple[str, ...] = ()
    families_present: Tuple[str, ...] = ()
    evidence_count: int = 0
    computationally_disjoint_groups: Tuple[Tuple[str, ...], ...] = ()

    # Contrato: sempre False nesta versão
    independent_confirmation: bool = False

    # Fatos estruturados (sem prosa decisória)
    summary_facts: dict = field(default_factory=dict)

    # Versionamento
    feature_contract_version: str = FEATURE_CONTRACT_VERSION
    reconciler_version: str = "1.0.0"


# ── WHITELIST DE EVIDÊNCIAS ELEGÍVEIS ──────────────────────────────────────────

# Fields primários que podem participar como observação direcional.
# Composites (is_composite=True na taxonomy) são excluídos por construção.
# Não é exaustiva: fields fora desta lista são rejeitados como "not_whitelisted".
DIRECTIONAL_WHITELIST: frozenset = frozenset({
    "flow.net.1m",
    "flow.imbalance.1m",
    "flow.buy_sell_ratio",
    "absorption.current",
    "orderbook.snapshot.imbalance",
    "market_structure.bos",
    "market_structure.sweep",
})


# ── ENGINE ─────────────────────────────────────────────────────────────────────

def _evidence_summary(ev: Evidence) -> dict:
    """Resumo mínimo de uma evidência para o resultado (sem objetos mutáveis)."""
    return {
        "source": ev.source,
        "family": ev.family.value,
        "evidence_type": ev.evidence_type.value,
        "direction": ev.direction.value,
        "validity": ev.validity.value,
        "calibration": ev.calibration.value,
        "counts_as_vote": ev.counts_as_vote,
        "observed_at_ms": ev.observed_at_ms,
    }


def _is_taxonomy_composite(field_id: str) -> bool:
    """Retorna True se o field_id é composite na taxonomy registrada."""
    entry = tx.FIELDS.get(field_id)
    return entry.is_composite if entry is not None else False


def _classify_evidence(ev: Evidence) -> str:
    """Classifica uma evidência para fins de triagem interna.

    Retorna: 'valid_directional', 'neutral', 'non_valid', 'composite',
             'not_whitelisted'.

    Ordem de regras:
    1. Validade: non-VALID é descartado imediatamente.
    2. Whitelist: se o source está na DIRECTIONAL_WHITELIST, ele participa
       como observação direcional MESMO se is_composite=True na taxonomy.
       Exemplo: absorption.current é composite (usa múltiplos inputs) mas
       tem label/direction canônica provada no P0 — pode aparecer como
       observação direcional com calibration=UNCALIBRATED_HEURISTIC e
       relation_to_flow=OVERLAPPING_LINEAGE (sem promoção a independent
       confirmation).
    3. Composite: taxonomy is_composite fora da whitelist -> non-voting.
    4. Fonte desconhecida -> not_whitelisted.
    """
    # 1. Validade
    if ev.validity != EvidenceValidity.VALID:
        return "non_valid"

    # 2. Whitelist tem prioridade sobre composite
    if ev.source in DIRECTIONAL_WHITELIST:
        if ev.direction in (EvidenceDirection.BULLISH, EvidenceDirection.BEARISH):
            return "valid_directional"
        if ev.direction == EvidenceDirection.NEUTRAL:
            return "neutral"
        # UNKNOWN direction com validade VALID -> não participante
        return "non_valid"

    # 3. Composite (pela taxonomy) fora da whitelist -> non-voting
    if _is_taxonomy_composite(ev.source):
        return "composite"

    # 4. Fonte não registrada na whitelist
    return "not_whitelisted"


def _build_redundancy_groups(
    valid_sources: list[str],
) -> tuple[list[dict], list[str]]:
    """Identifica REDUNDANT_EXACT entre sources válidos.

    Retorna: (groups, suppressed_aliases).
    Cada group: {representative, suppressed_aliases, shared_ancestors}.
    """
    if len(valid_sources) < 2:
        return [], []

    groups: dict[frozenset, list[str]] = {}
    for src in valid_sources:
        entry = tx.FIELDS.get(src)
        if entry is None:
            continue
        try:
            ancestors = tx.primary_ancestors(src)
        except (KeyError, ValueError):
            continue
        if ancestors:
            groups.setdefault(ancestors, []).append(src)

    result_groups: list[dict] = []
    all_suppressed: list[str] = []
    for ancestors, members in groups.items():
        if len(members) < 2:
            continue
        representative = members[0]
        suppressed = members[1:]
        result_groups.append({
            "representative": representative,
            "suppressed_aliases": tuple(sorted(suppressed)),
            "shared_ancestors": tuple(sorted(ancestors)),
        })
        all_suppressed.extend(suppressed)

    return result_groups, all_suppressed


def _build_lineage_relations(
    active_sources: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Classifica relações de lineage par-a-par entre sources ativos.

    Retorna: (overlap_relations, disjoint_relations, unknown_relations).
    Nunca retorna 'independent=True'.

    COMPOSITE do P1-C (quando um dos campos é is_composite na taxonomy) é
    tratado como relação informativa de overlap no reconciler quando os
    ancestrais são compartilhados — porque indica linhagem computacionalmente
    sobreposta. Exemplo: absorption.current (composite) vs flow.net.1m
    compartilham raw.aggtrade.* -> registrado como OVERLAPPING_LINEAGE.
    """
    overlaps: list[dict] = []
    disjoints: list[dict] = []
    unknowns: list[dict] = []

    for i in range(len(active_sources)):
        for j in range(i + 1, len(active_sources)):
            a, b = active_sources[i], active_sources[j]
            detail = indep.classify_relation(a, b)
            rel = detail["relation"]

            entry = {
                "field_a": a,
                "field_b": b,
                "relation": rel,
                "shared_ancestors": detail.get("shared_ancestors", []),
                "computationally_disjoint": rel == indep.Relation.DISJOINT_LINEAGE.value,
            }

            if rel == indep.Relation.OVERLAPPING_LINEAGE.value:
                overlaps.append(entry)
            elif rel == indep.Relation.COMPOSITE.value:
                # COMPOSITE com ancestrais compartilhados indica overlap
                # computacional — registrar como overlap informativo.
                if detail.get("shared_ancestors"):
                    entry["relation"] = indep.Relation.OVERLAPPING_LINEAGE.value
                    entry["composite_source"] = True
                    overlaps.append(entry)
                else:
                    unknowns.append(entry)
            elif rel == indep.Relation.DISJOINT_LINEAGE.value:
                disjoints.append(entry)
            elif rel == indep.Relation.UNKNOWN.value:
                unknowns.append(entry)
            # IDENTICAL_LINEAGE handled by redundancy groups

    return overlaps, disjoints, unknowns


def _compute_disjoint_groups(
    active_sources: list[str],
    disjoint_relations: list[dict],
) -> list[tuple[str, ...]]:
    """Agrupa sources que são computationally disjoint entre si.

    NÃO implica independência estatística. Apenas registra que os ancestrais
    primários registrados não se sobrepõem.
    """
    if not disjoint_relations:
        return []

    # Build adjacency of disjoint pairs
    disjoint_pairs: set[tuple[str, str]] = set()
    for rel in disjoint_relations:
        a, b = rel["field_a"], rel["field_b"]
        disjoint_pairs.add((min(a, b), max(a, b)))

    # Simple greedy grouping: cada grupo contém sources que são
    # todos pairwise disjoint
    groups: list[list[str]] = []
    for src in sorted(active_sources):
        placed = False
        for g in groups:
            if all((min(src, m), max(src, m)) in disjoint_pairs for m in g):
                g.append(src)
                placed = True
                break
        if not placed:
            groups.append([src])

    return [tuple(g) for g in groups if len(g) >= 2]


def reconcile(
    evidences: list[Evidence],
    symbol: str,
    observation_open_ms: int,
    observation_close_ms: int,
    causal_anchor_ms: int,
) -> ConfluenceShadowResult:
    """Reconcilia um conjunto de evidências e retorna classificação observacional.

    Determinístico: mesma entrada -> mesma saída. Sem randomização, sem state,
    sem side effects, sem I/O.

    Não produz: pesos, confidence, probability, score, trade action, entry/exit,
    threshold, TTL, decay. independent_confirmation = False sempre.
    """
    # ── Triagem ────────────────────────────────────────────────────────────
    valid_directional: list[Evidence] = []
    neutral_ev: list[Evidence] = []
    discarded: list[Evidence] = []
    composites: list[Evidence] = []

    for ev in evidences:
        cat = _classify_evidence(ev)
        if cat == "valid_directional":
            valid_directional.append(ev)
        elif cat == "neutral":
            neutral_ev.append(ev)
        elif cat == "composite":
            composites.append(ev)
        else:
            # non_valid ou not_whitelisted
            discarded.append(ev)

    # ── Redundância exata ──────────────────────────────────────────────────
    directional_sources = [ev.source for ev in valid_directional]
    redundancy_groups, suppressed_aliases = _build_redundancy_groups(
        directional_sources
    )

    # Filtrar suppressed do conjunto ativo (para lineage e status)
    suppressed_set = set(suppressed_aliases)
    active_directional = [
        ev for ev in valid_directional if ev.source not in suppressed_set
    ]
    active_sources = [ev.source for ev in active_directional]

    # ── Lineage ────────────────────────────────────────────────────────────
    overlaps, disjoints, unknowns = _build_lineage_relations(active_sources)
    disjoint_groups = _compute_disjoint_groups(active_sources, disjoints)

    # ── Direções presentes (entre ativos, após dedup) ──────────────────────
    directions: set[str] = set()
    for ev in active_directional:
        directions.add(ev.direction.value)
    for ev in neutral_ev:
        directions.add(ev.direction.value)

    # ── Famílias presentes ─────────────────────────────────────────────────
    families: set[str] = set()
    for ev in active_directional:
        families.add(ev.family.value)
    for ev in neutral_ev:
        families.add(ev.family.value)

    # ── Status observacional ───────────────────────────────────────────────
    has_bullish = any(
        ev.direction == EvidenceDirection.BULLISH for ev in active_directional
    )
    has_bearish = any(
        ev.direction == EvidenceDirection.BEARISH for ev in active_directional
    )
    has_neutral = len(neutral_ev) > 0
    has_any_directional = len(active_directional) > 0

    if has_bullish and has_bearish:
        status = ReconcilerStatus.MIXED_DIRECTIONS
    elif has_bullish:
        status = ReconcilerStatus.ALIGNED_BULLISH
    elif has_bearish:
        status = ReconcilerStatus.ALIGNED_BEARISH
    elif has_neutral and not has_any_directional:
        status = ReconcilerStatus.NEUTRAL_ONLY
    else:
        status = ReconcilerStatus.INSUFFICIENT_DATA

    # ── Summary facts (estruturado, sem prosa decisória) ───────────────────
    facts: list[str] = []
    for ev in active_directional:
        fact = f"{ev.source}_direction={ev.direction.value.upper()}"
        if ev.calibration == EvidenceCalibration.UNCALIBRATED_HEURISTIC:
            fact += "|calibration=UNCALIBRATED_HEURISTIC"
        if ev.evidence_type == EvidenceType.POINT_IN_TIME_L2:
            fact += "|scope=POINT_IN_TIME_L2"
        facts.append(fact)

    for rel in overlaps:
        facts.append(
            f"{rel['field_a']}_vs_{rel['field_b']}=OVERLAPPING_LINEAGE"
        )

    for ev in composites:
        facts.append(f"{ev.source}=COMPOSITE_NON_VOTING")

    summary = {
        "directions_present": sorted(directions),
        "status": status.value,
        "facts": facts,
    }

    # ── Construção do resultado imutável ───────────────────────────────────
    return ConfluenceShadowResult(
        symbol=symbol,
        observation_open_ms=observation_open_ms,
        observation_close_ms=observation_close_ms,
        causal_anchor_ms=causal_anchor_ms,
        status=status,
        valid_directional_evidence=tuple(
            _evidence_summary(ev) for ev in valid_directional
        ),
        neutral_evidence=tuple(
            _evidence_summary(ev) for ev in neutral_ev
        ),
        discarded_non_valid=tuple(
            _evidence_summary(ev) for ev in discarded
        ),
        exact_redundancy_groups=tuple(redundancy_groups),
        overlap_relations=tuple(overlaps),
        disjoint_relations=tuple(disjoints),
        unknown_lineage_relations=tuple(unknowns),
        observed_composites=tuple(
            _evidence_summary(ev) for ev in composites
        ),
        directions_present=tuple(sorted(directions)),
        families_present=tuple(sorted(families)),
        evidence_count=len(active_directional),
        computationally_disjoint_groups=tuple(disjoint_groups),
        independent_confirmation=False,
        summary_facts=summary,
    )
