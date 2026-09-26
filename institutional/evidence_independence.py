# institutional/evidence_independence.py
"""
P1-C — Independence Policy v1 (camada PURA de análise; sem decisão, sem score).

Pergunta respondida: "esses campos possuem os mesmos ancestrais, ancestrais
parcialmente compartilhados, ancestrais disjuntos ou lineage desconhecida?"
NUNCA: quantos votos, confiança, peso, cap ou recomendação de operação.

Semântica static vs runtime (§1 do contrato P1-C):
- `taxonomy.derived_from` = STATIC_POTENTIAL_DEPENDENCIES ("este campo PODE
  ser derivado destas fontes"; ex. whale.score lista CVD mesmo sendo fallback).
  Não afirma que todas participaram numa observação.
- `runtime_contributors` (opcional, por observação) = contribuintes reais
  (`{field_id: [ancestor_ids...]}`); quando fornecido p/ um campo, a análise
  usa esses contribuintes literais (sem travessia) com basis RUNTIME_OBSERVED.
  Nunca inventado aqui.

Regra crítica: DISJOINT_LINEAGE != estatisticamente independente. Significa
só "registry não encontrou ancestral primário compartilhado". O nome
INDEPENDENT é proibido neste módulo.

Folha (só stdlib + taxonomy/evidence). Sem ciclo. Sem tocar confluence,
event_bridge, payload, prompt, weights, TTL, caps, dedup ou counts_as_vote.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from institutional import evidence_taxonomy as tx


class Relation(str, Enum):
    """Relação de lineage entre dois fields (nunca 'INDEPENDENT')."""
    IDENTICAL_LINEAGE = "identical_lineage"
    OVERLAPPING_LINEAGE = "overlapping_lineage"
    DISJOINT_LINEAGE = "disjoint_lineage"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"


class Redundancy(str, Enum):
    """Só EXACT conta; overlap parcial nunca é redundante."""
    REDUNDANT_EXACT = "redundant_exact"
    NOT_REDUNDANT = "not_redundant"
    UNKNOWN = "unknown"


class LineageBasis(str, Enum):
    STATIC_POTENTIAL = "static_potential"
    RUNTIME_OBSERVED = "runtime_observed"


def _is_composite(field_id: str) -> Optional[bool]:
    """True/False se registrado; None se desconhecido."""
    entry = tx.FIELDS.get(field_id)
    return entry.is_composite if entry is not None else None


def _static_ancestors(field_id: str) -> Optional[frozenset]:
    """Ancestrais estáticos ou None se desconhecido/vazio (UNKNOWN)."""
    if field_id not in tx.FIELDS:
        return None
    ancestors = tx.primary_ancestors(field_id)
    return ancestors if ancestors else None


def _side_ancestors(field_id: str, runtime: Optional[dict],
                    ) -> tuple:
    """(ancestors|None, runtime_bool). Runtime usa lista literal, sem travessia."""
    if runtime is not None and field_id in runtime:
        contributors = runtime[field_id] or []
        return frozenset(str(c) for c in contributors), True
    return _static_ancestors(field_id), False


def eligible_for_independent_count(field_id: str) -> bool:
    """Só metadata de relatório/policy. Composite, desconhecido ou sem
    ancestrais provados => False. NUNCA toca Evidence.counts_as_vote."""
    entry = tx.FIELDS.get(field_id)
    if entry is None or entry.is_composite:
        return False
    ancestors = tx.primary_ancestors(field_id)
    return bool(ancestors)


def classify_relation(field_a: str, field_b: str,
                      runtime_contributors: Optional[dict] = None) -> dict:
    """Classificador puro par-a-par (ordem de regras do contrato P1-C §3).

    Family/horizon NUNCA sobrescrevem lineage (vão só em detalhes).
    Sem score/confidence/weight.
    """
    runtime = runtime_contributors or {}
    ancestors_a, rt_a = _side_ancestors(field_a, runtime or None)
    ancestors_b, rt_b = _side_ancestors(field_b, runtime or None)
    basis = (LineageBasis.RUNTIME_OBSERVED
             if (rt_a and rt_b) else LineageBasis.STATIC_POTENTIAL)
    comp_a = _is_composite(field_a)
    comp_b = _is_composite(field_b)

    def _detail(relation: Relation, shared: frozenset) -> dict:
        return {
            "relation": relation.value,
            "lineage_basis": basis.value,
            "ancestors_a": sorted(ancestors_a) if ancestors_a is not None else [],
            "ancestors_b": sorted(ancestors_b) if ancestors_b is not None else [],
            "shared_ancestors": sorted(shared),
            "is_composite_a": comp_a,
            "is_composite_b": comp_b,
            "eligible_for_independent_count": {
                "a": eligible_for_independent_count(field_a),
                "b": eligible_for_independent_count(field_b),
            },
        }

    if field_a == field_b:
        shared = ancestors_a if ancestors_a is not None else frozenset()
        return _detail(Relation.IDENTICAL_LINEAGE, shared)
    if ancestors_a is None or ancestors_b is None:
        return _detail(Relation.UNKNOWN, frozenset())
    if comp_a or comp_b:
        return _detail(Relation.COMPOSITE, ancestors_a & ancestors_b)
    if ancestors_a == ancestors_b and ancestors_a:
        return _detail(Relation.IDENTICAL_LINEAGE, ancestors_a)
    if ancestors_a & ancestors_b:
        return _detail(Relation.OVERLAPPING_LINEAGE, ancestors_a & ancestors_b)
    if ancestors_a and ancestors_b:
        return _detail(Relation.DISJOINT_LINEAGE, frozenset())
    return _detail(Relation.UNKNOWN, frozenset())


def redundancy(field_a: str, field_b: str,
               runtime_contributors: Optional[dict] = None) -> str:
    """REDUNDANT_EXACT só p/ fields distintos com o mesmo conjunto NÃO VAZIO
    de ancestrais. Overlap parcial nunca; identidade não precisa do rótulo."""
    if field_a == field_b:
        return Redundancy.NOT_REDUNDANT.value
    runtime = runtime_contributors or {}
    ancestors_a, _ = _side_ancestors(field_a, runtime or None)
    ancestors_b, _ = _side_ancestors(field_b, runtime or None)
    if ancestors_a is None or ancestors_b is None:
        return Redundancy.UNKNOWN.value
    if ancestors_a and ancestors_a == ancestors_b:
        return Redundancy.REDUNDANT_EXACT.value
    return Redundancy.NOT_REDUNDANT.value


def analyze_evidence_set(field_ids: list,
                          runtime_contributors: Optional[dict] = None) -> dict:
    """Relatório determinístico (NÃO decisão): sem confidence/votos/score/
    peso/cap/recomendação. Inclui horizons só p/ explicação."""
    unique_fields: list = []
    for fid in field_ids or []:
        if fid not in unique_fields:
            unique_fields.append(fid)
    runtime = runtime_contributors or {}
    basis = (LineageBasis.RUNTIME_OBSERVED.value
             if runtime and all(f in runtime for f in unique_fields)
             else LineageBasis.STATIC_POTENTIAL.value)

    families_present: set = set()
    fields_by_family: dict = {}
    composites: list = []
    unknown_fields: list = []
    ineligible_fields: list = []
    coverage: dict = {}
    horizons: dict = {}
    for fid in unique_fields:
        entry = tx.FIELDS.get(fid)
        if entry is None:
            unknown_fields.append(fid)
            ineligible_fields.append(fid)
            coverage[fid] = []
            horizons[fid] = None
            continue
        families_present.add(entry.family.value)
        fields_by_family.setdefault(entry.family.value, []).append(fid)
        if entry.is_composite:
            composites.append(fid)
            ineligible_fields.append(fid)
        try:
            ancestors = tx.primary_ancestors(fid)
        except (KeyError, ValueError):
            ancestors = frozenset()
        if not ancestors:
            unknown_fields.append(fid)
            ineligible_fields.append(fid)
            coverage[fid] = []
        else:
            coverage[fid] = sorted(ancestors)
        horizons[fid] = entry.horizon_ms

    identical_groups: dict = {}
    overlap_edges: list = []
    disjoint_pairs: list = []
    for i in range(len(unique_fields)):
        for j in range(i + 1, len(unique_fields)):
            a, b = unique_fields[i], unique_fields[j]
            detail = classify_relation(a, b, runtime_contributors)
            rel = detail["relation"]
            if rel == Relation.IDENTICAL_LINEAGE.value and a != b:
                key = (a, b) if a < b else (b, a)
                identical_groups.setdefault(key, detail["shared_ancestors"])
            elif rel == Relation.OVERLAPPING_LINEAGE.value:
                overlap_edges.append({"a": a, "b": b,
                                      "shared": detail["shared_ancestors"]})
            elif rel == Relation.DISJOINT_LINEAGE.value:
                disjoint_pairs.append([a, b])
    exact_groups: dict = {}
    for (a, b), shared in identical_groups.items():
        if redundancy(a, b, runtime_contributors) == Redundancy.REDUNDANT_EXACT.value:
            exact_groups.setdefault(tuple(shared), []).append([a, b])

    def _sort_pairs(pairs: list) -> list:
        return sorted(pairs, key=lambda p: (str(p[0]), str(p[1])))

    return {
        "unique_fields": unique_fields,
        "families_present": sorted(families_present),
        "fields_by_family": {k: sorted(v) for k, v in sorted(fields_by_family.items())},
        "composites": sorted(composites),
        "ineligible_fields": sorted(set(ineligible_fields)),
        "exact_redundancy_groups": {", ".join(k): _sort_pairs(v)
                                    for k, v in sorted(exact_groups.items())},
        "overlap_edges": sorted(overlap_edges,
                                key=lambda e: (str(e["a"]), str(e["b"]))),
        "disjoint_pairs": _sort_pairs(disjoint_pairs),
        "unknown_fields": sorted(set(unknown_fields)),
        "primary_ancestor_coverage": {k: coverage[k] for k in unique_fields},
        "horizons": {k: horizons[k] for k in unique_fields},
        "lineage_basis": basis,
    }
