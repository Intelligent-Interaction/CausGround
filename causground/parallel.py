"""Bounded CPU ingestion; serial and parallel builds share the same learner."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

from .builder import CausGroundBuilder


def _prepare_batch(rows, effect_policy, evidence_mode):
    builder = CausGroundBuilder(effect_policy=effect_policy)
    evidence = []
    for row in rows:
        try:
            value = builder.add(row)
        except Exception as exc:
            raise ValueError(f"Transition {row.get('id')}: {exc}") from exc
        if evidence_mode == "source":
            value = {key: value[key] for key in (
                "id", "action", "outcome", "provenance", "head", "effect_bindings", "interventions") if key in value}
        evidence.append(value)
    return evidence, builder._records, builder._sources


def prepare_transitions(rows, builder, workers, evidence_mode):
    """Keep at most two small batches per worker queued, regardless of corpus size."""
    rows = iter(rows)
    pending = deque()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < workers * 2:
                batch = list(islice(rows, 8))
                if not batch:
                    exhausted = True
                    break
                pending.append(executor.submit(_prepare_batch, batch, builder.effect_policy, evidence_mode))
            if not pending:
                break
            evidence, records, sources = pending.popleft().result()
            for identity, record in records.items():
                if identity in builder._sources and builder._sources[identity] != sources[identity]:
                    raise ValueError(f"conflicting evidence for id {identity}")
                builder._records[identity] = record
                builder._sources[identity] = sources[identity]
            yield from evidence
