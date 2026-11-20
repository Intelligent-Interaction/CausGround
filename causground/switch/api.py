"""Task-only SWITCH retrieval, with separate model text and structured evidence."""
from dataclasses import dataclass
from copy import deepcopy
import gzip
import json
from pathlib import Path
from . import _parser
from ._retrieval import GlobalCWMIndex

LIBRARY_SCHEMA = 'switch-cwm-library-v1'

@dataclass
class RetrievalResult:
    subgraph: dict
    context: str


def _read(path):
    path = Path(path)
    raw = path.read_bytes()
    data = gzip.decompress(raw) if path.suffix == '.gz' else raw
    return json.loads(data)


def _goals(value):
    if not isinstance(value, dict) or set(value) != {'goal_facts'} or not isinstance(value['goal_facts'], list):
        raise ValueError('goals must be {"goal_facts": [{"object": ..., "key": ..., "value": ...}]}')
    for fact in value['goal_facts']:
        if (not isinstance(fact, dict) or set(fact) != {'object', 'key', 'value'}
                or not isinstance(fact['object'], str) or not fact['object'].strip()
                or not isinstance(fact['key'], str)):
            raise ValueError('Each goal fact needs object, key, and value')
    # Match JSON serialization used by source retrieval and reject nonfinite numbers.
    json.dumps(value, allow_nan=False)
    return deepcopy(value)


def _validate_source_strings(value):
    """The frozen source grammar is line-based; reject unrepresentable payloads."""
    if isinstance(value, str):
        if any(c in value for c in ('\r', '\n', '\x00')):
            raise ValueError('Source strings cannot contain CR, LF, or NUL; use single-line source values')
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_source_strings(key)
            _validate_source_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_source_strings(item)


class SwitchCWM:
    """Load once, then retrieve(task). No model, GPU, evaluation or network dependency.

    Optional goals are typed facts from the task, not observations or answer choices.
    A bundled exact-task goal cache is used unless explicit goals are supplied.
    A cache miss still performs lexical retrieval and follows source dependencies.
    """
    def __init__(self, graph):
        if not isinstance(graph, dict) or graph.get('schema_version') != LIBRARY_SCHEMA:
            raise ValueError('Expected a switch-cwm-library-v1 resource')
        self.profile = 'switch'
        self.domain = 'SWITCH'
        graph = deepcopy(graph)
        entries = graph.get('occurrences')
        if not isinstance(entries, list) or not entries:
            raise ValueError('The library needs at least one source occurrence')
        rows, actions, produced = [], [], []
        for entry in entries:
            row, action, facts = entry['row'], entry['action'], entry['produced']
            _validate_source_strings(entry)
            if not isinstance(row.get('id'), str) or not isinstance(row.get('record_id'), str):
                raise ValueError('Source occurrence and record IDs must be strings')
            if any(not isinstance(f, (list, tuple)) or len(f) != 4 or f[-1] not in _parser.BASES for f in facts):
                raise ValueError('Produced facts require [node, attribute, value, source_basis]')
            # Reject facts beyond the inherited serializer limit instead of hiding them.
            if len(facts) > 16:
                raise ValueError('Source serializer supports at most 16 produced facts per occurrence')
            rows.append(row); actions.append(action); produced.append(facts)
        self._goals = {task: _goals(goal) for task, goal in graph.get('task_goals', {}).items()}
        self._provenance = {e['row']['id']: deepcopy(e.get('provenance', {})) for e in entries}
        self._index = GlobalCWMIndex(rows, actions, produced)

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = Path(__file__).resolve().parents[2] / 'data' / 'switch' / 'causground.json.gz'
        return cls(_read(path))

    def retrieve(self, task, *, version=None, goals=None, top_k=4, context_chars=14000):
        """Return evidence subgraph + prompt text, retaining exact selected source facts.

        version selects simple (default) or complex.
        context_chars caps whole-action selection in the chosen format.
        This is not a tokenizer limit. The caller must fit context + media + question
        + generation within their model's window. No model prompt is built here.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError('task must be nonempty text')
        for name, value in [('top_k', top_k), ('context_chars', context_chars)]:
            if type(value) is not int or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if context_chars > 14000:
            raise ValueError('context_chars must be at most 14000 for the frozen selection policies')
        origin = 'explicit_task_goals' if goals is not None else ('bundled_task_cache' if task in self._goals else 'absent')
        goal = _goals(goals) if goals is not None else deepcopy(self._goals.get(task))
        if version not in (None, 'simple', 'complex'):
            raise ValueError('version must be simple or complex')
        version = version or 'simple'
        result = self._index.retrieve(task, goal, top_k=top_k, context_chars=context_chars,
                                      detailed=version == 'complex')
        context, audit = result['context'], result['audit']
        addition_audit = audit['renderer_audit'] if version == 'simple' else None
        audit.update(goal_source=origin, goal_facts=deepcopy((goal or {}).get('goal_facts', [])),
                     version=version, additions='strict' if version == 'simple' else None,
                     augmentation=addition_audit,
                     rendered_context_characters=len(context),
                     goal_dependency_policy='expand_newly_selected_goal_predecessors',
                     local_graph_used=False, execution_verified=False)
        included = set(result['subgraph']['occurrence_ids'])
        nodes = []
        inventory = audit['renderer_audit']['inventory']['actions']
        for entry, parsed in zip(result['subgraph']['occurrences'], inventory):
            occurrence_id = entry['row']['id']
            nodes.append({'id': occurrence_id, 'source_record_id': entry['row']['record_id'],
                          'source_step_index': entry['row'].get('step_index'),
                          'action': parsed, 'source_inventory': entry,
                          'provenance': deepcopy(self._provenance[occurrence_id]),
                          'retrieved_by': audit['reasons'][occurrence_id],
                          'execution_verified': False})
        edges = [{'source': link['producer'], 'target': link['consumer'],
                  'relation': 'same_record_prestate_match', 'required': link['required'],
                  'witnesses': link['witnesses']}
                 for link in audit['dependency_links'] if link['producer'] in included and link['consumer'] in included]
        graph = {'schema_version': 'causground-switch-task-subgraph-v1', 'domain': 'SWITCH',
                 'task': task, 'nodes': nodes, 'edges': edges, 'audit': audit}
        # Goal/dependency witness caches belong to the loaded library. Never let
        # callers mutate them through a returned audit, even on repeated queries.
        return RetrievalResult(subgraph=deepcopy(graph), context=context)
