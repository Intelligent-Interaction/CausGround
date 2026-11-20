"""Original task-only TF-IDF selection, shared by simple and complex."""
from collections import defaultdict
import json
import re

from . import _source as source, _parser as parser, _strict as renderer

CWM_HEADER = 'TASK_CONDITIONED_CWM\nretrieval=task_only_global_repaired\nraw_source_text_exposed=false'
DETAILED_HEADER = (
    'TASK_CONDITIONED_CWM\n'
    'retrieval=task_text_tfidf_node_topk_with_local_sequence_and_causal_context '
    'UNION independent_global_typed_goal_state_backward_closure\n'
    'raw_source_text_exposed=false'
)


class GlobalCWMIndex:
    def __init__(self, rows, actions, produced):
        self.rows, self.actions, self.produced = rows, actions, produced
        if len({row['id'] for row in rows}) != len(rows):
            raise ValueError('duplicate source occurrence ID')
        self.index = source.TfidfIndex([
            source.node_doc(action, facts)
            for action, facts in zip(actions, produced)
        ])
        self.blocks = []
        self.rule_parts = []
        for row, action, facts in zip(rows, actions, produced):
            block = source.render_action(1, row, action, facts, {'task_fuzzy_node'})
            text, audit = renderer.render('source-only rendering', CWM_HEADER + '\n\n' + block)
            self.blocks.append(block.split('\n', 2)[2])
            self.rule_parts.append(text[len(renderer.HEADER):].rstrip('\n')
                                   if audit['selected_source_indices'] else '')

        byrecord = defaultdict(list)
        for p, row in enumerate(rows):
            byrecord[row['record_id']].append(p)
        self.neighbors, self.previous, self.record_predecessor = {}, {}, {}
        for sequence in byrecord.values():
            sequence.sort(key=lambda p: (rows[p].get('step_index', 0), rows[p]['id']))
            for n, p in enumerate(sequence):
                self.record_predecessor[p] = sequence[max(0, n - 1):n]
                self.neighbors[p] = self.record_predecessor[p] + sequence[n + 1:n + 2]
                links = []
                for need in source.state_facts(rows[p].get('current_state_t') or {}):
                    prior = next((q for q in reversed(sequence[:n])
                                  if any(self.prerequisite(need, fact) for fact in produced[q])), None)
                    if prior is not None:
                        links.append({'producer': prior, 'required': need,
                                      'witnesses': [f for f in produced[prior] if self.prerequisite(need, f)]})
                self.previous[p] = links

        self.goal_lookup = defaultdict(set)
        for p, (action, facts) in enumerate(zip(actions, produced)):
            for fact in facts:
                for name in source.source_node_names(fact[0], action):
                    self.goal_lookup[name, fact[1]].add(p)
        self.rank_cache, self.goal_cache = {}, {}
        # Preserve the existing evidence schema; owner metadata never filters retrieval.
        generic = {'unknown', 'object', 'device', 'appliance', 'physical_object', 'ui_component',
                   'button', 'switch', 'screen', 'display', 'panel', 'light', 'lighting', 'water',
                   'temperature', 'time', 'power', 'mode', 'control', 'function', 'state', 'value', 'system'}
        owners = []
        for action in actions:
            owner = source.norm(action['owner_type'])
            owned = {owner} if owner and owner not in generic else set()
            if not owned:
                owned = {source.norm(x['node']) for x in action['participants']
                         if x['role'] in {'intent_subject', 'control', 'target'}
                         and x['node_type'] in {'appliance', 'device'}
                         and not re.fullmatch(r'e\d+', x['node'])
                         and source.norm(x['node']) not in generic}
            owners.append(owned)
        record_owners = {record: set().union(*(owners[p] for p in sequence))
                         for record, sequence in byrecord.items()}
        self.owner_evidence = []
        for row, owned in zip(rows, owners):
            same_record = record_owners[row['record_id']]
            self.owner_evidence.append({
                'types': sorted(owned or (same_record if len(same_record) == 1 else set())),
                'basis': ('source_action_owner_or_declared_subject' if owned else
                          'unique_owner_in_same_record' if len(same_record) == 1 else 'unknown_or_ambiguous'),
            })

    @staticmethod
    def prerequisite(need, fact):
        return (source.norm(need[0]) == source.norm(fact[0]) and need[1] == fact[1]
                and source.compatible_value(need[2], fact[2]))

    def goal_seeds(self, goal):
        key = json.dumps(goal, sort_keys=True, ensure_ascii=False)
        if key not in self.goal_cache:
            found = {}
            for fact in (goal or {}).get('goal_facts', []):
                names = source.symbolic_names(fact.get('object'))
                attribute = source.canon_attr(fact.get('key'))
                candidates = set()
                for name in names:
                    candidates.update(self.goal_lookup.get((name, attribute), set()))
                for p in sorted(candidates):
                    witnesses = [f for f in self.produced[p]
                                 if names and not names.isdisjoint(source.source_node_names(f[0], self.actions[p]))
                                 and attribute == f[1] and source.compatible_value(fact.get('value'), f[2])]
                    if witnesses:
                        found.setdefault(p, []).append({'goal': fact, 'witnesses': witnesses})
            self.goal_cache[key] = found
        return self.goal_cache[key]

    def signature(self, p):
        action = self.actions[p]
        return (action['operator'], tuple(action['concepts']),
                tuple(sorted((source.canon_attr(x.get('attribute')), source.canon_value(x.get('value')))
                             for x in action['parameters'])),
                tuple(sorted((attribute, source.canon_value(value))
                             for _, attribute, value, _ in self.produced[p])))

    def block(self, p, number, reasons):
        return f"ACTION_NODE action_{number}\n  retrieved_by={','.join(sorted(reasons[p]))}\n" + self.blocks[p]

    def retrieve(self, task, goal=None, top_k=4, context_chars=14000, *, detailed=False):
        if task not in self.rank_cache:
            self.rank_cache[task] = self.index.rank(task)
        fuzzy, seen, skipped = [], set(), []
        for score, p in self.rank_cache[task]:
            if score <= 0:
                break
            signature = self.signature(p)
            if signature in seen:
                skipped.append({'id': self.rows[p]['id'], 'reason': 'duplicate_signature'})
                continue
            seen.add(signature)
            fuzzy.append(p)
            if len(fuzzy) == top_k:
                break
        selected, reasons = set(fuzzy), defaultdict(set)
        for p in fuzzy:
            reasons[p].add('task_fuzzy_node')
            for q in self.neighbors[p]:
                selected.add(q)
                relation = ('fuzzy_sequence_predecessor' if q in self.record_predecessor[p]
                            else 'fuzzy_sequence_successor') if detailed else 'sequence_neighbor'
                reasons[q].add(relation)
        frontier, expanded, dependencies = list(fuzzy), set(fuzzy), []
        while frontier:
            p = frontier.pop()
            for link in self.previous[p]:
                q = link['producer']
                selected.add(q)
                reasons[q].add('fuzzy_causal_predecessor' if detailed else 'fuzzy_predecessor')
                dependencies.append({'consumer': p, **link})
                if q not in expanded:
                    expanded.add(q)
                    frontier.append(q)
        goals = self.goal_seeds(goal)
        selected.update(goals)
        for p in goals:
            reasons[p].add('goal_backward')
        # Keep the historical policy: only newly selected goal predecessors expand.
        frontier, expanded = list(goals), set()
        while frontier:
            p = frontier.pop()
            if p in expanded:
                continue
            expanded.add(p)
            for link in self.previous[p]:
                q = link['producer']
                newly_selected = q not in selected
                selected.add(q)
                reasons[q].add('goal_dependency')
                dependencies.append({'consumer': p, **link})
                if q not in expanded and newly_selected:
                    frontier.append(q)
        ordered = fuzzy + sorted(selected - set(fuzzy), key=lambda p: (
            self.rows[p]['record_id'], self.rows[p].get('step_index', 0), self.rows[p]['id']))
        included, omitted, hidden = [], [], []
        if detailed:
            blocks = [DETAILED_HEADER]
            length = len(DETAILED_HEADER) + 96
            for p in ordered:
                block = self.block(p, len(included) + 1, reasons)
                if length + len(block) + 2 > context_chars:
                    omitted.append(p)
                else:
                    included.append(p)
                    blocks.append(block)
                    length += len(block) + 2
            if omitted:
                blocks.append(f'[CWM_NODES_OMITTED_AT_FIXED_LIMIT={len(omitted)}]')
            text = '\n\n'.join(blocks)
            inventory, metadata = parser._parse_original(text, allow_empty=True)
            render_audit = {'schema_version': 'switch-detailed-render-audit-v1',
                            'policy': 'historical_detailed_cwm', 'inventory': inventory,
                            'retrieval_metadata': metadata, 'source_context': text}
        else:
            length = len(renderer.HEADER) + 1
            for p in ordered:
                part = self.rule_parts[p]
                if not part:
                    hidden.append(p)
                elif length + len(part) > context_chars:
                    omitted.append(p)
                else:
                    included.append(p)
                    length += len(part)
            blocks = [self.block(p, n, reasons) for n, p in enumerate(included, 1)]
            original = CWM_HEADER + ('\n\n' + '\n\n'.join(blocks) if blocks else '')
            text, render_audit = renderer.render(task, original)
        if len(text) > context_chars:
            raise ValueError('context cap smaller than rendered context')

        ids = lambda positions: [self.rows[p]['id'] for p in positions]
        # Legacy audit flags stay for existing consumers, not as runtime switches.
        audit = {
            'mode': 'cwm_budget_control', 'task': task, 'goal_supplied': goal is not None,
            'transitions': [], 'fuzzy_ids': ids(fuzzy), 'selected_ids': ids(ordered),
            'included_ids': ids(included), 'omitted_ids': ids(omitted),
            'hidden_no_strict_facts_ids': ids(hidden), 'goal_seed_ids': ids(sorted(goals)),
            'goal_witnesses': {self.rows[p]['id']: v for p, v in goals.items()},
            'reasons': {self.rows[p]['id']: sorted(reasons[p]) for p in ordered},
            'dependency_links': [{'consumer': self.rows[x['consumer']]['id'],
                                  'producer': self.rows[x['producer']]['id'],
                                  'required': x['required'], 'witnesses': x['witnesses']} for x in dependencies],
            'fuzzy_skips': skipped, 'context_chars': len(text), 'rule_count': len(included),
            'index_documents_include_preconditions': False, 'state_values_fixed': False,
            'eligibility_before_top_k': False, 'safe_semantic_dedup': False,
            'context_budget_policy': ('historical_detailed_block_budget_without_local_reserve' if detailed
                                     else 'strict_prompt_14000_no_local_graph'),
            'renderer_audit': render_audit, 'object_guard_applied': False, 'task_owner_types': [],
            'task_owner_alias_matches': [], 'owner_guard_excluded_ids': [],
            'included_owner_evidence': {self.rows[p]['id']: self.owner_evidence[p] for p in included},
        }
        # The public API deep-copies the final graph, including cached evidence.
        return {'context': text, 'audit': audit, 'subgraph': {
            'occurrence_ids': ids(included),
            'occurrences': [{'row': self.rows[p], 'action': self.actions[p], 'produced': self.produced[p]}
                            for p in included],
        }}
