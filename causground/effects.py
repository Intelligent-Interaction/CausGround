"""Effect evidence, context bindings, observed branches and stable outcomes.

The optional VirtualHome relevance policy is an explicit environment prior
ported from vh_oracle.semantic_graph.split_effects, not an induced action table.
Neither this policy nor this module supplies any required-condition answers.
"""
from collections import Counter
from copy import deepcopy
import math

from .predicates import canonical_edge, evaluate, export_predicate, instantiate, observed_effect_confirmed, token


def _arguments(action):
    by_role = {a['role']: a['node'] for a in action['arguments']}
    if action['name'] in {'POUR', 'PUTBACK', 'PUTIN'}:
        return [by_role['source'], by_role['destination']]
    return [a['node'] for a in action['arguments']]


def classify_effects(record, policy):
    if policy not in {'observed', 'virtualhome'}:
        raise ValueError('effect_policy must be observed or virtualhome')
    action, actor = record['action']['name'], record['action']['actor']
    args = _arguments(record['action']) if policy == 'virtualhome' else []
    first, second = (args + [None, None])[:2]
    def core(effect):
        if policy == 'observed' or 'edge' not in effect:
            return True
        edge, op = effect['edge'], effect['operation']
        source, relation, target = edge['source'], edge['relation'], edge['target']
        if source == target: return False
        held = relation in {'HOLDS_LH', 'HOLDS_RH'} and source == actor and target == first
        release = held and op == 'remove_edge'
        if action == 'DROP': return release
        if action == 'GRAB':
            return (held and op == 'add_edge') or (op == 'remove_edge' and relation in {'ON', 'INSIDE'} and source == first)
        if action in {'POUR', 'PUTBACK', 'PUTIN'}:
            destination_relation = 'ON' if action == 'PUTBACK' else 'INSIDE'
            return release or (op == 'add_edge' and relation == destination_relation and source == first and target == second)
        if action == 'PUTOBJBACK':
            return release or (op == 'add_edge' and relation in {'ON', 'INSIDE'} and source == first)
        if action == 'PUTOFF': return op == 'remove_edge' and relation == 'ON' and source == first and target == actor
        if action == 'PUTON': return release or (op == 'add_edge' and relation == 'ON' and source == first and target == actor)
        if action in {'SIT', 'LIE'}: return op == 'add_edge' and relation == 'ON' and source == actor and target == first
        if action in {'WALK', 'RUN', 'FIND'}:
            return op == 'add_edge' and ((relation == 'CLOSE' and {source, target} == {actor, first}) or
                                       (relation == 'INSIDE' and source == actor and target == first))
        return action == 'TURNTO' and op == 'add_edge' and relation == 'FACING' and source == actor and target == first
    groups = {'core_effects': [], 'contextual_effects': [],
              'classification_basis': 'virtualhome_original_semantic_policy' if policy == 'virtualhome' else 'unclassified_observed_changes'}
    for effect in record['changes']:
        groups['core_effects' if core(effect) else 'contextual_effects'].append(deepcopy(effect))
    return groups


def references(effect):
    if 'edge' in effect: return {effect['edge']['source'], effect['edge']['target']}
    node = effect.get('node')
    return {node['id'] if isinstance(node, dict) else node} if node is not None else set()


def remap(value, binding):
    if isinstance(value, list): return [remap(v, binding) for v in value]
    if not isinstance(value, dict): return value
    result = {k: binding.get(v, v) if k in {'id', 'node', 'source', 'target'} and isinstance(v, str)
              else remap(v, binding) for k, v in value.items()}
    if all(k in result for k in ('source', 'relation', 'target')): result = canonical_edge(result)
    return result


def lift_effects(record, bindings):
    """Retain non-argument endpoints as variables with observed pre-state anchors.

    Anchors describe possible bindings, not guaranteed unique selectors. An
    effect-only endpoint stays unresolved rather than being leaked into pre-state.
    """
    effects = record['head']['core_effects']
    nodes = {n['id']: n for n in record['after']['nodes'] + record['before']['nodes']}
    external = set().union(*(references(e) for e in effects)) - set(bindings) if effects else set()
    anchors = {}
    for node_id in external:
        anchors[node_id] = [remap(e, bindings) for e in record['before']['edges']
                            if (e['source'] == node_id and e['target'] in bindings) or
                               (e['target'] == node_id and e['source'] in bindings)]
    def order(node_id):
        anonymous = {node_id: '$context'}
        return (nodes.get(node_id, {}).get('type', ''), token(sorted((remap(e, anonymous) for e in anchors[node_id]), key=token)), node_id)
    mapping = dict(bindings)
    for i, node_id in enumerate(sorted(external, key=order)):
        mapping[node_id] = '$context' + str(i)
    contexts = [{'variable': mapping[n], 'type': nodes.get(n, {}).get('type'),
                 'before_relations': sorted({token(remap(e, mapping)): remap(e, mapping) for e in anchors[n]}.values(), key=token),
                 'binding_status': 'pre_state_anchor_not_guaranteed_unique' if anchors[n] else 'unresolved_from_pre_state'}
                for n in sorted(external, key=lambda n: mapping[n])]
    lifted = sorted({token(remap(e, mapping)): remap(e, mapping) for e in effects}.values(), key=token)
    unconfirmed = {token(remap(e, mapping)) for e in effects if not observed_effect_confirmed(e, record['before'], record['after'])}
    return {'effects': lifted, 'unconfirmed_effects': [i for i, e in enumerate(lifted) if token(e) in unconfirmed],
            'context_bindings': contexts}, mapping


def post_candidates(record, bindings):
    candidates = {}
    for effect in record['head']['core_effects']:
        if not references(effect) <= set(bindings): continue
        effect = remap(effect, bindings)
        op = effect['operation']
        if op in {'add_edge', 'remove_edge'}:
            predicate = {'type': 'edge', **effect['edge']}
            if op == 'remove_edge': predicate = {'type': 'not', 'predicate': predicate}
            candidates[token(predicate)] = predicate
            edge = effect['edge']
            if edge['relation'] in {'HOLDS_LH', 'HOLDS_RH'}:
                held = {'type': 'any_of', 'meaning': 'held_by_either_hand', 'predicates': [
                    {'type': 'edge', **{**edge, 'relation': hand}} for hand in ('HOLDS_LH', 'HOLDS_RH')]}
                predicate = held if op == 'add_edge' else {'type': 'not', 'predicate': held}
                candidates[token(predicate)] = predicate
        elif op in {'update_attribute', 'set_attribute'} and effect.get('after') is not None:
            p = {'type': 'node_attribute', 'node': effect['node'], 'attribute': effect['attribute'], 'operator': 'equals', 'value': effect['after']}
            candidates[token(p)] = p
        elif op == 'remove_node':
            p = {'type': 'not', 'predicate': {'type': 'node_exists', 'node': effect['node']}}
            candidates[token(p)] = p
    return candidates


def outcome_summary(records, min_support):
    candidates = {}
    for r in records: candidates.update(r['_post_candidates'])
    stable, pending, rejected = [], [], []
    for _, predicate in sorted(candidates.items()):
        counts = Counter()
        for r in records:
            value = evaluate(instantiate(predicate, r['_bindings']), r['after'])
            counts['unknown' if value is None else 'true' if value else 'false'] += 1
        item = {'predicate': deepcopy(predicate), 'support': counts['true'], 'false': counts['false'], 'unknown': counts['unknown'],
                'successful_executions': len(records), 'basis': 'observed_success_postcondition_consistency', 'necessity_verified': False}
        if counts['false']: rejected.append(item)
        elif counts['true'] == len(records) and len(records) >= min_support: stable.append(item)
        else: pending.append(item)
    return {'postconditions': stable, 'pending_postconditions': pending, 'rejected_postconditions': rejected}


def effect_branches(records, labels, predicates, truth, min_support):
    """Small empirical decision tree. Unknown tests never mean false.

    Leaves retain all alternative outcomes when pre-state features cannot separate
    them. This is an in-sample description, not a proof or held-out predictor.
    """
    if not records: return []
    affected = set()
    for record in records:
        for effect in record['_local_effects']:
            if 'edge' in effect:
                e = effect['edge']; affected.add(('edge', e['source'], e['relation'], e['target']))
            elif 'attribute' in effect:
                affected.add(('attribute', effect['node'], effect['attribute']))
    def relevance(predicate):
        p = export_predicate(predicate)
        if p.get('type') == 'edge': return int(('edge', p['source'], p['relation'], p['target']) in affected)
        if p.get('type') == 'node_attribute': return int(('attribute', p['node'], p['attribute']) in affected)
        return 0
    features = []
    for _, p in sorted(predicates.items()):
        values = [truth(p, r) for r in records]
        if 'unknown' not in values and len(set(values)) > 1: features.append((p, values))
    def entropy(indices):
        counts = Counter(labels[i] for i in indices)
        return -sum((n/len(indices))*math.log2(n/len(indices)) for n in counts.values())
    branches = []
    def visit(indices, path, available, depth):
        counts = Counter(labels[i] for i in indices)
        split = None
        if len(counts) > 1 and depth < 8:
            base = entropy(indices)
            for f in available:
                p, values = features[f]
                yes = [i for i in indices if values[i] == 'true']
                no = [i for i in indices if values[i] == 'false']
                if min(len(yes), len(no)) < min_support: continue
                gain = base - (len(yes)*entropy(yes)+len(no)*entropy(no))/len(indices)
                # Prefer the changing relation/attribute when predictive gains tie;
                # this avoids an arbitrary shorter coincident feature winning.
                candidate = (round(gain, 12), relevance(p), -len(token(p)), -f)
                if gain > 1e-12 and (split is None or candidate > split[0]): split = candidate, f, yes, no
        if split:
            _, f, yes, no = split
            predicate = export_predicate(features[f][0])
            rest = [x for x in available if x != f]
            visit(yes, path+[predicate], rest, depth+1)
            visit(no, path+[{'type': 'not', 'predicate': predicate}], rest, depth+1)
        else:
            branches.append({'when': {'type': 'all_of', 'predicates': path},
                'alternatives': [{'variant_id': label, 'support': n} for label, n in sorted(counts.items())],
                'evidence_ids': sorted(records[i]['id'] for i in indices), 'support': len(indices),
                'deterministic_in_observed_records': len(counts) == 1,
                'sufficient_support': len(indices) >= min_support,
                'basis': 'observed_pre_state_partition', 'generalization_verified': False})
    visit(list(range(len(records))), [], list(range(len(features))), 0)
    return branches


def summarize(records, predicates, truth, min_support, rule_id):
    successful = [r for r in records if r['outcome'] == 'success']
    variants, labels = {}, []
    identifiers = {key: f"{rule_id}/effect:{i}" for i, key in enumerate(sorted({token(r['_effect_variant']) for r in successful}), 1)}
    for r in successful:
        v = r['_effect_variant']; key = token(v)
        item = variants.setdefault(key, {**deepcopy(v), 'id': identifiers[key], 'evidence_ids': []})
        item['evidence_ids'].append(r['id']); labels.append(item['id'])
    items = [{**v, 'evidence_ids': sorted(v['evidence_ids']), 'support': len(v['evidence_ids'])} for _, v in sorted(variants.items())]
    return {'effect_variants': items,
            'effect_branches': effect_branches(successful, labels, predicates, truth, min_support),
            **outcome_summary(successful, min_support)}
