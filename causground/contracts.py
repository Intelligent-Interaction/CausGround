"""Shared predicate contradiction and phase invalidation helpers; no graph search."""
from .knowledge import normalize_predicate, predicate_shape, effect_predicate


def opposite(a, b):
    a,b=predicate_shape(a),predicate_shape(b)
    if a.get('type')=='not': return a['predicate']==b
    if b.get('type')=='not': return b['predicate']==a
    if a.get('type')==b.get('type')=='node_attribute':
        return (a['node'],a['attribute'])==(b['node'],b['attribute']) and a.get('operator')==b.get('operator')=='equals' and a['value']!=b['value']
    return False


def invalidates(node, predicate):
    if any(opposite(p['predicate'],predicate) for p in node.get('postconditions',[])):return True
    for variant in node.get('effect_variants',[]):
        for effect in variant.get('effects',[]):
            p=effect_predicate(effect)
            if p and opposite(p,predicate):return True
            if effect.get('operation')=='delete_edges_matching' and predicate.get('type')=='edge':
                pattern=effect.get('pattern',{})
                if all(pattern.get(k,predicate[k])==predicate[k] for k in ('source','relation','target')):return True
    # Preserve the legacy location-invalidating contract as a labeled prior.
    if node.get('source_format')=='legacy_robocasa' and predicate.get('type')=='edge' and predicate.get('relation') in {'ON','INSIDE'}:
        args={a['role']:a.get('node') for a in node['action']['arguments']}
        if args.get('object')==predicate.get('source') and node['action']['name'] in {'PICK','PLACE'}:
            return not any(predicate_shape(p['predicate'])==predicate_shape(predicate) for p in node.get('postconditions',[]))
    return False


def validate_initial_facts(facts):
    if facts is None:return []
    if not isinstance(facts,list):raise ValueError('initial_state must be a predicate list')
    output=[]
    for fact in facts:
        if not isinstance(fact,dict):raise ValueError('Initial facts must be predicates')
        p=normalize_predicate(fact)
        base=p['predicate'] if p.get('type')=='not' else p
        if base.get('type') not in {'edge','node_attribute'}:raise ValueError('Unsupported initial fact')
        fields=('source','relation','target') if base['type']=='edge' else ('node','attribute')
        if any(not isinstance(base.get(k),str) or not base[k] or base[k].startswith('$') for k in fields):raise ValueError('Initial fact requires explicit entities')
        if base['type']=='node_attribute' and (base.get('operator')!='equals' or 'value' not in base or base['value'] is None):raise ValueError('Initial attribute fact must specify a known value')
        if any(opposite(p,q) for q in output):raise ValueError('Contradictory initial-state predicates')
        output.append(p)
    return output


def canonicalize_facts(facts, aliases):
    def entity(value):
        seen=set()
        while value in aliases and aliases[value]!=value:
            if value in seen:raise ValueError('Cyclic object aliases')
            seen.add(value);value=aliases[value]
        return value
    def walk(value):
        if isinstance(value,list):return [walk(x) for x in value]
        if isinstance(value,dict):return {k:entity(v) if k in {'node','source','target'} and isinstance(v,str) else walk(v) for k,v in value.items()}
        return value
    return validate_initial_facts(walk(facts))
