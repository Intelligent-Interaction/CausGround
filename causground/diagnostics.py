"""Conservative consistency diagnostics on the shared result; no retrieval."""
from .contracts import invalidates


def analyze(graph, vocabulary):
    nodes={m['id']:m for m in graph['mechanisms'] if not m['candidate_only']}
    edges={(d['producer'],d['consumer']) for d in graph['dependencies'] if d['producer'] in nodes and d['consumer'] in nodes}
    edges.update((d['source'],d['target']) for d in graph.get('phase_order_constraints',[]) if d['source'] in nodes and d['target'] in nodes)
    adjacency={k:set() for k in nodes}
    for a,b in edges:adjacency[a].add(b)
    def reachable(a,b):
        seen=set(); todo=list(adjacency[a])
        while todo:
            x=todo.pop()
            if x==b:return True
            if x not in seen:seen.add(x);todo.extend(adjacency[x])
        return False
    remaining=set(nodes);order=[]
    while remaining:
        ready=sorted(n for n in remaining if not any(b==n and a in remaining for a,b in edges))
        if not ready:break
        order.extend(ready);remaining.difference_update(ready)
    threats=[]
    for d in graph['dependencies']:
        a,b=d['producer'],d['consumer']
        if a not in nodes or b not in nodes:continue
        for c,node in nodes.items():
            if c in {a,b} or reachable(c,a) or reachable(b,c):continue
            if invalidates(node,d['predicate']):
                threats.append({'producer':a,'consumer':b,'threat':c,'predicate':d['predicate'],'status':'possible_unordered_threat','alternative_resolution_required':True})
    # Count only simultaneous final holding goals, never all candidate PICKs.
    resources=[]; holding=set(vocabulary.get('holding_relations',['HOLDS']))
    final_phase=max((m.get('phase',0) for m in graph['goal_spec'].get('goal_metadata',[])),default=0)
    held=[]
    for i,p in enumerate(graph['goal_spec']['goals']):
        meta=graph['goal_spec'].get('goal_metadata',[{}]*len(graph['goal_spec']['goals']))[i]
        if (meta.get('scope')=='final' or (not meta.get('scope') and meta.get('phase',0)==final_phase)) and p.get('type')=='edge' and p.get('relation') in holding:held.append(p['target'])
    capacity=vocabulary.get('holding_capacity',1 if graph.get('domain')=='RoboCasa' else 2)
    if len(set(held))>capacity:resources.append({'resource':'holding_capacity','capacity':capacity,'objects':sorted(set(held)),'status':'conflicting_final_goals'})
    return {'order':order,'cycle_nodes':sorted(remaining),'has_cycle':bool(remaining),'state_threats':threats,'resource_conflicts':resources,
            'conflict_analysis':'conservative_declared_effects','order_is_execution_plan':False}
