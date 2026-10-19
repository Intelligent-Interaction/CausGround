"""Query-scoped contracts, with assumptions distinct from learned evidence."""
from copy import deepcopy
import re
from .cg_text import mentioned, placement_requests, subjects, matches, mask_negations, expand_collective_requests
from .cg_instantiate import apply_bindings


ENCLOSURES={'fridge','freezer','drawer','cabinet','microwave','toaster_oven','dishwasher'}


def enrich_query_contracts(pool,description,graph):
    pool=deepcopy(pool);notes=[];text,_=mask_negations(description.lower().replace('-',' '))
    objects=set(graph.get('public_object_type_hierarchy',{}))
    targets=set()
    for m in pool:
        if m['action']=='TILT':
            for fx in m.get('effects',[]):
                source=fx.get('edge',{}).get('source')
                if fx.get('operation')=='add_edge' and source and source!='agent':
                    fx['applicability_condition']={'type':'edge','source':source,'relation':'INSIDE',
                                                   'target':m['bindings']['$container'],'present':True}
                    fx['applicability_basis']='recorded_content_membership_required; historical members are not unconditional transfer outcomes'
        if any(fx.get('attribute')=='washed_locations' for fx in m.get('effects',[])):
            m['conditions'].append({'predicate':{'type':'node_attribute','node':'sink','attribute':'is_on','value':True},
                'basis':'process_definition_guard_water_running','necessity_verified':False})
        objects.update(v for r,v in m['bindings'].items() if r in {'$object','$tool','$container'} and v and not v.startswith('$'))
        targets.update(v for r,v in m['bindings'].items() if r in {'$target','$fixture'} and v and not v.startswith('$'))
    targets|=objects
    # A missing typed target/relation can still use the public generic contract.
    # Only text-requested bindings are created, never a Cartesian scene plan.
    calls={m['api_call'] for m in pool}
    generic={m['action']:m for m in graph.get('generic_mechanisms',[])}
    requests=expand_collective_requests(placement_requests(text,objects,targets),text,objects)
    if re.search(r'\b(?:all|every)\s+(?:the\s+)?(?:objects|items)\b',text):
        destinations=[w for _,_,w in mentioned(text,targets) if w in ENCLOSURES]
        if destinations:requests.append(('$objects_each',destinations[0],'IN'))
    for obj,target,relation in requests:
        for action,bindings in [('PICK',{'$object':obj}),('PLACE',{'$object':obj,'$target':target,'$relation':relation})]:
            values=[bindings['$'+p] for p in generic.get(action,{}).get('parameters',[])]
            call=action+'('+','.join(values)+')'
            if action not in generic or call in calls:continue
            base=generic[action]
            m={'id':'query_public_'+str(len(pool)+1),
               'action':action,'api_call':call,'bindings':bindings,
               'conditions':apply_bindings(base['conditions'],bindings),
               'effects':apply_bindings(base['effects'],bindings),'evidence_ids':[],
               'parent_generic_id':base['id'],'instantiation_basis':'public_generic_contract + explicit_query_binding',
               'query_goal_request':True,'new_type_outcome_observed':False,
               'role_type_constraints':{'$object':{'binding_policy':'resolve_requested_scene_object'},
                                        '$target':{'binding_policy':'resolve_requested_scene_destination'}}}
            if obj=='$objects_each':m['role_type_constraints']['$object']={'binding_policy':'each_object_from_unresolved_scene_inventory','quantifier':'all','inventory_verified':False}
            for fx in m['effects']:
                if fx.get('operation')=='add_edge_from_argument':
                    fx['operation']='add_edge'
                    fx['edge']['relation']=fx['relation_value_map'][fx['edge'].pop('relation_argument')]
            pool.append(m);calls.add(call)
    from .cg_goals import text_segments
    segments,_=text_segments(text)
    placements=placement_requests(text,objects,targets,with_offsets=True)
    def phase_at(offset):return max((s for s in segments if s['offset']<=offset),key=lambda s:s['offset'],default={'phase':0})['phase']
    source_enclosures={};scoped_sources=[]
    for segment in segments:
        chunk=segment['text']
        for match in re.finditer(r'\bfrom\s+(?:(?:the|a|an|open|closed)\s+)*',chunk):
            fixtures=mentioned(chunk[match.end():],ENCLOSURES)
            if not fixtures or fixtures[0][0]!=0:continue
            enclosure=fixtures[0][2]
            head=chunk[:match.start()]
            # The latest coordinated subject selects its own source enclosure.
            head=re.split(r'\band\s+(?:(?:one|a|an|the)\s+)',head)[-1]
            for obj in subjects(head,objects):
                source_enclosures[obj]=enclosure
                next_placement=next((offset for o,t,r,offset in placements if o==obj and offset>=segment['offset']),None)
                through=phase_at(next_placement) if next_placement is not None else segment['phase']
                scoped_sources.append({'object':obj,'enclosure':enclosure,'from_phase':segment['phase'],'through_phase':through})
    for m in pool:
        enclosure=None
        if m['action']=='PICK':
            for scoped in scoped_sources:
                if scoped['object']!=m['bindings'].get('$object'):continue
                condition={'predicate':{'type':'node_attribute','node':scoped['enclosure'],'attribute':'is_open','value':True},
                           'basis':'query_enclosure_access_contract_assumption','necessity_verified':False}
                m.setdefault('query_phase_conditions',[]).append({'from_phase':scoped['from_phase'],
                    'through_phase':scoped['through_phase'],'condition':condition})
        if m['action']=='PLACE' and m['bindings'].get('$relation')=='IN':
            target=m['bindings'].get('$target')
            if target in ENCLOSURES:enclosure=target
        if not enclosure:continue
        p={'type':'node_attribute','node':enclosure,'attribute':'is_open','value':True}
        if not any(c.get('predicate')==p for c in m['conditions']):
            m['conditions'].append({'predicate':p,'basis':'query_enclosure_access_contract_assumption',
                                    'necessity_verified':False})
            m.setdefault('query_context_requirements',[]).append({'predicate':p,'basis':'text source/destination enclosure',
                'assumption':'access through an open enclosure; scene-specific access is unverified'})
    if source_enclosures:notes.append({'basis':'query_enclosure_access_contract_assumption','source_occurrences':scoped_sources,'verified':False})
    scene_facts=[]
    for enclosure in ENCLOSURES:
        if re.search(r'\b(?:in|inside|into)\s+(?:(?:the|an?|one)\s+)?open\s+'+re.escape(enclosure.replace('_',' '))+r'\b',text):
            scene_facts.append({'type':'node_attribute','node':enclosure,'attribute':'is_open','value':True,'basis':'text_asserted_initial_scene'})
    if scene_facts:notes.append({'basis':'text_asserted_initial_scene','scene_facts':scene_facts})
    return pool,notes
