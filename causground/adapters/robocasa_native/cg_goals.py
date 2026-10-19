"""Text goals and requested temporal relations, independent of episode labels."""
from collections import defaultdict
import re
from .cg_text import mentioned, matches, placement_requests, mask_negations, pickup_subjects, expand_collective_requests
from .cg_actions import actions_in
from .predicates import effect_predicate, predicate_key


VERBS=r'place|put|turn|close|pick|start|take|press|move|retrieve|wash|stir|open|scrub|grab|return|slide|wait'


def normalize(text):
    text=text.lower().replace('-',' ')
    for a,b in [('turning','turn'),('placing','place'),('picking','pick'),('closing','close'),('opening','open'),('starting','start'),('scrubbing','scrub'),('stirring','stir'),('pressing','press')]:
        text=re.sub(r'\b'+a+r'\b',b,text)
    return text


def text_segments(description):
    text=normalize(description);parts=[];relations=[];start=0;last_marker=None
    separators=re.compile(r'[.;]|\bthen\b|\bbefore\b|\bafter\b|(?:,\s*(?:and\s+)?|\band\s+)(?=(?:'+VERBS+r')\b)')
    leading='before' if text.lstrip().startswith('before ') else 'after' if text.lstrip().startswith('after ') else None
    for m in separators.finditer(text):
        chunk=text[start:m.start()].strip(' ,.');marker=m.group().strip(' ,.')
        if chunk:
            id=len(parts);parts.append({'id':id,'text':chunk,'offset':start})
            if id and last_marker:
                if last_marker=='after':relations.append({'before':id,'after':id-1,'basis':'explicit_after'})
                elif last_marker=='before':relations.append({'before':id-1,'after':id,'basis':'explicit_before'})
                else:relations.append({'before':id-1,'after':id,'basis':'explicit_then' if last_marker=='then' else 'text_command_sequence'})
        last_marker=marker if marker else 'text_command_sequence';start=m.end()
    chunk=text[start:].strip(' ,.')
    if chunk:
        id=len(parts);parts.append({'id':id,'text':chunk,'offset':start})
        if id:
            if last_marker=='after':relations.append({'before':id,'after':id-1,'basis':'explicit_after'})
            else:relations.append({'before':id-1,'after':id,'basis':'explicit_before' if last_marker=='before' else 'explicit_then' if last_marker=='then' else 'text_command_sequence'})
    if leading and len(parts)>1:
        relations=[r for r in relations if {r['before'],r['after']}!={0,1}]
        relations.append({'before':1 if leading=='before' else 0,'after':0 if leading=='before' else 1,'basis':'explicit_'+leading})
    # Remap phase numbers to requested execution order, rather than text offsets.
    incoming={p['id']:0 for p in parts};successors=defaultdict(set)
    for r in relations:
        if r['after'] not in successors[r['before']]:successors[r['before']].add(r['after']);incoming[r['after']]+=1
    ready=sorted(i for i,n in incoming.items() if n==0);order=[]
    while ready:
        n=ready.pop(0);order.append(n)
        for t in successors[n]:
            incoming[t]-=1
            if incoming[t]==0:ready.append(t);ready.sort()
    rank={id:i for i,id in enumerate(order)}
    for p in parts:p['phase']=rank.get(p['id'],p['id'])
    for r in relations:r['before']=rank.get(r['before'],r['before']);r['after']=rank.get(r['after'],r['after'])
    return parts,relations


def extract_goals(description,mechanisms,scores=None):
    scores=scores or {};text,prohibitions=mask_negations(normalize(description));segments,temporal=text_segments(text)
    unmodeled=[]
    objects=set();targets=set()
    for m in mechanisms:
        for role in ['$object','$container','$tool']:
            v=m['bindings'].get(role)
            if v and not v.startswith('$'):objects.add(v)
        v=m['bindings'].get('$target')
        if v and not v.startswith('$'):targets.add(v)
    targets|=objects
    spans=mentioned(text,sorted(objects));same_span=defaultdict(list)
    for a,b,w in spans:same_span[(a,b)].append(w)
    canonical={}
    for words in same_span.values():
        preferred=min(words,key=lambda w:(w.count('_'),len(w)))
        for w in words:canonical[w]=preferred
    requests=expand_collective_requests(placement_requests(text,objects,targets,with_offsets=True),text,objects)
    requests=[(canonical.get(o,o),t,r,offset) for o,t,r,offset in requests]
    goals=[];seen=set();notes=[]
    def add(p,segment,basis,confidence='explicit_text',mechanism_ref=None,constraints=None,requested_action=None):
        key=(predicate_key(p),segment['phase'])
        if key in seen:return
        seen.add(key)
        goals.append({'id':'goal_'+str(len(goals)+1),
                      'predicate':p,'phase':segment['phase'],'source_clause':segment['text'],
                      'source_offset':segment['offset'],'basis':basis,'confidence':confidence,
                      'requested_action':requested_action,
                      'candidate_mechanism_ref':mechanism_ref,'binding_constraints':constraints or {}})
    for obj,target,relation,offset in requests:
        segment=max((s for s in segments if s['offset']<=offset),key=lambda s:s['offset'],default=segments[0])
        qualifiers={}
        qualified=re.fullmatch(r'(plate|bowl|cup|mug|glass)_(on|in)_the_(.+)',target)
        if qualified:
            qualifiers={'$target':{'binding_policy':'resolve_base_object_with_requested_location','object_class':qualified[1],
                                  'scene_relation':qualified[2],'scene_target':qualified[3],'verified':False}}
            target=qualified[1]
        loop=[m for m in mechanisms if m['action']=='PLACE' and m['bindings'].get('$object')==obj and
              m['bindings'].get('$target')=='$'+target+'_each' and m['bindings'].get('$relation')==relation]
        bound_target=loop[0]['bindings']['$target'] if loop else target
        add({'type':'edge','source':obj,'relation':{'IN':'INSIDE','ON':'ON','NEAR':'NEAR'}[relation],
             'target':bound_target,'present':True},segment,'parsed_placement_request',constraints=qualifiers or (loop[0].get('role_type_constraints') if loop else None),requested_action='PLACE')
    if not requests and re.search(r'\b(?:appropriate|suitable|right)\b.*\btool\b',text):
        destination=next((w for a,b,w in reversed(mentioned(text,targets)) if w not in objects or any(m['bindings'].get('$target')==w for m in mechanisms)),None)
        options=[m for m in mechanisms if m['action']=='PLACE' and m['bindings'].get('$target')==destination and
                 any(set(t.get('groups',[]))&{'tool','utensil'} for t in m.get('role_types',{}).get('$object',[]))]
        def tool_score(mechanism):
            obj=mechanism.get('bindings',{}).get('$object','').lower()
            score=scores.get(mechanism['id'],0)
            # Keep the selection grounded in the requested function when no
            # embedding scores are available. The alternatives remain visible
            # in binding_constraints instead of being treated as verified.
            if re.search(r'\b(?:peel|skin)\b',text) and obj=='peeler': score+=2.0
            if re.search(r'\b(?:slice|chop|dice)\b',text) and obj=='knife': score+=1.0
            if re.search(r'\b(?:cut|cutting)\b',text) and obj=='knife': score+=0.5
            return score
        options.sort(key=tool_score,reverse=True)
        if options:
            m=options[0];obj=m['bindings']['$object'];relation=m['bindings']['$relation']
            basis = 'functional_tool_semantic_candidate' if scores else 'functional_tool_contract_candidate'
            confidence = 'semantic_function_inference' if scores else 'conditional_function_selection'
            add({'type':'edge','source':obj,'relation':{'ON':'ON','IN':'INSIDE','NEAR':'NEAR'}[relation],'target':destination,'present':True},
                segments[-1],basis,confidence,m['id'],
                {'$object':{'binding_policy':'choose_tool_for_requested_function','verified':False,
                            'alternatives':list(dict.fromkeys(x['bindings']['$object'] for x in options[:5]))}},'PLACE')
    for m in mechanisms:
        if m['action']=='PLACE' and m['bindings'].get('$object')=='$objects_each':
            add({'type':'edge','source':'$objects_each','relation':'INSIDE','target':m['bindings']['$target'],'present':True},
                segments[0],'quantified_scene_inventory_goal','conditional_binding',m['id'],m.get('role_type_constraints'),'PLACE')
    if re.search(r'\bsort\b.*\b(?:opposite sides|separate sides)\b',text):
        unmodeled.append({'clause':text,'reason':'relative-side partition has no state predicate/contract in the shared CausalGround mechanism library'})
    # Typed counterpart requests cannot be resolved to a unique scene instance.
    for m in mechanisms:
        if m['action']=='PLACE' and m.get('query_goal_request') and '$counterpart' in m['bindings'].get('$target',''):
            segment=next((s for s in segments if 'counterpart' in s['text']),segments[-1])
            for fx in m['effects']:
                if fx.get('operation')=='add_edge' and fx['edge']['source']!='agent':
                    add(effect_predicate(fx),segment,'typed_counterpart_intent','conditional_binding',m['id'],m.get('role_type_constraints'))
    for s in segments:
        allowed=actions_in(s['text']);controls=allowed-{'PICK','PLACE','NAVIGATE'}
        if not re.search(r'\bwait\b|pops up|toaster cycle',s['text']):controls.discard('WAIT')
        if not re.search(r'\b(turn|switch|rotate|set|wash|rinse)\b',s['text']):controls.discard('ROTATE')
        if 'pops up' in s['text']:controls.add('WAIT')
        if 'press down' in s['text']:controls.discard('PRESS')
        if not re.search(r'(?:^|\b(?:and|then)\s+)open\b',s['text']):controls.discard('OPEN')
        activation=bool(re.search(r'\b(?:turn|start|press|switch|set)\b',s['text']) and
                        ('start' in s['text'] or 'press' in s['text'] or 'on' in s['text'] or 'off' in s['text'] or 'timer' in s['text']))
        if activation:controls.discard('PRESS');controls.discard('ROTATE');controls.add('ACTIVATE')
        if activation and re.search(r'\b(?:wash|rinse)\b',s['text']) and 'spout' in s['text']:controls.add('ROTATE_PROCESS')
        for action in sorted(controls):
            options=[m for m in mechanisms if m['action']==action or
                     (action=='ACTIVATE' and any(fx.get('attribute')=='is_on' for fx in m.get('effects',[])) and m['action'] in {'PRESS','ROTATE'}) or
                     (action=='ROTATE_PROCESS' and m['action']=='ROTATE' and any(fx.get('attribute')=='washed_locations' for fx in m.get('effects',[])))]
            if not options:continue
            burner_options=options
            if action in {'STIR','SCRUB'}:
                body=re.split(r'\b(?:with|using)\b',re.split(r'\b(?:stir|scrub)\b',s['text'],1)[-1],1)[0]
                location=re.search(r'\b(?:in|inside)\s+(.+)',body)
                if action=='STIR' and location:body=location[1]
                explicit_targets=mentioned(body,objects|targets)
                if explicit_targets:
                    target=explicit_targets[0][2]
                    matching=[m for m in options if target in [m['bindings'].get('$target'),m['bindings'].get('$container')]]
                    if not matching:
                        unmodeled.append({'clause':s['text'],'requested_target':target,'reason':'no process contract matches the explicit target'});continue
                    options=matching
            vocab=[v for m in options for v in m['bindings'].values() if v and not v.startswith('$') and v not in {'on','off','in','out'}]
            local_words={v for _,_,v in mentioned(s['text'],vocab)}
            previous=mentioned(text[:s['offset']],vocab)
            recent=max((a for a,b,v in previous),default=-1)
            context_words={v for a,b,v in previous if a==recent}
            local=[m for m in options if any(v in local_words for v in m['bindings'].values())]
            context=[m for m in options if any(v in context_words for v in m['bindings'].values())]
            options=local or context or options
            unique_wait=action=='WAIT' and len({m['bindings'].get('$fixture') for m in options})==1
            if not local and not context and not unique_wait and action in {'OPEN','CLOSE','ACTIVATE','WAIT','SLIDE'}:
                unmodeled.append({'clause':s['text'],'reason':'device binding has no explicit or preceding antecedent'});continue
            if action in {'ROTATE','ACTIVATE'} and 'burner' in s['text']:
                options=[m for m in burner_options if m['bindings'].get('$fixture','').endswith('_burner') or m['bindings'].get('$fixture') in {'burner','$burner'}]
                hint=next((t for o,t,r,offset in reversed(requests) if 'burner' in t and offset<s['offset']),None)
                direct=[w for _,_,w in mentioned(s['text'],[m['bindings'].get('$fixture','') for m in options])]
                desired=(direct or [hint])[0]
                matching=[m for m in options if m['bindings'].get('$fixture')==desired or
                          (desired in {'stove_burner','burner'} and m['bindings'].get('$fixture')=='$burner')]
                if matching:options=matching
                elif not direct and not hint:
                    unmodeled.append({'clause':s['text'],'reason':'burner identity cannot be grounded from preceding placement'});continue
                if not options:continue
                if desired in {'stove_burner','burner'}:notes.append({'binding_equality':{'left':'$burner','right':desired},'verified':False})
            options.sort(key=lambda m:scores.get(m['id'],0),reverse=True)
            chosen=[options[0]]
            if local and action in {'OPEN','CLOSE','ACTIVATE'}:
                chosen=[]
                for word in sorted(local_words):
                    candidates=[m for m in options if m['bindings'].get('$fixture')==word]
                    if candidates:chosen.append(candidates[0])
                chosen=chosen or [options[0]]
            for m in chosen:
                outcomes=[fx for fx in m.get('effects',[]) if fx.get('operation')=='set_attribute' and fx.get('node')!='agent']
                if action=='ACTIVATE':outcomes=[fx for fx in outcomes if fx.get('attribute')=='is_on']
                if not outcomes:continue
                p=effect_predicate(outcomes[0])
                if re.search(r'\b(?:off|stop)\b',s['text']) and p.get('attribute')=='is_on':p['value']=False
                if action=='SLIDE' and 'out' in s['text']:p['value']=True
                elif action=='SLIDE' and re.search(r'\b(?:in|back)\b',s['text']):p['value']=False
                confidence='explicit_text' if local else 'resolved_previous_device' if context else 'inferred_device'
                add(p,s,'text_control_or_process_intent',confidence,m['id'],m.get('role_type_constraints'),m['action'])
            if len(options)>1:notes.append({'phase':s['phase'],'action':action,'candidate_mechanisms':[m['id'] for m in options[:5]],
                                          'reason':'device or control binding ambiguity; ranking is not dependency evidence'})
        # Standalone pick requests produce HOLDS goals; picking needed for later
        # placement/tool use is inferred from the consumer contract instead.
        if 'PICK' in allowed:
            for obj in pickup_subjects(s['text'],objects):
                obj=canonical.get(obj,obj)
                used_by_placement=any(o==obj and offset>=s['offset'] and offset<s['offset']+len(s['text'])+1 for o,t,r,offset in requests)
                collective=len({canonical.get(o,o) for o in pickup_subjects(s['text'],objects)})>1
                if collective and any(o==obj and offset>=s['offset'] for o,t,r,offset in requests):used_by_placement=True
                used_by_process=any(m['action'] in {'STIR','SCRUB','TILT'} and
                    obj in [m['bindings'].get('$tool'),m['bindings'].get('$container')] and
                    m['action'] in actions_in(s['text']) for m in mechanisms)
                if used_by_placement or used_by_process:continue
                add({'type':'edge','source':'agent','relation':'HOLDS','target':obj,'present':True},s,'standalone_pick_request',requested_action='PICK')
    # Final goals are distinct from intermediate location/attribute milestones.
    if re.search(r'\brelease\b',text):unmodeled.append({'clause':text,'reason':'release destination/contract is unspecified; no arbitrary PLACE inserted'})
    final={}
    for g in sorted(goals,key=lambda g:g['phase']):
        p=g['predicate'];key=predicate_key(p)
        if p['type']=='edge' and p['source']!='agent' and p['relation'] in {'ON','INSIDE'}:family=('location',p['source'])
        elif p['type']=='edge' and p['source']=='agent' and p['relation']=='HOLDS':family=('location',p['target'])
        else:family=key[:-1]
        final[family]=g
    final_ids={g['id'] for g in final.values()}
    for g in goals:g['scope']='final' if g['id'] in final_ids else 'milestone'
    return {'goals':goals,'final_goal_states':list(final.values()),'temporal':temporal,'segments':segments,
            'ambiguities':notes,'prohibitions':prohibitions,'unmodeled_text_requirements':unmodeled,
            'unparsed_description':not goals,'input_is_description_only':True}
