"""Bind shared templates using query text and the public object hierarchy only."""
from copy import deepcopy
import json
import re
from .cg_goal_requests import mentioned, placement_requests


def apply_bindings(value,bindings):
    if isinstance(value,str):return bindings.get(value,value)
    if isinstance(value,list):return [apply_bindings(v,bindings) for v in value]
    if isinstance(value,dict):return {k:apply_bindings(v,bindings) for k,v in value.items()}
    return value


def template_instances(graph,description):
    hierarchy=graph.get('public_object_type_hierarchy',{})
    templates=graph.get('role_templates',[])
    aliases={k:[{'class':k,'groups':v,'kind':'movable'}] for k,v in hierarchy.items()}
    for group in ['vegetable','fruit','meat','condiment','food']:
        aliases[group]=[{'class':group,'groups':[group],'kind':'movable'}]
    for m in graph['specific_mechanisms']:
        for role in ['$object','$container','$tool']:
            value=m['bindings'].get(role)
            if value and value.count('_')<=4:
                aliases.setdefault(value,m.get('role_types',{}).get(role) or [{'class':None,'groups':[],'kind':'unresolved_descriptor'}])
    objects=list(dict.fromkeys(word for _,_,word in mentioned(description,list(aliases))))
    class_aliases={}
    for template in templates:
        if template['action']!='PLACE':continue
        cls=template['role_type_constraints']['$target']['class']
        if 'Fridge' in cls:names=['fridge','freezer']
        elif 'Cabinet' in cls:names=['cabinet']
        elif cls in {'Stove','Stovetop'}:names=['stove','stove_burner']
        elif cls=='Counter':names=['counter','dining_counter']
        elif cls=='CoffeeMachine':names=['coffee_machine_tray','coffee_machine']
        else:names=[re.sub(r'(?<!^)(?=[A-Z])','_',cls).lower()]
        class_aliases[cls]=names
    # Target role descriptors in existing mechanisms can retain relative size
    # or shelf qualifiers. They are vocabulary, never a task lookup table.
    extra={}
    for m in graph['specific_mechanisms']:
        value=m['bindings'].get('$target')
        if value and value.count('_')<=5:
            for t in m.get('role_types',{}).get('$target',[]):extra.setdefault(t['class'],[]).append(value)
    for cls,values in extra.items():class_aliases.setdefault(cls,[]).extend(values)
    target_words=[v for values in class_aliases.values() for v in values]
    target_words.extend(m['bindings']['$target'] for m in graph['specific_mechanisms'] if '$target' in m['bindings'])
    targets=list(dict.fromkeys(word for _,_,word in mentioned(description,target_words)))
    requests=placement_requests(description,objects,targets)
    requested=set(requests)
    instances=[]
    identifiers={}
    def instance_id(template, mechanism):
        key=(template['id'], json.dumps({'bindings':mechanism['bindings'], 'constraints':mechanism['role_type_constraints']}, sort_keys=True))
        return identifiers.setdefault(key, f"query_binding:{len(identifiers)+1:04d}")
    for template in templates:
        action=template['action']
        bindings=[]
        if action=='PICK':bindings=[({'$object':o},False) for o in objects]
        elif action=='PLACE':
            cls=template['role_type_constraints']['$target']['class']
            for obj in objects:
                for target in targets:
                    if target not in class_aliases.get(cls,[]) or obj==target:continue
                    relation=template['bindings']['$relation']
                    bindings.append(({'$object':obj,'$target':target},(obj,target,relation) in requested))
        elif action=='ROTATE' and 'burner' in description.lower():
            positions=re.findall(r'\b(front|rear)[ -](left|right|center)\b',description.lower())
            fixtures=[a+'_'+b+'_burner' for a,b in positions] or ['$burner']
            bindings=[({'$fixture':f},True) for f in fixtures]
        for values,is_goal in bindings:
            m=deepcopy(template)
            replacements={template['bindings'][role]:value for role,value in values.items()}
            m['bindings']={**template['bindings'],**values}
            m['conditions']=apply_bindings(m['conditions'],replacements)
            m['effects']=apply_bindings(m['effects'],replacements)
            m['parent_template_id']=template['id']
            order={'PICK':['object'],'PLACE':['object','target','relation'],'ROTATE':['fixture','state']}[action]
            m['api_call']=action+'('+','.join(m['bindings']['$'+p] for p in order)+')'
            m['role_types']=dict(template['role_types'])
            if action=='ROTATE':
                m['role_type_constraints']['$fixture']['binding_policy']='same_burner_as_query_placed_cookware'
            for role,value in values.items():
                if role=='$object':
                    m['role_types'][role]=aliases[value]
                    if value in {'condiment_bottle','vegetable','fruit','meat','condiment','food'}:
                        m['role_type_constraints'][role]['group']='condiment' if value=='condiment_bottle' else value
            quantifier_target=(action=='PLACE' and re.search(r'\b(?:each|every)\s+(?:of the\s+)?'+re.escape(values['$target'].replace('_',' '))+r'(?:s|es)?\b',description.lower()))
            if quantifier_target:
                if template['role_type_constraints']['$target']['class'] in {'plate','tupperware','bowl','Plate','Tupperware','Bowl'} and not any(x in values['$target'] for x in ['larger','smaller','left','right','top','bottom','highest']):
                    role='$'+values['$target']+'_each'
                    m['effects']=apply_bindings(m['effects'],{values['$target']:role})
                    m['conditions']=apply_bindings(m['conditions'],{values['$target']:role})
                    m['bindings']['$target']=role
                    m['role_type_constraints']['$target']['binding_policy']='each_destination_with_one_object_of_each_requested_type'
                    m['api_call']=action+'('+','.join(m['bindings']['$'+p] for p in order)+')'
            m['query_goal_request']=is_goal or (action=='PICK' and bool(m['role_type_constraints']['$object'].get('group')))
            m['id']=instance_id(template,m)
            m['instantiation_basis']='shared_parameterized_contract + public_type_hierarchy + query_text_only'
            m['new_type_outcome_observed']=False
            instances.append(m)
    if re.search(r'\b(counterparts?|matching (?:object|item)s?)\b',description.lower()):
        for template in templates:
            if template['action']!='PLACE' or template['bindings']['$relation']!='NEAR':continue
            cls=template['role_type_constraints']['$target']['class']
            for obj in objects:
                group='condiment' if obj=='condiment_bottle' else None
                if not any(t['class']==cls for t in aliases[obj]) and not (group and group in hierarchy.get(cls,[])):continue
                m=deepcopy(template);target='$counterpart_of_'+obj
                repl={'$object':obj,'$target':target}
                m['bindings']={'$object':obj,'$target':target,'$relation':'NEAR'}
                m['conditions']=apply_bindings(m['conditions'],repl);m['effects']=apply_bindings(m['effects'],repl)
                m['role_types']['$object']=aliases[obj]
                for role in ['$object','$target']:
                    if group:m['role_type_constraints'][role]['group']=group
                m['role_type_constraints']['$target']['binding_policy']='counterpart_matching_query_object_type_or_category'
                m['role_type_constraints']['$target']['destination_context']=next((x for x in ['cabinet','fridge','drawer'] if x in description.lower()),None)
                m['parent_template_id']=template['id'];m['query_goal_request']=True
                m['api_call']='PLACE('+obj+','+target+',NEAR)'
                m['id']=instance_id(template,m)
                m['instantiation_basis']='query_counterpart_relation + public_type_hierarchy + shared_NEAR_contract'
                m['new_type_outcome_observed']=False;instances.append(m)
    # A requested support relation can use a carrier already requested on the
    # same support surface. This is an alternative subgraph, not an order proof.
    for m in instances:
        if m['action']!='PLACE':continue
        obj,target,rel=[m['bindings'][r] for r in ['$object','$target','$relation']]
        def same_support(a,b):
            pa=set(re.findall(r'(front|rear|left|right|center)',a));pb=set(re.findall(r'(front|rear|left|right|center)',b))
            if pa and pb and pa!=pb:return False
            return a==b or any(a in names and b in names for cls,names in class_aliases.items() if cls in {'Stove','Stovetop'})
        if rel=='ON' and any(a==obj and c=='ON' and any(x==target and same_support(y,b) and z=='ON' for x,y,z in requests) for a,b,c in requests):
            m['query_goal_request']=True
            m['goal_relation_basis']='requested_object_on_carrier_and_carrier_on_support'
            m['inferred_support_is_alternative']=True
    return instances,requests
