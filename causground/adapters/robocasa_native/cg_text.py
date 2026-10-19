"""Scoped subject/destination parsing for the dependency interface."""
import re
from .cg_goal_requests import phrases as original_phrases


def phrases(word):
    values=original_phrases(word)
    if word=='jam':values=values+['jar of jam']
    if word=='baguette':values=values+['baguette bread']
    if word=='sink_spout':values=values+['spout']
    return values


def matches(text,word):
    return [(m.start(),m.end()) for phrase in phrases(word) for m in re.finditer(r'\b'+re.escape(phrase)+r'(?:s|es)?\b',text.lower())]


def mentioned(text,words):
    spans=sorted(set((a,b,w) for w in words for a,b in matches(text,w)))
    return [(a,b,w) for a,b,w in spans if not any(c<=a and b<=d and d-c>b-a for c,d,_ in spans)]


def subjects(text,objects):
    text=re.sub(r'\bfrom\s+.*?(?=\s+and\s+(?:one|a|an|the)\b|,|$)','',text)
    text=re.split(r'\b(?:for|with|containing|that has|which contains|alongside|next to|near)\b',text,1)[0]
    return list(dict.fromkeys(w for _,_,w in mentioned(text,objects)))


def pickup_subjects(text,objects):
    text=re.split(r'\b(?:and\s+)?(?:(?:lightly|briefly)\s+)?(?:place|put|stir|scrub|clean|press|turn|release)\w*\b',text,1)[0]
    text=re.sub(r'\bfrom\s+.*?(?=\s+and\s+(?:one|a|an|the)\b|,|$)','',text)
    return subjects(re.split(r'\b(?:to|on|in|into|inside|onto)\b',text,1)[0],objects)


def mask_negations(text):
    prohibitions=[]
    def mask(match):
        prohibitions.append({'text':match.group(),'offset':match.start(),'enforcement_verified':False})
        return ' '*len(match.group())
    return re.sub(r"\b(?:do not|don't|never|without|avoid)\b.*?(?=[.;]|\bthen\b|$)",mask,text),prohibitions


def expand_collective_requests(requests,text,objects):
    result=[]
    for request in requests:
        obj=request[0]
        plural=bool(re.search(r'\b'+re.escape(obj.replace('_',' '))+r'(?:s|es)\b',text))
        members=[word for _,_,word in mentioned(text,objects) if word.endswith('_'+obj) and word!=obj]
        members=list(dict.fromkeys(members))
        for member in members if plural and len(members)>1 else [obj]:result.append((member,*request[1:]))
    return result


def placement_requests(text,objects,targets,with_offsets=False):
    separators=re.compile(r'[.;]|\bthen\b|(?:,\s*(?:and\s+)?|\band )(?=(?:place|put|turn|close|pick|start|take|press|move|retrieve|wash|stir|open|scrub|stack)\b)')
    chunks=[];start=0;text=text.lower()
    for match in separators.finditer(text):
        chunks.append((start,text[start:match.start()]));start=match.end()
    chunks.append((start,text[start:]))
    previous=[];result=[]
    for offset,clause in chunks:
        pickup=re.search(r'\b(?:pick|retrieve|grab|take|select)\b',clause)
        placement=re.search(r'\b(?:place|put|move|return|stack|wash|gather)\b',clause)
        if pickup:previous=pickup_subjects(clause[:placement.start()] if placement else clause,objects) or previous
        if not placement:
            if not (pickup and re.search(r'\btake\b.*\bto\b',clause)):continue
            placement=pickup
        requests=[];last_end=placement.start();adjacent_subjects=[]
        for start,end,target in mentioned(clause,targets):
            if start<placement.start():continue
            before=clause[:start]
            prep=re.search(r'\b(in|inside|into|on top of|onto|on|under|to|near|next to|alongside)\s+(?:(?:the|a|an|each|one|other|open|closed)\s+)*$',before)
            if not prep:continue
            source=subjects(before[last_end:prep.start()],objects)
            if not source and re.search(r'\b(?:it|them|both)\b',before[last_end:prep.start()]):source=previous
            if not source and adjacent_subjects:source=adjacent_subjects
            source=[o for o in source if o!=target]
            rel='IN' if prep[1] in {'in','inside','into'} else 'NEAR' if prep[1] in {'near','next to','alongside'} else 'ON'
            if prep[1]=='to' and 'group' in clause[start:end]:rel='NEAR'
            requests.extend((o,target,rel) for o in source)
            if rel=='NEAR' and source:adjacent_subjects=source
            if source:last_end=end
        result.extend((*request,offset+placement.start()) if with_offsets else request for request in requests)
        if requests:previous=list(dict.fromkeys(o for o,_,_ in requests))
    return result if with_offsets else list(dict.fromkeys(result))
