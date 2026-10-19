"""Extract entity/goal requests from text without trajectory annotations."""
import re

ALIASES={'kettle_non_electric':['kettle'],'hotdog_bun':['bun'],'sandwich_bread':['bread'],
         'coffee_machine_tray':['coffee machine dispenser','coffee machine tray'],
         'stove_burner':['burner','stove burner'],'fridge':['refrigerator','fridge'],
         'condiment_bottle':['condiment bottle'],'tupperware':['tupperware','tupperwares']}


def phrases(word):
    values=[word.replace('_',' '),*ALIASES.get(word,[])]
    group=re.fullmatch(r'other_(.+)_bottles?',word)
    if group:values.extend([group[1].replace('_',' ')+' group',group[1].replace('_',' ')+'s group'])
    return list(dict.fromkeys(values))


def matches(text,word):
    result=[]
    for p in phrases(word):
        result.extend((m.start(),m.end()) for m in re.finditer(r'\b'+re.escape(p)+r'(?:s|es)?\b',text.lower()))
    return result


def mentioned(text,words):
    spans=[(a,b,word) for word in words for a,b in matches(text,word)]
    # Prefer the complete phrase, e.g. toaster oven rather than its substring.
    return [(a,b,word) for a,b,word in sorted(set(spans)) if not any(c<=a and b<=d and d-c>b-a for c,d,_ in spans)]


def placement_requests(text,objects,targets):
    clauses=re.split(r'[.;]|\bthen\b|(?:,\s*|\band )(?=(?:place|put|turn|close|pick|start|take|press|move|retrieve|wash|stir|open|scrub)\b)',text.lower())
    result=[];previous=[]
    def subject_head(text):
        return re.split(r'\b(?:from|for|with|containing|that has|which contains)\b',text,1)[0]
    for clause in clauses:
        subjects=[word for _,_,word in mentioned(subject_head(clause),objects)]
        if re.search(r'\b(pick|retrieve|grab|take|select)\b',clause) and not re.search(r'\b(?:to|on|in|onto|into)\b',clause.split(' from ')[0]):
            previous=list(dict.fromkeys(subjects))
        if not re.search(r'\b(place|put|move|return|stack|take|wash)\b',clause):continue
        target_spans=mentioned(clause,targets)
        requests=[];last_target_end=0
        for start,end,target in target_spans:
            before=clause[:start]
            prep=re.search(r'\b(in|inside|into|on top of|onto|on|under|to|near|next to|alongside)\s+(?:(?:the|a|an|each|one|other)\s+)*$',before)
            if not prep:continue
            # "from the plate to the board" selects the destination after to.
            if re.search(r'\bfrom\s+(?:(?:the|a|an|each|one|other)\s+)*$',before):continue
            source_text=before[last_target_end:prep.start()]
            source=[word for _,_,word in mentioned(subject_head(source_text),objects) if word!=target]
            if not source and re.search(r'\b(it|them)\b',source_text):source=previous
            relation='IN' if prep[1] in {'in','inside','into'} else 'NEAR' if prep[1] in {'near','next to','alongside'} else 'ON'
            if prep[1]=='to' and re.search(r'\bgroups?\b',clause[start:end]):relation='NEAR'
            requests.extend((obj,target,relation) for obj in dict.fromkeys(source))
            if source:last_target_end=end
        result.extend(requests)
        if requests:previous=list(dict.fromkeys(x[0] for x in requests))
    return list(dict.fromkeys(result))
