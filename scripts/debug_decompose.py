import json
import re
from bin_eval.llm import call_llm_sync
from bin_eval.decomposer import DECOMPOSER_USER, DECOMPOSER_SYSTEM

schema = open('examples/sample_schema.json', 'r', encoding='utf-8').read()
task = 'Extract all invoice fields including metadata, billing info, line items, and totals'
prompt = DECOMPOSER_USER.format(task_prompt=task, extraction_schema=schema)
response = call_llm_sync(prompt=prompt, system=DECOMPOSER_SYSTEM)
print('RESPONSE LEN', len(response))
print('COUNT ```', response.count('```'))
print('ENDS WITH ```?', response.strip().endswith('```'))
print('START REPR')
print(repr(response[:1200]))
print('END REPR')
print(repr(response[-1200:]))
text = response.strip()
if text.startswith('```'):
    lines = text.split('\n')
    json_lines = []
    in_block = False
    for line in lines:
        if line.strip().startswith('```') and not in_block:
            in_block = True
            continue
        elif line.strip() == '```' and in_block:
            break
        elif in_block:
            json_lines.append(line)
    extracted = '\n'.join(json_lines)
else:
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    extracted = match.group(1) if match else text
print('EXTRACTED LEN', len(extracted))
print('EXTRACTED START')
print(repr(extracted[:1200]))
print('EXTRACTED END')
print(repr(extracted[-1200:]))
try:
    data = json.loads(extracted)
    print('PARSED OK keys', list(data.keys()))
    print('REQS', len(data.get('requirements', [])))
    print('Q', len(data.get('questions', [])))
except Exception as e:
    print('PARSE ERROR', type(e).__name__, e)
    if hasattr(e, 'pos'):
        pos = e.pos
        print('ERROR POS', pos)
        print('BEFORE', repr(extracted[max(0, pos-80):pos]))
        print('AFTER', repr(extracted[pos:pos+80]))
