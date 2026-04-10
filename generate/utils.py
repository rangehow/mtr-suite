import re

def split_questions(text, cid):
 # regexmatchquestion
    
    if '\n</think>\n\n' in text:
        text = text.split('\n</think>\n\n')[-1]
    pattern = r'\[(\d+)\]\s*(.+?)\s*$'
    
 # matchcontent
    matches = re.findall(pattern, text, re.MULTILINE)

    result = []
 # retain
    for did, query in matches[:1]:
        result.append({'did': did, 'query': query.strip(), 'cid': cid})
    
    return result