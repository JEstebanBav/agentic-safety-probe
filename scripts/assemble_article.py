"""Assemble the complete research text from English sections and table transcriptions."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT/'docs/paper'


def bib_field(entry, name):
    match = re.search(r'\b'+name+r'\s*=\s*\{', entry)
    if not match:
        return ''
    start = i = match.end()
    depth = 1
    while depth:
        depth += (entry[i] == '{') - (entry[i] == '}')
        i += 1
    return entry[start:i-1].replace('{','').replace('}','')


def main():
    tables = json.loads((PAPER/'evidence/table_transcription.json').read_text(encoding='utf-8'))
    bib = (PAPER/'references.bib').read_text(encoding='utf-8')
    entries = {}
    for match in re.finditer(r'@\w+\{([^,]+),', bib):
        end = bib.find('\n@', match.end())
        entries[match.group(1)] = bib[match.start():end if end != -1 else len(bib)]
    citations = {key:i+1 for i,key in enumerate(entries)}
    references = []
    for key, entry in entries.items():
        title = bib_field(entry,'title')
        url = bib_field(entry,'url') or 'https://doi.org/'+bib_field(entry,'doi')
        venue = (bib_field(entry,'booktitle') or bib_field(entry,'journal') or
                 bib_field(entry,'institution') or bib_field(entry,'organization') or
                 bib_field(entry,'note') or 'Preprint')
        authors = bib_field(entry,'author').replace(' and ', '; ')
        references.append(f'[{citations[key]}] {authors} ({bib_field(entry,"year")}). [{title}]({url}). {venue}.')
    sections = sorted((PAPER/'sections').glob('*.md'))
    assert len(sections) == 8, sections
    article = '\n\n'.join(path.read_text(encoding='utf-8').strip() for path in sections)+'\n\n## References\n\n'+'\n\n'.join(references)+'\n'
    for key, item in tables.items():
        heading = '**Table '+str(item['number'])+'. '+item['caption']+'**\n\n'
        rendered = heading+'| '+' | '.join(item['headers'])+' |\n'
        rendered += '| '+' | '.join(['---']*len(item['headers']))+' |\n'
        rendered += '\n'.join('| '+' | '.join(row)+' |' for row in item['rows'])
        assert article.count('{{table:'+key+'}}') == 1, key
        article = article.replace('{{table:'+key+'}}',rendered)
        article = article.replace('{{t:'+key+'}}','Table '+str(item['number']))
    article = re.sub(r'\{\{c:([^}]+)\}\}',lambda m:'['+str(citations[m[1]])+']',article)
    if '{{' in article:
        raise ValueError('Unresolved article placeholders: '+str(re.findall(r'\{\{.*?\}\}',article)))
    (PAPER/'article.md').write_text(article,encoding='utf-8')
    print(f'Assembled {len(sections)} research sections, {len(tables)} tables and {len(references)} references.')


if __name__ == '__main__':
    main()
