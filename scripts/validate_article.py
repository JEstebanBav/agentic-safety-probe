"""Check scientific content and exported documents; optionally compare the source."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pymupdf
from docx import Document
from lxml import html

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT/'docs/paper'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numbers(text, original=False):
    if original:
        text = re.sub(r'\\(?:eqref|ref|label|textcite|parencite)\{[^}]+\}', '', text)
        text = re.sub(r'\\includegraphics(?:\[[^]]*\])?\{[^}]+\}', '', text)
        text = re.sub(r'(?<=\d)\.(?=\d{3}\b)', '', text)
        text = text.replace('{,}', '.')
    text = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)
    text = re.sub(r'<[^>]+>', ' ', text).replace('\u2212', '-')
    return set(re.findall(r'(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])', text))


def cell_numbers(text):
    text = text.replace('\u2212', '-')
    return re.findall(r'-?\d+(?:\.\d+)?', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    md = (PAPER/'article.md').read_text(encoding='utf-8')
    pdf = pymupdf.open(PAPER/'article.pdf')
    pdf_text = '\n'.join(p.get_text() for p in pdf)
    word = Document(PAPER/'article.docx')
    word_text = '\n'.join(p.text for p in word.paragraphs)
    web = (PAPER/'article.html').read_text(encoding='utf-8')
    transcriptions = json.loads((PAPER/'evidence/table_transcription.json').read_text(encoding='utf-8'))
    record_path = PAPER/'evidence/document_sources.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    report = {
        'pdf_pages':len(pdf),
        'markdown_words':len(md.split()),
        'markdown_tables':len(re.findall(r'^\*\*Table \d+\.',md,re.M)),
        'pdf_tables':len(set(re.findall(r'Table (\d+)\.',pdf_text))),
        'pdf_embedded_images':sum(len(p.get_images()) for p in pdf),
        'docx_tables':len(word.tables),
        'docx_images':len(word.inline_shapes),
        'display_equations':len(re.findall(r'<p class="equation">',md)),
        'printed_references':len(re.findall(r'^\[\d+\] ',md,re.M)),
        'pdf_external_links':sum(1 for p in pdf for link in p.get_links() if link.get('uri')),
        'standalone_html_embedded_images':web.count('src="data:image/png;base64,'),
        'replacement_characters_in_pdf':pdf_text.count('\ufffd'),
        'unresolved_placeholders':'{{' in md,
    }
    assert report['markdown_tables']==report['pdf_tables']==report['docx_tables']==16,report
    assert report['pdf_embedded_images']==report['docx_images']==report['standalone_html_embedded_images']==7,report
    assert report['display_equations']==16 and report['printed_references']==17,report
    assert not report['unresolved_placeholders'] and report['replacement_characters_in_pdf']==0
    assert all(len(p.get_text().strip())>100 for p in pdf)
    assert all(abs(p.rect.width-612)<0.1 and abs(p.rect.height-792)<0.1 for p in pdf)
    assert word.styles['Normal'].font.size.pt==12
    assert word.styles['Normal'].paragraph_format.line_spacing==2
    assert sha(PAPER/'article_full_research.pdf')==sha(PAPER/'article.pdf')
    assert sha(PAPER/'article_full_research.docx')==sha(PAPER/'article.docx')
    report['layout']={'page_size':'US Letter','body_font_points':12,'body_line_spacing':2,'margins_inches':1}
    report['full_research_aliases_match']=True

    authors=['Juan Esteban Espitia Leiva','Helen Stefany Penagos Galvis']
    for author in authors:
        assert author in pdf[0].get_text() and author in pdf.metadata['author']
        assert author in word_text and author in word.core_properties.author
    for body in (md,pdf_text,word_text,web):
        assert 'Research AI Safety' in body
        for phrase in ['Universidad','Externado','thesis','dissertation','Overleaf','scripts/render_paper.py',
                       'scripts/build_paper.py','Spanish source','degree documents','English manuscript']:
            assert not re.search(r'\b'+re.escape(phrase)+r'\b',body,re.I),phrase
    assert 'Claude' in md and 'second independent classification' in md

    # Check every numeric cell in all 96 source-table rows after export.
    for i,(key,item) in enumerate(transcriptions.items()):
        table=word.tables[i]
        assert len(table.rows)==len(item['rows'])+1,key
        for row_index,expected in enumerate(item['rows'],1):
            actual=[cell.text for cell in table.rows[row_index].cells]
            assert len(actual)==len(expected),key
            for source_cell,word_cell in zip(expected,actual):
                visible=html.fragment_fromstring('<span>'+source_cell+'</span>').text_content()
                assert cell_numbers(visible)==cell_numbers(word_cell),(key,source_cell,word_cell)
    report['all_96_table_rows_preserved_in_docx']=True

    # Deep section headings must survive the editable export as well.
    for heading in re.findall(r'^#{3,5} (.+)$',md,re.M):
        assert heading in word_text,heading
    report['all_research_subheadings_present_in_docx']=True
    assert sha(PAPER/'evidence/metrics.json')==record['english_metrics_sha256']
    report['numerical_evidence_unchanged']=True

    source_check={}
    section_map=[
        ('chapters/Chapter1/1.5_chapter01-introduction.tex','01_introduction.md','## 1. Introduction'),
        ('chapters/Chapter2/chapter02.tex','02_context_and_related_work.md','## 2. Context and related work'),
        ('chapters/Chapter3/chapter03.tex','03_design_and_methods.md','## 3. Research design and methods'),
        ('chapters/Chapter4/chapter04.tex','04_results_and_evaluation.md','## 4. Results and evaluation'),
        ('chapters/Chapter5/chapter05.tex','05_discussion.md','## 5. Discussion'),
        ('chapters/Chapter6/chapter06.tex','06_conclusions_and_recommendations.md','## 6. Conclusions and recommendations'),
    ]
    if args.source_root:
        original_texts=[]
        for original_path,target_path,heading in section_map:
            original=(args.source_root/original_path).read_text(encoding='utf-8')
            original_texts.append(original)
            target=md.split(heading,1)[1].split('\n## ',1)[0]
            original_headings=re.findall(r'\\(?:section|subsection|subsubsection)\*?\{([^}]+)\}',original)
            if target_path=='01_introduction.md':
                original_headings.remove('Estructura del documento')
            translated=(PAPER/'sections'/target_path).read_text(encoding='utf-8')
            english_headings=re.findall(r'^#{3,5} (.+)$',translated,re.M)
            assert len(original_headings)==len(english_headings),(original_path,original_headings,english_headings)
            missing=sorted(numbers(original,True)-numbers(target))
            assert not missing,(original_path,missing)
            source_check[original_path]={
                'translation':f'sections/{target_path}',
                'source_sha256':sha(args.source_root/original_path),
                'source_research_subheadings':original_headings,
                'translated_research_subheadings':english_headings,
                'missing_numeric_values':missing,
            }
        source_citations=set(re.findall(r'\\(?:textcite|parencite)\{([^}]+)\}','\n'.join(original_texts)))
        target_citations=set(re.findall(r'\{\{c:([^}]+)\}\}','\n'.join(p.read_text(encoding='utf-8') for p in (PAPER/'sections').glob('*.md'))))
        assert source_citations==target_citations,(source_citations,target_citations)
        report['all_16_source_citation_keys_preserved']=True
        report['all_reported_numeric_values_preserved']=True
        original_metrics=args.source_root/'evidencia/metricas_finales.json'
        assert json.loads(original_metrics.read_text(encoding='utf-8'))==json.loads((PAPER/'evidence/metrics.json').read_text(encoding='utf-8'))
        report['metrics_equal_to_final_source']=True
        report['source_hashes_valid']=all(sha(args.source_root/path.split('docs/tesis/Tesis_final/',1)[1])==value for path,value in record['source_files_sha256'].items())
        assert report['source_hashes_valid']
        original_equations=sum(len(re.findall(r'\\begin\{equation\}',text)) for text in original_texts)
        assert original_equations==report['display_equations'],original_equations
        report['all_16_source_display_equations_retained']=True
        source_check['presentation_exclusions']=[
            'Institutional cover, university affiliation, degree-modality and advisor blocks.',
            'Duplicated-language abstract, front-matter indexes and document-roadmap paragraphs.',
            'Degree-related wording and the university-setting clause; the research rationale is retained.',
            'File-production instructions and archive/translation commentary; experimental reproducibility is retained.',
        ]
        source_check['bibliography_resolution']='Original 17-entry bibliography retained; the Pinto URL is pinned to v1 to match the original title and claims. Later v3 findings are excluded.'
        source_check['methodological_scope']='English prose follows the original scientific sections and preserves their reported claims and limitations; headings, equation/table numbering and decimal typography follow the standalone format.'
        (PAPER/'evidence/content_alignment.json').write_text(json.dumps(source_check,ensure_ascii=False,indent=2),encoding='utf-8')

    report['editorial_checks']={'authors':authors,'tagline':'Research AI Safety','university_affiliation_absent':True,'document_production_notes_absent':True}
    report['output_sha256']={name:sha(PAPER/name) for name in ['article.md','article.pdf','article.docx','article.html','references.bib','top_10_related_research.md','research_contribution.md']}
    report['verified']=True
    (PAPER/'evidence/document_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    record['adaptation_date']='2026-10-29'
    record['table_count']=16
    record['figure_count']=7
    record['translation_scope']='Complete scientific sections, rather than a condensed article.'
    record['layout']={'page_size':'US Letter','body_font_points':12,'body_line_spacing':2,'margins_inches':1}
    record['printed_reference_keys']=re.findall(r'@\w+\{([^,]+),',(PAPER/'references.bib').read_text(encoding='utf-8'))
    record['bibliography_change']='Retain original reference collection and pin Pinto to v1; remove later v3 material added in the condensed version.'
    record['coverage']['results']='All 16 source tables, all seven figures, and all source results prose; the added probe-comparison table is removed and the original probe-comparison paragraph retained.'
    record['coverage']['literature']='Complete original research background and literature review with every original citation key.'
    record['coverage']['methodology']='Complete original study design, corpus construction, examples, lexical verification, mathematical formulation, analysis procedure, reproducibility, validation, intervention and ethics.'
    record['coverage']['introduction']='Original problem statement, research question, rationale and all research objectives; administrative document roadmaps excluded.'
    record['coverage']['discussion']='All original discussion subsections, limitations, conclusions, contributions and recommendations.'
    record['coverage']['bibliography']='Complete original 17-entry bibliography, including the regional and industry references used in the original rationale; later Pinto v3 material excluded.'
    record['coverage']['supporting_sections']='Original experimental configuration, research reproducibility, all 14 notation entries and all 12 abbreviations.'
    record['coverage']['removed']='Institutional cover, university affiliation, degree-modality and advisor blocks, duplicate-language abstract, front-matter indexes, document roadmaps, and document-production instructions.'
    record['coverage'].pop('uncited_contextual_references',None)
    record['editorial_revision']='Scientific paragraphs expanded against the original wording; US Letter, 12-point type and double-spaced body text with one-inch margins. Both authors and Research AI Safety retained; institutional and document-production material excluded.'
    record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
