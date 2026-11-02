"""Render the English manuscript without a LaTeX installation or model calls.

Install documentation dependencies from requirements-docs.txt, then run:
    python scripts/render_paper.py

article.md is the editable source. This script only writes article.html,
article.pdf and article.docx beside it; it does not rewrite the source.
"""
from __future__ import annotations

import base64
import re
import shutil
from pathlib import Path

import pymupdf
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE
from lxml import html as lhtml
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / 'docs/paper'
AUTHORS = ('Juan Esteban Espitia Leiva', 'Helen Stefany Penagos Galvis')
CSS = '''
body { font-family: serif; font-size: 12pt; line-height: 2; color: #18222c; }
h1 { font-size: 22pt; line-height: 1.12; color: #172e45; margin: 0 0 14pt; }
h2 { font-size: 17pt; line-height: 1.3; color: #172e45; margin: 18pt 0 12pt; page-break-after: avoid; }
h3 { font-size: 13pt; line-height: 1.3; color: #215969; margin: 16pt 0 9pt; page-break-after: avoid; }
h4 { font-size: 12pt; line-height: 1.3; color: #215969; margin: 14pt 0 8pt; page-break-after: avoid; }
h5 { font-size: 12pt; line-height: 1.3; font-style: italic; color: #18222c; margin: 12pt 0 7pt; page-break-after: avoid; }
.section-start { page-break-before: always; }
.heading-lead { page-break-inside: avoid; break-inside: avoid; }
.article-front { line-height: 1.5; }
p { margin: 0 0 9pt; } a { color: #17547a; text-decoration: none; }
table { border-collapse: collapse; width: 100%; font-size: 10pt; line-height: 1.15; margin: 9pt 0 12pt; }
.research-table { page-break-inside: avoid; break-inside: avoid; }
th, td { padding: 4pt 3pt; border-bottom: 0.5pt solid #ccd3da; vertical-align: top; }
th { font-weight: bold; text-align: left; }
img { max-width: 100%; height: auto; }
figure { margin: 12pt 0; } .caption { font-size: 10pt; line-height: 1.25; color: #3c4a56; }
.figure-block { font-size: 10pt; line-height: 1.25; }
.table-caption { font-size: 10pt; line-height: 1.25; page-break-after: avoid; }
.equation { font-size: 12pt; line-height: 1.4; padding: 7pt 0; }
code { font-family: monospace; font-size: 10pt; }
'''


def hyperlink(paragraph, text, url):
    rel = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    link = OxmlElement('w:hyperlink')
    link.set(qn('r:id'), rel)
    run = OxmlElement('w:r')
    props = OxmlElement('w:rPr')
    colour = OxmlElement('w:color')
    colour.set(qn('w:val'), '17547A')
    props.append(colour)
    run.append(props)
    value = OxmlElement('w:t')
    value.text = text
    run.append(value)
    link.append(run)
    paragraph._p.append(link)


def inline(paragraph, node, bold=False, italic=False, sub=False, sup=False):
    def add(value):
        if not value:
            return
        run = paragraph.add_run(value)
        run.bold = bold
        run.italic = italic
        run.font.subscript = sub
        run.font.superscript = sup
    add(node.text)
    for child in node:
        tag = child.tag.lower()
        if tag == 'a':
            hyperlink(paragraph, child.text_content(), child.get('href', ''))
        elif tag == 'br':
            paragraph.add_run().add_break()
        elif tag == 'img':
            pass
        else:
            inline(paragraph, child, bold or tag in ('b', 'strong'), italic or tag in ('i', 'em'),
                   sub or tag == 'sub', sup or tag == 'sup')
        add(child.tail)


def write_docx(markup):
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = section.bottom_margin = Inches(1)
    section.left_margin = section.right_margin = Inches(1)
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(12)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 2
    for name in ['Heading 1', 'Heading 2', 'Heading 3', 'Heading 4', 'Title']:
        doc.styles[name].font.color.rgb = RGBColor.from_string('172E45')
        doc.styles[name].font.name = 'Times New Roman'
    body = lhtml.fragment_fromstring(markup, create_parent='div')
    front_matter = True
    for element in body:
        tag = element.tag.lower()
        if tag in ['h1', 'h2', 'h3', 'h4', 'h5']:
            p = doc.add_paragraph(style={'h1': 'Title', 'h2': 'Heading 1', 'h3': 'Heading 2', 'h4': 'Heading 3', 'h5': 'Heading 4'}[tag])
            p.paragraph_format.keep_with_next = True
            inline(p, element)
            if tag == 'h2' and element.text_content() != 'Abstract':
                front_matter = False
                p.paragraph_format.page_break_before = True
        elif tag == 'table':
            rows = element.xpath('./thead/tr | ./tbody/tr | ./tr')
            cols = len(rows[0])
            t = doc.add_table(rows=0, cols=cols)
            t.style = 'Light Shading Accent 1'
            for i, row in enumerate(rows):
                cells = t.add_row().cells
                for c, cell in zip(cells, row):
                    p = c.paragraphs[0]
                    inline(p, cell, bold=(i == 0))
                    for run in p.runs:
                        run.font.size = Pt(10)
                    p.paragraph_format.line_spacing = 1
                    p.paragraph_format.space_after = Pt(2)
                    p.paragraph_format.space_before = Pt(2)
                if i == 0:
                    repeat = OxmlElement('w:tblHeader')
                    t.rows[-1]._tr.get_or_add_trPr().append(repeat)
            doc.add_paragraph().paragraph_format.space_after = Pt(1)
        elif tag == 'p' and element.find('img') is not None:
            img = element.find('img')
            p = doc.add_paragraph()
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(PAPER / img.get('src')), width=Inches(6.5))
            caption = doc.add_paragraph(img.get('alt', ''))
            caption.paragraph_format.line_spacing = 1
            for r in caption.runs:
                r.font.size = Pt(10)
                r.italic = True
        elif tag == 'p':
            p = doc.add_paragraph()
            inline(p, element)
            if front_matter:
                p.paragraph_format.line_spacing = 1.5
            if element.get('class') == 'equation':
                p.paragraph_format.left_indent = Inches(0.15)
                p.paragraph_format.line_spacing = 1.4
            if element.text_content().startswith('Table '):
                p.paragraph_format.line_spacing = 1
                p.paragraph_format.keep_with_next = True
                for run in p.runs:
                    run.font.size = Pt(10)
        elif tag in ('ul', 'ol'):
            for item in element:
                p = doc.add_paragraph(style='List Bullet' if tag == 'ul' else 'List Number')
                inline(p, item)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('Refusal Direction in Agentic Language Models  |  ')
    page = OxmlElement('w:fldSimple')
    page.set(qn('w:instr'), 'PAGE')
    footer._p.append(page)
    for r in footer.runs:
        r.font.size = Pt(8)
    doc.core_properties.title = 'Transformation of the Refusal Direction in Agentic Language Models'
    doc.core_properties.author = '; '.join(AUTHORS)
    doc.core_properties.language = 'en'
    doc.save(PAPER / 'article.docx')


def main():
    source = (PAPER / 'article.md').read_text(encoding='utf-8')
    if '{{' in source:
        raise ValueError('Unresolved manuscript placeholders')
    markdown = MarkdownIt('commonmark', {'html': True}).enable('table')
    markup = markdown.render(source)
    # Put figure captions below figures in all rendered formats.
    def figure(match):
        element = lhtml.fromstring(match.group(0))
        element.set('width', '468')
        alt = element.get('alt', '')
        from html import escape
        return lhtml.tostring(element, encoding='unicode') + '<br/><span class="caption">' + escape(alt) + '</span>'
    pdf_markup = re.sub(r'<img\b[^>]*>', figure, markup)
    pdf_markup = pdf_markup.replace('<p><img', '<p class="figure-block"><img')
    pdf_markup = re.sub(r'<h2>(?!Abstract</h2>)(.*?)</h2>',
                        r'<h2 class="section-start">\1</h2>', pdf_markup)
    front, first_section = pdf_markup.split('<h2 class="section-start">', 1)
    pdf_markup = '<div class="article-front">'+front+'</div><h2 class="section-start">'+first_section
    pdf_markup = re.sub(r'<p><strong>(Table \d+\..*?)</strong></p>',
                        r'<p class="table-caption"><strong>\1</strong></p>', pdf_markup, flags=re.S)
    pdf_markup = re.sub(r'(<p class="table-caption"><strong>Table \d+\..*?</p>\s*<table>.*?</table>)',
                        r'<div class="research-table">\1</div>', pdf_markup, flags=re.S)
    page_rect = pymupdf.paper_rect('letter')
    frame = pymupdf.Rect(72, 72, page_rect.width-72, page_rect.height-72)
    # Keep each subsection heading with its opening research paragraph.
    pdf_markup = re.sub(r'((?:<h[345]\b[^>]*>.*?</h[345]>\s*)+<p\b[^>]*>.*?</p>)',
                        r'<div class="heading-lead">\1</div>', pdf_markup, flags=re.S)
    story = pymupdf.Story(pdf_markup, user_css=CSS, archive=pymupdf.Archive(str(PAPER)))
    pdf = story.write_with_links(lambda n, filled: (page_rect, frame, None))
    for i, page in enumerate(pdf):
        page.insert_text((46, page_rect.height-23), 'REFUSAL DIRECTION IN AGENTIC LANGUAGE MODELS', fontsize=7, color=(0.30,0.36,0.42))
        page.insert_text((page_rect.width-60, page_rect.height-23), str(i+1), fontsize=8)
    toc=[]
    for heading in re.findall(r'^## (.+)$',source,re.M):
        for i,page in enumerate(pdf):
            if heading in page.get_text():
                toc.append([1,heading,i+1])
                break
    if toc:
        pdf.set_toc(toc)
    pdf.set_metadata({'title':'Transformation of the Refusal Direction in Agentic Language Models',
                      'author':'; '.join(AUTHORS),'subject':'AI safety research: refusal representations in agentic language models'})
    pdf.save(PAPER/'article.pdf',deflate=True)
    pages=len(pdf)
    pdf.close()

    def embed(match):
        path=PAPER/match.group(1)
        encoded=base64.b64encode(path.read_bytes()).decode('ascii')
        return 'src="data:image/png;base64,'+encoded+'"'
    standalone=re.sub(r'src="(figures/[^"]+)"',embed,pdf_markup)
    web_css=CSS+'''@page {size:letter;margin:1in;} @media screen {body{max-width:6.5in;margin:32px auto;padding:30px;}}
    @media print {h2,h3,h4,h5{break-after:avoid;}tr{break-inside:avoid;}}'''
    (PAPER/'article.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="author" content="'+ '; '.join(AUTHORS) +'">'
        '<title>Refusal Direction in Agentic Language Models</title>'
        '<style>'+web_css+'</style></head><body>'+standalone+'</body></html>',encoding='utf-8')
    write_docx(markup)
    for suffix in ('pdf', 'docx'):
        shutil.copyfile(PAPER/f'article.{suffix}', PAPER/f'article_full_research.{suffix}')
    print(f'Rendered article.pdf ({pages} pages), article.docx and standalone article.html')


if __name__ == '__main__':
    main()
