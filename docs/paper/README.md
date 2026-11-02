# Research article

[PDF](article.pdf) · [Editable Word document](article.docx) · [Complete Markdown](article.md) · [Standalone HTML](article.html) · [Related research](top_10_related_research.md)

[English research presentation](Research_Presentation.pptx): 36 slides covering the current 96-pair results, methodology, limitations and references.

[Full research PDF](article_full_research.pdf) · [Full research Word document](article_full_research.docx). These files contain the same current text as `article.pdf` and `article.docx`. The reading layout uses US Letter, 12-point text, double-spaced body paragraphs and one-inch margins. Tables and figure captions use tighter spacing for readability.

**Juan Esteban Espitia Leiva and Helen Stefany Penagos Galvis**  
Research AI Safety

The article contains the complete scientific account: problem statement, research question and objectives, context and related work, research design, corpus construction and verification, mathematical formulation, evaluation procedures, results, discussion, conclusions, contributions and recommendations. Supporting sections retain experimental configuration, reproducibility details, notation and abbreviations. The article has 16 research tables and seven figures and preserves the complete 17-entry reference collection.

The content preserves the reported methodology, numerical values, confidence intervals, rounding, model-specific findings and limitations. No experiments were rerun and no later research findings were substituted. The manuscript shows both authors without a university affiliation and contains no degree-presentation material or document-generation instructions.

The editable research text is organised in [sections/](sections/). Table transcriptions are in [table_transcription.json](evidence/table_transcription.json); numerical evidence and source hashes are in [metrics.json](evidence/metrics.json). The internal coverage record documents the alignment of the scientific content and the limited presentation changes. These records are separate from the PDF and Word document.

After editing section files, assemble and render from the repository root:

```powershell
python scripts/assemble_article.py
python scripts/render_paper.py
```

To rebuild the English PowerPoint presentation from the same saved evidence and figures:

```powershell
python scripts/build_research_presentation.py
```

`article.md` is the assembled Markdown document; change the section files to preserve edits across assembly. Optional rendering dependencies are listed in `requirements-docs.txt`. On the original computer the workspace interpreter is `..\.venv\Scripts\python.exe`.

To regenerate numerical evidence and figures from saved runs, use `python scripts/build_paper.py` with the experiment dependencies installed. This does not update the transcribed research text or tables automatically. It does not run models.

For a focused overview of the literature, see the [top 10 related studies](top_10_related_research.md).
