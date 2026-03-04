# Fix Figure 1: Remove Embedded Text from Architecture Diagram

## Problem
The compiled PDF shows "￾" characters and embedded title/spec text **inside the figure image itself**. These must be removed from the graphic file before PDF generation.

## Solution
You must edit `figures/architecture.pdf` to remove ALL embedded text, leaving only the diagram boxes, arrows, and pipeline stage labels.

### Option A: Use Adobe Acrobat / Preview (Easiest for PDF)
1. Open `figures/architecture.pdf` in Adobe Acrobat or Preview
2. Identify and delete these elements:
   - Title text: "CalibraTeach: 7-Stage Real-Time Fact Verification Pipeline"
   - Spec line: "NVIDIA RTX 4090 GPU, 24GB, FP16 precision, batch size 1"
   - Performance line: "Mean latency: 67.68ms, Throughput: 14.78 claims/sec"
   - Any other metadata text inside the graphic
3. Keep only:
   - Boxes for each pipeline stage (Evidence Retrieval, Relevance Filtering, etc.)
   - Arrows connecting boxes
   - Stage labels inside/near boxes
4. Save as `figures/architecture.pdf` (overwrite)

### Option B: Redraw in Draw.io / Figma / PowerPoint (Better Quality)
1. Open a new diagram tool (draw.io, Figma, PowerPoint, or Visio)
2. Create the 7-stage pipeline:
   - Stage 1: Evidence Retrieval
   - Stage 2: Relevance Filtering
   - Stage 3: NLI Ensemble
   - Stage 4: Confidence Aggregation
   - Stage 5: Temperature Scaling
   - Stage 6: Selective Prediction
   - Stage 7: Explanation Generation
3. Use plain boxes, arrows, and text labels ONLY - no embedded titles or specs
4. Export as PDF → `figures/architecture.pdf`

### Option C: Use LaTeX TikZ (Professional, Automated)
Create a file `figures/architecture_tikz.tex` with a TikZ diagram, then include it with:
```latex
\input{figures/architecture_tikz.tex}
```
This ensures zero hidden characters.

## Verification Checklist
After editing, verify that the resulting PDF shows:
- [ ] Figure 1 spans two columns (wide layout)
- [ ] NO title text visible inside the figure
- [ ] NO "GPU:", "PyTorch", "Latency", "Throughput" text inside figure
- [ ] Only pipeline stage boxes/arrows/labels visible
- [ ] All details (hardware, performance) appear ONLY in the caption below
- [ ] Zero "￾" characters anywhere in compiled PDF
- [ ] Text search for "GPU:" returns only matches in caption, not in figure image

## LaTeX Configuration (Already Applied)
The main .tex file now has:
```latex
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
```
This prevents hidden character encoding issues during compilation.

---

**Next Step:** Edit `figures/architecture.pdf` using one of the methods above, then recompile the .tex file and verify using PDF text search.
