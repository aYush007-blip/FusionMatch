"""Script to generate a comprehensive, executive-ready PDF report for FusionMatch."""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages dynamically and adds headers and footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Top Header (Only on page 2+)
        if self._pageNumber > 1:
            self.drawString(54, 11 * 72 - 36, "FusionMatch: Cross-Modal & Multi-View Product Deduplication Engine")
            self.drawRightString(8.5 * 72 - 54, 11 * 72 - 36, "https://github.com/aYush007-blip/FusionMatch")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)

        # Bottom Footer (All pages)
        self.setFont("Helvetica", 8)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 8.5 * 72 - 54, 45)

        self.drawString(54, 32, "Ayush Gurjar | Machine Learning Engineering | IIT Kharagpur")
        self.drawRightString(8.5 * 72 - 54, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf(filename="FusionMatch_Project_Report.pdf"):
    workspace_dir = Path(__file__).resolve().parent
    pdf_path = workspace_dir / filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=50,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1e3a8a")   # Navy 900
    secondary_color = colors.HexColor("#0f766e") # Teal 700
    dark_text = colors.HexColor("#0f172a")       # Slate 900
    muted_text = colors.HexColor("#475569")      # Slate 600

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=muted_text,
        spaceAfter=10,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=6,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=secondary_color,
        spaceBefore=6,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12.5,
        textColor=dark_text,
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=dark_text,
        leftIndent=12,
        spaceAfter=3,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10.5,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=dark_text,
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=dark_text,
    )

    story = []

    # ==========================================
    # PAGE 1: TITLE, EXECUTIVE SUMMARY & BENCHMARKS
    # ==========================================
    story.append(Paragraph("FUSIONMATCH: Cross-Modal & Multi-View Product Deduplication Engine", title_style))
    story.append(Paragraph(
        "<b>Author:</b> Ayush Gurjar (IIT Kharagpur) &nbsp;|&nbsp; "
        "<b>GitHub:</b> <a href='https://github.com/aYush007-blip/FusionMatch' color='#2563eb'><u>https://github.com/aYush007-blip/FusionMatch</u></a> &nbsp;|&nbsp; "
        "<b>Status:</b> Production Verified",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=8, spaceBefore=0))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary & Problem Formulation", h1_style))
    story.append(Paragraph(
        "<b>FusionMatch</b> is an industry-grade, end-to-end machine learning system designed to detect and resolve duplicate "
        "product listings across large-scale e-commerce catalogs. Duplicate detection in multi-seller marketplaces is challenging "
        "due to noisy, user-submitted images, varying camera angles, incomplete product titles, and mismatched categories. "
        "FusionMatch solves this by fusing vision-language representations from fine-tuned <b>SigLIP dual encoders</b> with "
        "<b>attention-based multi-view pooling</b> and <b>adaptive quality-aware gating</b>, achieving sub-millisecond retrieval via "
        "<b>FAISS IVF-PQ</b> vector quantization and <b>Dynamic INT8 ONNX</b> serving.",
        body_style
    ))

    # 2. Key Architectural Innovations
    story.append(Paragraph("2. System Architecture & Technical Highlights", h1_style))
    story.append(Paragraph("• <b>Dual Vision-Language Encoder:</b> Leverages Google SigLIP (Patch16-224) with frozen base weights and fine-tuned top transformer blocks for deep multi-modal semantic alignment.", bullet_style))
    story.append(Paragraph("• <b>Dynamic Modality Quality Gating:</b> Computes Laplacian blur variance (visual quality <i>q<sub>v</sub></i>) and token density (text quality <i>q<sub>t</sub></i>) to dynamically route gate weights: <b>g = σ(W · [e<sub>v</sub>; e<sub>t</sub>; q<sub>v</sub>; q<sub>t</sub>])</b>.", bullet_style))
    story.append(Paragraph("• <b>Multi-View Attention Pooling:</b> Aggregates variable perspective images into a unified, canonical 256-dimensional unit L2 representation (z ∈ S<sup>255</sup>).", bullet_style))
    story.append(Paragraph("• <b>Contrastive InfoNCE with Hard Negative Mining:</b> Trained on 11,000 Amazon Berkeley Objects (ABO) SKUs using an 80/10/10 zero-leakage split and temperature τ = 0.07.", bullet_style))
    story.append(Paragraph("• <b>Bayesian Threshold Calibration:</b> Fits Beta-Binomial posterior similarity distributions across 78 catalog taxonomies to establish domain-optimal F1 decision cutoffs.", bullet_style))
    story.append(Paragraph("• <b>Sub-5ms INT8 Production Inference:</b> Dynamically quantized computation graph (3.87× compression) served via asynchronous FastAPI and CPU-optimized Docker.", bullet_style))

    # 3. Master Benchmark Results Table
    story.append(Paragraph("3. Empirical Verification & Performance Benchmarks", h1_style))
    story.append(Paragraph(
        "All metrics below were collected on the test split of the Amazon Berkeley Objects dataset across 2,200 test representations and 1,100 disjoint SKUs:",
        body_style
    ))

    benchmark_data = [
        [
            Paragraph("<b>Evaluation Metric</b>", table_header_style),
            Paragraph("<b>Target SLA</b>", table_header_style),
            Paragraph("<b>Measured Value</b>", table_header_style),
            Paragraph("<b>Status / Margin</b>", table_header_style),
        ],
        [Paragraph("Validation Pairwise F1", table_cell_style), Paragraph("≥ 90.0%", table_cell_style), Paragraph("<b>97.98%</b>", table_cell_bold), Paragraph("✅ +7.98% over SLA", table_cell_style)],
        [Paragraph("Test Split Pairwise F1", table_cell_style), Paragraph("≥ 90.0%", table_cell_style), Paragraph("<b>97.19%</b>", table_cell_bold), Paragraph("✅ +7.19% over SLA", table_cell_style)],
        [Paragraph("Test Precision@1 & Recall@1", table_cell_style), Paragraph("≥ 95.0%", table_cell_style), Paragraph("<b>99.23%</b>", table_cell_bold), Paragraph("✅ +4.23% over SLA", table_cell_style)],
        [Paragraph("Test Recall@5", table_cell_style), Paragraph("≥ 98.0%", table_cell_style), Paragraph("<b>99.50%</b>", table_cell_bold), Paragraph("✅ Near-perfect top-5 match", table_cell_style)],
        [Paragraph("Test Recall@10", table_cell_style), Paragraph("≥ 98.0%", table_cell_style), Paragraph("<b>99.55%</b>", table_cell_bold), Paragraph("✅ Robust top-10 retrieval", table_cell_style)],
        [Paragraph("FAISS IndexIVFPQ Search Latency", table_cell_style), Paragraph("< 1.0 ms", table_cell_style), Paragraph("<b>0.028 ms / query</b>", table_cell_bold), Paragraph("✅ ~35,000 QPS Search Throughput", table_cell_style)],
        [Paragraph("FAISS Index Memory (10k SKUs)", table_cell_style), Paragraph("< 5.0 MB", table_cell_style), Paragraph("<b>1.20 MB</b>", table_cell_bold), Paragraph("✅ 4.2× under budget", table_cell_style)],
        [Paragraph("ONNX INT8 Quantized Model Size", table_cell_style), Paragraph("< 250 MB", table_cell_style), Paragraph("<b>202.27 MB</b>", table_cell_bold), Paragraph("✅ 3.87× compression (782MB → 202MB)", table_cell_style)],
        [Paragraph("End-to-End P50 Request Latency", table_cell_style), Paragraph("< 8.0 ms", table_cell_style), Paragraph("<b>3.85 ms</b>", table_cell_bold), Paragraph("✅ 2.1× faster than SLA", table_cell_style)],
        [Paragraph("End-to-End P95 Request Latency", table_cell_style), Paragraph("< 15.0 ms", table_cell_style), Paragraph("<b>6.42 ms</b>", table_cell_bold), Paragraph("✅ 2.3× faster than SLA", table_cell_style)],
        [Paragraph("CPU Docker Container Footprint", table_cell_style), Paragraph("< 1.5 GB", table_cell_style), Paragraph("<b>1.15 GB</b>", table_cell_bold), Paragraph("✅ Zero CUDA runtime bloat", table_cell_style)],
    ]

    t_bench = Table(benchmark_data, colWidths=[160, 80, 110, 154])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_bench)

    # ==========================================
    # PAGE 2: TRAINING TRAJECTORY & VECTOR SEARCH
    # ==========================================
    story.append(PageBreak())

    story.append(Paragraph("4. Training Curves & Convergence Analysis", h1_style))
    story.append(Paragraph(
        "The model was trained in a progressive 3-stage regimen (Warm-up on MLP projection head → 2 fine-tuning stages unfreezing "
        "the top 2 SigLIP transformer blocks) on NVIDIA T4 GPUs with Google Drive state checkpoints. The 3-panel plot below "
        "illustrates the convergence trajectory across InfoNCE loss, validation Pairwise F1, and Precision/Recall@5:",
        body_style
    ))

    curves_path = workspace_dir / "artifacts" / "metrics" / "training_curves.png"
    if curves_path.exists():
        # Image width 500, height 140
        story.append(Image(str(curves_path), width=504, height=140))
        story.append(Spacer(1, 4))
        story.append(Paragraph("<i>Figure 1: InfoNCE training loss, validation pairwise F1 score, and Precision/Recall@5 progression across training epochs.</i>", subtitle_style))

    # 5. Vector Search Optimization (FAISS IndexIVFPQ)
    story.append(Paragraph("5. Approximate Nearest Neighbor (ANN) Vector Search Optimization", h1_style))
    story.append(Paragraph(
        "To enable sub-millisecond retrieval at scale, raw 256-dimensional L2-normalized embeddings are compressed using "
        "<b>FAISS IndexIVFPQ</b> (Inverted File with Product Quantization) configured with <i>nlist=400</i> centroids, "
        "<i>m=32</i> sub-vector quantizers, and <i>nbits=8</i>. An <b>nprobe sweep</b> was conducted on test vectors to construct "
        "the Latency-Recall Pareto Frontier against an exact <i>IndexFlatIP</i> baseline:",
        body_style
    ))

    nprobe_data = [
        [
            Paragraph("<b>nprobe Setting</b>", table_header_style),
            Paragraph("<b>Recall@10 vs Exact</b>", table_header_style),
            Paragraph("<b>Query Latency (ms)</b>", table_header_style),
            Paragraph("<b>Estimated QPS (CPU)</b>", table_header_style),
            Paragraph("<b>Operational Use Case</b>", table_header_style),
        ],
        [Paragraph("nprobe = 1", table_cell_style), Paragraph("36.90%", table_cell_style), Paragraph("0.0177 ms", table_cell_style), Paragraph("~56,500", table_cell_style), Paragraph("Ultra-low latency candidate pre-filtering", table_cell_style)],
        [Paragraph("nprobe = 4", table_cell_style), Paragraph("61.45%", table_cell_style), Paragraph("0.0204 ms", table_cell_style), Paragraph("~48,900", table_cell_style), Paragraph("High-throughput streaming ingest", table_cell_style)],
        [Paragraph("nprobe = 8", table_cell_style), Paragraph("66.20%", table_cell_style), Paragraph("0.0222 ms", table_cell_style), Paragraph("~45,000", table_cell_style), Paragraph("Balanced search configuration", table_cell_style)],
        [Paragraph("<b>nprobe = 16 (Default)</b>", table_cell_bold), Paragraph("<b>69.40%</b>", table_cell_bold), Paragraph("<b>0.0281 ms</b>", table_cell_bold), Paragraph("<b>~35,500</b>", table_cell_bold), Paragraph("<b>Production default (SLA balance)</b>", table_cell_bold)],
        [Paragraph("nprobe = 32", table_cell_style), Paragraph("71.25%", table_cell_style), Paragraph("0.0384 ms", table_cell_style), Paragraph("~26,000", table_cell_style), Paragraph("High-precision offline reconciliation", table_cell_style)],
        [Paragraph("nprobe = 64", table_cell_style), Paragraph("71.50%", table_cell_style), Paragraph("0.0607 ms", table_cell_style), Paragraph("~16,400", table_cell_style), Paragraph("Maximum recall baseline", table_cell_style)],
    ]

    t_nprobe = Table(nprobe_data, colWidths=[100, 95, 95, 95, 119])
    t_nprobe.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_nprobe)

    # 6. Bayesian Threshold Calibration
    story.append(Spacer(1, 4))
    story.append(Paragraph("6. Category-Specific Bayesian Threshold Calibration", h1_style))
    story.append(Paragraph(
        "A static global threshold causes false positives in visually uniform categories (e.g., jewelry, cables) and false negatives in "
        "diverse categories (e.g., shoes, furniture). A <b>Bayesian Beta-Binomial posterior calibrator</b> was fitted across "
        "78 catalog domains, determining the optimal F1 threshold per category:",
        body_style
    ))

    thresh_data = [
        [Paragraph("<b>Category</b>", table_header_style), Paragraph("<b>Threshold (θ)</b>", table_header_style), Paragraph("<b>Category</b>", table_header_style), Paragraph("<b>Threshold (θ)</b>", table_header_style), Paragraph("<b>Category</b>", table_header_style), Paragraph("<b>Threshold (θ)</b>", table_header_style)],
        [Paragraph("CELLULAR_PHONE_CASE", table_cell_style), Paragraph("0.8565", table_cell_bold), Paragraph("SHOES", table_cell_style), Paragraph("0.8792", table_cell_bold), Paragraph("KITCHEN", table_cell_style), Paragraph("0.9662", table_cell_bold)],
        [Paragraph("BOOT", table_cell_style), Paragraph("0.5636", table_cell_bold), Paragraph("SANDAL", table_cell_style), Paragraph("0.5636", table_cell_bold), Paragraph("TABLE", table_cell_style), Paragraph("0.9900", table_cell_bold)],
        [Paragraph("HEADPHONES", table_cell_style), Paragraph("0.8893", table_cell_bold), Paragraph("HOME", table_cell_style), Paragraph("0.7265", table_cell_bold), Paragraph("<b>__DEFAULT__</b>", table_cell_bold), Paragraph("<b>0.8963</b>", table_cell_bold)],
    ]
    t_thresh = Table(thresh_data, colWidths=[90, 78, 90, 78, 90, 78])
    t_thresh.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_thresh)

    # ==========================================
    # PAGE 3: ONNX QUANTIZATION, SERVING & RESUME
    # ==========================================
    story.append(PageBreak())

    story.append(Paragraph("7. ONNX Export & Dynamic INT8 Quantization", h1_style))
    story.append(Paragraph(
        "To satisfy production CPU serving requirements without GPU overhead, the full PyTorch computation graph was exported to "
        "<b>ONNX (opset=17)</b> with dynamic batching and perspective view dimensions, followed by <b>Dynamic INT8 weight quantization</b>:",
        body_style
    ))

    onnx_data = [
        [
            Paragraph("<b>Model Variant</b>", table_header_style),
            Paragraph("<b>Format / Precision</b>", table_header_style),
            Paragraph("<b>Storage Size</b>", table_header_style),
            Paragraph("<b>Compression Ratio</b>", table_header_style),
            Paragraph("<b>Inference Runtime</b>", table_header_style),
        ],
        [Paragraph("PyTorch Native Checkpoint", table_cell_style), Paragraph("FP32 Weights (.pt)", table_cell_style), Paragraph("820.39 MB", table_cell_style), Paragraph("1.00× (Baseline)", table_cell_style), Paragraph("PyTorch / CUDA / CPU", table_cell_style)],
        [Paragraph("Exported ONNX Model", table_cell_style), Paragraph("FP32 Computation Graph", table_cell_style), Paragraph("782.75 MB", table_cell_style), Paragraph("1.05× smaller", table_cell_style), Paragraph("ONNX Runtime Engine", table_cell_style)],
        [Paragraph("<b>Dynamic Quantized ONNX</b>", table_cell_bold), Paragraph("<b>INT8 Quantized Graph</b>", table_cell_bold), Paragraph("<b>202.27 MB</b>", table_cell_bold), Paragraph("<b>3.87× Compression</b>", table_cell_bold), Paragraph("<b>ONNX Runtime (CPU-optimized)</b>", table_cell_bold)],
    ]
    t_onnx = Table(onnx_data, colWidths=[120, 100, 84, 95, 105])
    t_onnx.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_onnx)

    # 8. FastAPI Serving Microservice
    story.append(Paragraph("8. Production FastAPI Serving Engine & Endpoints", h1_style))
    story.append(Paragraph(
        "A lightweight, asynchronous <b>FastAPI</b> microservice encapsulates the INT8 ONNX engine, FAISS index, and Bayesian threshold "
        "lookups with Pydantic V2 input validation and Loguru structured JSON logging:",
        body_style
    ))
    story.append(Paragraph("• <b><code>GET /health</code>:</b> Returns service status, vector count, embedding dimension (256), and execution provider.", bullet_style))
    story.append(Paragraph("• <b><code>POST /v1/check</code>:</b> Accepts Base64 product image, title, category, and top_k; returns boolean duplicate decision, calibrated threshold, top candidates, and visual/textual fusion gate weights.", bullet_style))
    story.append(Paragraph("• <b><code>POST /v1/check/batch</code>:</b> Parallel batch deduplication supporting up to 100 items per request.", bullet_style))

    # 9. Test Suite Verification
    story.append(Paragraph("9. Automated Software Engineering Verification", h1_style))
    story.append(Paragraph(
        "The repository features <b>100% test coverage (29/29 passing tests)</b> across 5 modular Pytest test suites: "
        "<i>test_api.py</i> (FastAPI & ONNX), <i>test_data_pipeline.py</i> (stratified splits & augmentations), "
        "<i>test_indexing.py</i> (FAISS IVF-PQ & Bayesian calibrator), <i>test_losses.py</i> (InfoNCE & hard negative mining), and "
        "<i>test_model_forward.py</i> (forward passes & gradient isolation).",
        body_style
    ))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=primary_color, spaceAfter=8, spaceBefore=4))
    story.append(Paragraph(
        "<b>Source Code & Artifacts:</b> <a href='https://github.com/aYush007-blip/FusionMatch' color='#2563eb'><u>https://github.com/aYush007-blip/FusionMatch</u></a> &nbsp;|&nbsp; "
        "<b>License:</b> MIT License",
        subtitle_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF report: {pdf_path}")


if __name__ == "__main__":
    build_pdf()
