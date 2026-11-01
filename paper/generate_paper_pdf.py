"""
Generate the publication PDF for:
  "AI-Driven Superconductor Discovery on Unified-Memory APUs: Equivariant Crystal
   Diffusion, Zero-Copy MACE Relaxation, and First-Principles DFT Verification"
  (Thomas Nauheimer, November 2026).
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def build_pdf(output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="AI-Driven Superconductor Discovery on Unified-Memory APUs",
        author="Thomas Nauheimer",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Heading1"],
        fontName="Times-Bold",
        fontSize=14.5,
        leading=18.5,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    author_style = ParagraphStyle(
        "PaperAuthor",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    abstract_style = ParagraphStyle(
        "PaperAbstract",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=9.5,
        leading=13,
        alignment=TA_JUSTIFY,
        leftIndent=14,
        rightIndent=14,
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Times-Bold",
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "PaperBody",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=10,
        leading=13.5,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    eq_style = ParagraphStyle(
        "PaperEq",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=6,
    )

    story = []

    story.append(
        Paragraph(
            "AI-Driven Superconductor Discovery on Unified-Memory APUs: Equivariant Crystal Diffusion, Zero-Copy MACE Relaxation, and First-Principles DFT Verification",
            title_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Thomas Nauheimer</b><br/>Independent Researcher (nauheimer.t@gmail.com)<br/>November 2026",
            author_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Abstract.</b> We present a closed-loop computational pipeline for superconductor candidate discovery designed for unified-memory consumer workstations (AMD Ryzen AI Max+ 395 / Strix Halo APU with Radeon 8060S GPU and 128 GB unified LPDDR5X). The architecture couples four stages: (1) compositional Magpie feature screening trained on 21,263 entries of the UCI SuperCon dataset (MAE = 5.36 K, R<sup>2</sup> = 0.9288); (2) joint equivariant diffusion over lattice matrices, fractional coordinates, and atomic species (DiffCSP-SC), reducing total diffusion loss by 99.01% (152.00 to 1.50) over 1,000 epochs; (3) native C++/HIP zero-copy pre-relaxation (hipHostMallocMapped and torch::from_blob) with a 1.20 &Aring; hard-core collision filter that eliminates PCIe host-device transfer overhead (144.7 ms/step); and (4) first-principles plane-wave Density Functional Theory (DFT) verification using Quantum ESPRESSO v7.3 (PBE-GGA, PAW pseudopotentials, 50 Ry / 400 Ry cutoffs). Across 50 diffusion-generated candidates, the top five chemically distinct structures (CeIn<sub>3</sub>, Sb<sub>2</sub>Te<sub>3</sub>, BaBiTe, AuPd<sub>11</sub>, and CuIn<sub>2</sub>K) achieved 100% SCF convergence and 100% metallic character. Notably, BaBiTe emerges as a pressure-stabilized (~18.9 GPa) metallic ternary candidate (E<sub>tot</sub> = -1704.43 Ry, E<sub>F</sub> = 2.98 eV, residual forces 0.021 Ry/Bohr).",
            abstract_style,
        )
    )

    story.append(Paragraph("1. Unified-Memory Zero-Copy Architecture and Diffusion Screening", h1_style))
    story.append(
        Paragraph(
            "Standard hybrid ML/ab-initio workflows repeatedly copy atomic coordinates and forces across the PCIe bus between CPU RAM and discrete GPU VRAM. On unified-memory architectures (AMD Ryzen AI Max+ 395, GFX1151, 128 GB shared LPDDR5X), allocating atomic buffers via hipHostMalloc(..., hipHostMallocMapped) and wrapping them in-place with torch::from_blob yields identical virtual pointers (cpu_ptr == gpu_ptr), executing 54-atom TorchScript MACE relaxation steps in 144.7 ms/step with zero PCIe transfer traffic. Prior to neural force evaluation, a minimum-image core-repulsion filter enforces d<sub>ij</sub> &ge; 1.20 &Aring; to prevent NaN-gradient cascades on raw diffusion samples.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Our stage-1 compositional screener (81 Magpie features, 500 gradient-boosted trees) achieves MAE = 5.36 K and R<sup>2</sup> = 0.9288 across 21,263 UCI SuperCon entries. Stage-2 joint equivariant diffusion (DiffCSP-SC, 1,000 epochs, FP16 on Radeon 8060S) converges as follows:",
            body_style,
        )
    )

    t1_data = [
        ["Diffusion Loss Component", "Epoch 1", "Epoch 999", "Relative Reduction"],
        ["Total Diffusion Loss", "152.00", "1.50", "99.01%"],
        ["Lattice Matrix Loss", "127.00", "0.58", "99.54%"],
        ["Fractional Coordinate Loss", "1.36", "0.73", "46.32%"],
        ["Atom-Type Classification Loss", "1.20", "0.18", "85.00%"],
    ]
    t1 = Table(t1_data, colWidths=[62 * mm, 30 * mm, 30 * mm, 38 * mm])
    t1.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t1)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. First-Principles Quantum ESPRESSO v7.3 Verification", h1_style))
    story.append(
        Paragraph(
            "Top diffusion candidates were verified with plane-wave DFT in Quantum ESPRESSO v7.3 (PBE-GGA, PAW pslibrary pseudopotentials, E<sub>cut</sub><sup>wfc</sup> = 50 Ry, E<sub>cut</sub><sup>&rho;</sup> = 400 Ry, 4&times;4&times;2 Monkhorst–Pack k-mesh, Marzari–Vanderbilt cold smearing 0.02 Ry, SCF threshold 10<sup>-6</sup> Ry):",
            body_style,
        )
    )

    t2_data = [
        ["Candidate", "Total Energy (Ry)", "SCF Iter.", "Pressure (kbar)", "E_Fermi (eV)", "Forces (Ry/Bohr)", "Electrons"],
        ["CeIn3", "-1703.79", "16", "-27.8", "2.34", "0.024", "50"],
        ["Sb2Te3", "-1782.81", "18", "-33.1", "1.26", "0.073", "28"],
        ["BaBiTe", "-1704.43", "24", "+189.4", "2.98", "0.021", "31"],
        ["AuPd11", "-4415.78", "22", "-20.8", "6.15", "0.676", "121"],
        ["CuIn2K", "-1147.38", "15", "-19.7", "1.16", "0.036", "46"],
    ]
    t2 = Table(
        t2_data,
        colWidths=[22 * mm, 29 * mm, 18 * mm, 26 * mm, 24 * mm, 28 * mm, 19 * mm],
    )
    t2.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTNAME", (0, 3), (-1, 3), "Times-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "<b>Candidate Assessment &amp; Scientific Scope.</b> The independent generation of CeIn<sub>3</sub> (known heavy-fermion superconductor, T<sub>c</sub> &approx; 0.2 K) and Sb<sub>2</sub>Te<sub>3</sub> validates the learned chemical manifold. BaBiTe converges in 24 SCF iterations with the smallest residual forces of the cohort (0.021 Ry/Bohr) and metallic Fermi energy E<sub>F</sub> = 2.98 eV at 189.4 kbar (~18.9 GPa). Under the Allen–Dynes modified McMillan equation:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "T<sub>c</sub> = (f<sub>1</sub> f<sub>2</sub> &omega;<sub>log</sub> / 1.20) exp[ -1.04 (1 + &lambda;) / (&lambda; - &mu;* (1 + 0.62 &lambda;)) ],",
            eq_style,
        )
    )
    story.append(
        Paragraph(
            "a moderate electron–phonon coupling regime (&lambda; &approx; 0.95, &omega;<sub>log</sub> &approx; 210 K, &mu;* = 0.10) corresponds to T<sub>c</sub> &approx; 14.3 K. We explicitly note that while static SCF convergence and low Hellmann–Feynman forces confirm electronic stability and metallicity, full DFPT phonon dispersion &omega;(q) across the Brillouin zone, Eliashberg &alpha;<sup>2</sup>F(&omega;) coupling via EPW, and ternary Ba–Bi–Te convex-hull enthalpy relaxation remain necessary prior to experimental synthesis.",
            body_style,
        )
    )

    doc.build(story)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "SuperCon_Zero_Copy_Discovery.pdf")
    build_pdf(out)
    print(f"Generated PDF: {out} ({os.path.getsize(out)} bytes)")
