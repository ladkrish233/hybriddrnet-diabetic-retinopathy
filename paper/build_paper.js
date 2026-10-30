const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, BorderStyle, PageBreak,
} = require("docx");

const FILL_COLOR = "C00000"; // red text for [FILL: ...] placeholders

function fillRun(text) {
  return new TextRun({ text, color: FILL_COLOR, italics: true });
}

function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
}

function heading(text, level) {
  return new Paragraph({ text, heading: level, spacing: { before: 300, after: 150 } });
}

function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}

function makeCell(text, opts = {}) {
  const { bold = false, width = 1500, shade = null } = opts;
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text, bold })] })],
  });
}

// --- Related-work table data (extracted verbatim from
//     Diabetic_Retinopathy_Research_Comparison.docx) -----------------------
const relatedWork = [
  ["Paper & Authors", "Core Model / Architecture", "Dataset(s)", "Key Results"],
  ["P1 Shah et al. (2024)", "YOLOv8 Small + SVM", "Kaggle DR", "SVM Acc 84%; YOLO lesion detection 78%"],
  ["P2 Bhattacharjee et al. (2021)", "Random Forest (vs SVM, Naive Bayes)", "Kaggle DR (resized)", "Acc 76.5%; Sens 77.2%; Spec 93.3%"],
  ["P3 Mateen et al. (2025)", "Narrative review: lesion features, multimodal, XAI", "DRIVE, Messidor, IDRID, STARE, etc.", "Shift from handcrafted ML to transformer/YOLO-based XAI"],
  ["P4 Yang et al. (2025)", "Systematic review: oculomics for systemic disease", "38 studies (UK Biobank, SEED, EyePACS)", "Retinal AI AUC 0.676-0.971 for systemic vascular complications"],
  ["P5 Zaylaa & Kourtian (2025)", "8 pretrained DNNs (ResNet-50, GoogleNet, SqueezeNet, etc.)", "APTOS 2019 (3,662 images)", "ResNet-50/GoogleNet Acc 93.56%; GoogleNet Sens 99.44%"],
  ["P6 Jabbar et al. (2024)", "Fine-tuned VGG16 (transfer learning)", "Kaggle EyePACS, Messidor, IDRID", "Acc 97.60%; Sens 98.3%; Spec 98.4%"],
  ["P7 Bi et al. (2025)", "Meta-analysis of DL models (CNN, Inception, ResNet, DenseNet)", "47 studies (10 meta-analyzed, 188,268 scans)", "Pooled OR screening completion 1.88; diagnostic accuracy 1.33"],
  ["P8 Sushith et al. (2025)", "Temporal-Aware Hybrid DL (multi-scale CNN + LSTM + attention)", "DRIVE, Kaggle DR, EyePACS", "DRIVE Acc 97.50%; Kaggle 94.04%; EyePACS 96.90%"],
  ["P9 Bhoopalan et al. (2025)", "Task-Optimized ViT (TOVIT) + pruning + 8-bit quantization", "EyePACS, Messidor-2, APTOS 2019, DIARETDB1", "Acc 99%; F1 >93%; Raspberry Pi 4: 8 FPS, 120ms"],
  ["P10 Sreekumar & Vimina (2026)", "ViT + Wasserstein DCGAN + GPCA optimizer", "APTOS-2019, DRD", "APTOS Acc 98.20%; DRD Acc 98.62%; Spec 98.82%"],
  ["P11 Vijayalakshmi et al. (2025)", "Multi-View Cross-Attention ViT (MVCAVIT) + CNN + PSO fusion", "DRTID (3,100 two-field sets)", "Acc 98.9%; F1 98.8%; Lesion IoU 88.7%"],
  ["P12 Yi et al. (2021)", "RA-EfficientNet (EfficientNetB0 + residual attention)", "APTOS 2019, EyePACS", "2-grade Acc 98.36% (kappa 96.72%); 5-grade Acc 93.55%"],
  ["P13 Zhang et al. (2024)", "Multi-level EfficientNet-B0 & ResNet-18 ensembles", "UWF4DR, DeepDRiD", "Quality AUROC 0.9051; RDR AUROC 0.9811; DME AUROC 0.9820"],
  ["P14 Dai et al. (2021)", "DeepDR: transfer-learning multi-task (ResNet + Mask R-CNN)", "SIM Cohort (666,383 images), CNDCS, NDSP, EyePACS", "Grading AUC 0.955; lesion AUCs 0.90-0.97"],
  ["P15 Tymchenko et al. (2020)", "Multi-task 3-head CNN (classification + regression + ordinal)", "APTOS 2019, EyePACS 2015, IDRID, Messidor", "APTOS QWK 0.9255; binary Acc 99.3%"],
  ["P16 Majumder & Kehtarnavaz (2021)", "MSEDenseNet (SE-DenseNet) & MXception", "APTOS 2019, EyePACS", "APTOS QWK 0.90 (Acc 86%); EyePACS QWK 0.88 (Acc 82%)"],
  ["P17 Shaban et al. (2020)", "Modified VGG-19 (18 conv + 3 FC layers)", "APTOS 2019", "3-class Acc 88-89%; QWK 0.91-0.92"],
  ["P18 Yadav (CHARUSAT)", "DeepReti (DenseNet + residual block)", "Clinical fundus dataset", "Acc 95.34%; Sens 98.83%; Spec 93.82%; Kappa 90.34"],
];

function buildRelatedWorkTable() {
  const colWidths = [2400, 3200, 2400, 3000];
  const rows = relatedWork.map((r, i) =>
    new TableRow({
      children: r.map((text, j) => makeCell(text, { width: colWidths[j], bold: i === 0, shade: i === 0 ? "D9E2F3" : null })),
    })
  );
  return new Table({ width: { size: 11000, type: WidthType.DXA }, columnWidths: colWidths, rows });
}

function buildResultsTable() {
  const header = ["Metric", "Value"];
  const metricRows = [
    "Overall Accuracy",
    "Quadratic Weighted Kappa (QWK)",
    "Macro Precision",
    "Macro Recall (Sensitivity)",
    "Macro Specificity",
    "Macro F1-score",
    "Macro AUROC (one-vs-rest)",
  ];
  const colWidths = [5500, 5500];
  const rows = [
    new TableRow({ children: header.map((t) => makeCell(t, { width: 5500, bold: true, shade: "D9E2F3" })) }),
    ...metricRows.map((m) =>
      new TableRow({
        children: [
          makeCell(m, { width: 5500 }),
          new TableCell({
            width: { size: 5500, type: WidthType.DXA },
            children: [new Paragraph({ children: [fillRun("[FILL from outputs/results_summary.json]")] })],
          }),
        ],
      })
    ),
  ];
  return new Table({ width: { size: 11000, type: WidthType.DXA }, columnWidths: colWidths, rows });
}

function buildExtendedComparisonTable() {
  const colWidths = [2400, 3200, 2400, 3000];
  const header = new TableRow({
    children: relatedWork[0].map((t, j) => makeCell(t, { width: colWidths[j], bold: true, shade: "D9E2F3" })),
  });
  const existingRows = relatedWork.slice(1).map((r, i) =>
    new TableRow({ children: r.map((text, j) => makeCell(text, { width: colWidths[j] })) })
  );
  const oursRow = new TableRow({
    children: [
      makeCell("Proposed: HybridDRNet (this work)", { width: colWidths[0], bold: true, shade: "FCE4D6" }),
      makeCell("EfficientNetV2-S + Transformer fusion + dual classification/ordinal head", { width: colWidths[1], shade: "FCE4D6" }),
      makeCell("APTOS 2019", { width: colWidths[2], shade: "FCE4D6" }),
      new TableCell({
        width: { size: colWidths[3], type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: "FCE4D6" },
        children: [new Paragraph({ children: [fillRun("[FILL: Acc __%; QWK __; AUROC __]")] })],
      }),
    ],
  });
  return new Table({ width: { size: 11000, type: WidthType.DXA }, columnWidths: colWidths, rows: [header, ...existingRows, oursRow] });
}

function figurePlaceholder(caption) {
  return [
    new Paragraph({
      border: { top: { style: BorderStyle.SINGLE, size: 6, color: "999999" }, bottom: { style: BorderStyle.SINGLE, size: 6, color: "999999" }, left: { style: BorderStyle.SINGLE, size: 6, color: "999999" }, right: { style: BorderStyle.SINGLE, size: 6, color: "999999" } },
      spacing: { before: 200, after: 60 },
      children: [fillRun(`[Insert ${caption} here — generated in outputs/ by the notebook]`)],
      alignment: AlignmentType.CENTER,
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 20 })],
    }),
  ];
}

const references = [
  "Bhattacharjee, S., et al. (2021). Random forest-based diabetic retinopathy classification with handcrafted features.",
  "Bhoopalan, K., et al. (2025). TOVIT: Task-Optimized Vision Transformer for edge-deployed diabetic retinopathy grading.",
  "Bi, Y., et al. (2025). Deep learning for diabetic retinopathy screening: A systematic review and meta-analysis.",
  "Dai, L., et al. (2021). A deep learning system for detecting diabetic retinopathy across the disease spectrum (DeepDR). Nature Communications.",
  "Jabbar, A., et al. (2024). Transfer learning with VGG16 for diabetic retinopathy detection.",
  "Majumder, S., & Kehtarnavaz, N. (2021). Multitasking deep learning model for detection of five stages of diabetic retinopathy.",
  "Mateen, M., et al. (2025). Explainable AI for diabetic retinopathy: A narrative review.",
  "Shaban, M., et al. (2020). A convolutional neural network for the screening and staging of diabetic retinopathy. PLOS ONE.",
  "Shah, P., et al. (2024). YOLOv8 and SVM-based lesion detection for diabetic retinopathy.",
  "Sreekumar, A., & Vimina, E. R. (2026). ViT-WDCGAN: Vision Transformer with Wasserstein GAN augmentation for diabetic retinopathy.",
  "Sushith, M., et al. (2025). Temporal-aware hybrid deep learning for diabetic retinopathy progression modeling.",
  "Tymchenko, B., Marchenko, P., & Spodarets, D. (2020). Deep learning approach to diabetic retinopathy detection. ICPRAM.",
  "Vijayalakshmi, R., et al. (2025). MVCAVIT: Multi-view cross-attention Vision Transformer for diabetic retinopathy grading.",
  "Yadav, et al. Deep Reti: DenseNet-based diabetic retinopathy detection (CHARUSAT applied clinical project).",
  "Yang, D., et al. (2025). Retinal oculomics for systemic disease prediction: A systematic review.",
  "Yi, S.-L., et al. (2021). RA-EfficientNet: Residual attention EfficientNet for diabetic retinopathy grading.",
  "Zaylaa, A., & Kourtian, S. (2025). Comparative evaluation of pretrained deep neural networks for diabetic retinopathy classification.",
  "Zhang, Y., et al. (2024). Multi-level EfficientNet-ResNet ensembles for ultra-widefield diabetic retinopathy screening.",
];

const doc = new Document({
  sections: [
    {
      properties: {},
      children: [
        new Paragraph({ text: "Deep Retinal Grading with a Hybrid CNN-Transformer Network:", heading: HeadingLevel.TITLE, spacing: { after: 80 } }),
        new Paragraph({ text: "Closing the Gaps in Diabetic Retinopathy Classification Literature", heading: HeadingLevel.TITLE, spacing: { after: 300 } }),
        new Paragraph({ children: [fillRun("[FILL: Author name(s)]")], alignment: AlignmentType.CENTER, spacing: { after: 40 } }),
        new Paragraph({ children: [fillRun("[FILL: Affiliation]")], alignment: AlignmentType.CENTER, spacing: { after: 40 } }),
        new Paragraph({ children: [fillRun("[FILL: Contact email]")], alignment: AlignmentType.CENTER, spacing: { after: 400 } }),

        heading("Abstract", HeadingLevel.HEADING_1),
        new Paragraph({
          children: [fillRun(
            "[FILL: 150-250 words. Summarize the problem (automated DR grading), the gap " +
            "(no reviewed work combines a CNN-Transformer hybrid, dual classification/ordinal " +
            "head, full metric reporting, and explainability together), the proposed " +
            "HybridDRNet architecture, the dataset (APTOS 2019), and headline results — " +
            "insert your Accuracy/QWK/AUROC numbers from outputs/results_summary.json once " +
            "training is complete.]"
          )],
        }),

        heading("1. Introduction", HeadingLevel.HEADING_1),
        p("Diabetic retinopathy (DR) is a leading cause of preventable blindness in working-age adults, and automated fundus-image grading has become one of the most active applications of deep learning in ophthalmology. A large and fast-growing body of work has pushed reported accuracy above 98-99% using increasingly specialized architectures — Vision Transformers, GAN-based augmentation, multi-view fusion, and edge-optimized quantization."),
        p("However, comparing these results directly is difficult: different papers report different subsets of metrics (accuracy, sensitivity/specificity, Quadratic Weighted Kappa, AUROC), use different datasets, and rarely combine their individually strong ideas into a single architecture. Table 1 summarizes 18 representative papers spanning classical machine learning, CNN transfer learning, Vision Transformers, GAN-augmented training, multi-task ordinal regression, and large-scale clinical deployments."),
        p("This work proposes HybridDRNet, an architecture that combines the complementary strengths identified across that literature — CNN local feature extraction, Transformer global context, multi-task ordinal-aware supervision, and built-in explainability — and evaluates it on the same benchmark (APTOS 2019) used by the majority of the top-performing reviewed papers, reporting every metric used anywhere in the comparison so results are directly comparable."),

        heading("2. Related Work", HeadingLevel.HEADING_1),
        p("Table 1 reproduces the comparison table compiled across 18 papers on automated diabetic retinopathy classification, covering study type, core architecture, preprocessing, datasets, and key reported results."),
        buildRelatedWorkTable(),
        new Paragraph({ spacing: { before: 100, after: 300 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Table 1. Summary of reviewed diabetic retinopathy classification literature.", italics: true, size: 20 })] }),

        heading("3. Gaps Identified in the Literature", HeadingLevel.HEADING_1),
        bullet("No reviewed paper combines a CNN + Transformer hybrid with a dual classification/ordinal-regression head while reporting QWK, AUROC, and a full confusion matrix together — most report only one or two of these."),
        bullet("Explainability (XAI) is explicitly flagged as an open need by the narrative review (Paper 3), yet only a minority of empirical papers include any visual explanation of predictions."),
        bullet("Class-imbalance handling via a dedicated GAN (Paper 10) is effective but computationally expensive; no reviewed paper evaluates a lighter-weight alternative (focal loss + weighted sampling) at comparable accuracy."),
        bullet("Ordinal-aware training (Paper 15) achieves the best QWK in the table (0.9255) but uses a pure CNN backbone; no reviewed paper pairs ordinal-aware multi-task supervision with a Transformer-fused backbone."),

        heading("4. Proposed Methodology", HeadingLevel.HEADING_1),
        heading("4.1 Dataset", HeadingLevel.HEADING_2),
        p("APTOS 2019 Blindness Detection (3,662 labeled fundus images, 5-class DR severity: No DR, Mild, Moderate, Severe, Proliferative DR), the dataset used by 6 of the 18 reviewed papers (Papers 5, 9, 10, 12, 15, 16, 17), chosen specifically so results are directly comparable to the strongest results in Table 1."),
        heading("4.2 Preprocessing", HeadingLevel.HEADING_2),
        p("Each image is circle-cropped to the fundus boundary, CLAHE-equalized on the luminance channel, and background-subtracted via weighted Gaussian blurring (the standard 'Ben Graham' preprocessing used across top APTOS/EyePACS solutions and mirrored, in some form, by Papers 2, 5, 6, 8, 9, 10, 12, 13, and 14)."),
        heading("4.3 Architecture: HybridDRNet", HeadingLevel.HEADING_2),
        p("An EfficientNetV2-S backbone extracts local lesion-level features (microaneurysms, hemorrhages, exudates). The final feature map is projected and flattened into spatial tokens, prepended with a learnable CLS token, and passed through a 4-layer Transformer encoder (8 attention heads) to model global spatial context such as vessel layout and optic-disc position. The normalized CLS token feeds two heads: a 5-way softmax classification head and a scalar ordinal-regression head, trained jointly."),
        ...figurePlaceholder("Figure 1. HybridDRNet architecture diagram (backbone -> tokenization -> Transformer fusion -> dual head)."),
        heading("4.4 Loss Function", HeadingLevel.HEADING_2),
        p("A combined loss, L = alpha * FocalLoss(classification) + (1 - alpha) * MSE(ordinal regression), with alpha = 0.6. Focal loss (gamma = 2) with inverse-frequency class weights down-weights the majority 'No DR' class; the ordinal-regression term directly targets QWK, since DR severity is an ordinal scale and cross-entropy alone ignores the ordering between classes."),
        heading("4.5 Training Setup", HeadingLevel.HEADING_2),
        p("AdamW optimizer, OneCycle learning-rate schedule, mixed-precision training, gradient clipping (max norm 5.0), a WeightedRandomSampler for class balance, and early stopping on validation QWK (patience 6 epochs). Test-time augmentation (identity + horizontal + vertical flip, averaged) is applied at evaluation and inference time. Full implementation details are in dr_model.py and DR_Training_Colab.ipynb."),

        heading("5. Experimental Results", HeadingLevel.HEADING_1),
        p("Results below are computed with test-time augmentation on the held-out validation split (15% of APTOS 2019, stratified by class)."),
        buildResultsTable(),
        new Paragraph({ text: "", spacing: { after: 200 } }),
        heading("5.1 Confusion Matrix", HeadingLevel.HEADING_2),
        ...figurePlaceholder("Figure 2. Confusion matrix (outputs/confusion_matrix.png)."),
        heading("5.2 ROC Curves", HeadingLevel.HEADING_2),
        ...figurePlaceholder("Figure 3. One-vs-rest ROC curves per DR grade (outputs/roc_curves.png)."),
        heading("5.3 Comparison Against Reviewed Literature", HeadingLevel.HEADING_2),
        p("Table 2 extends the literature comparison in Table 1 with the proposed model's results."),
        buildExtendedComparisonTable(),
        new Paragraph({ text: "", spacing: { after: 100 } }),
        ...figurePlaceholder("Figure 4. Accuracy and QWK bar charts against the reviewed literature (outputs/comparison_vs_literature.png, outputs/qwk_comparison_vs_literature.png)."),

        heading("6. Explainability", HeadingLevel.HEADING_1),
        p("Grad-CAM is computed on the CNN feature map immediately preceding Transformer fusion, so explanations reflect the same local features the classifier uses regardless of the downstream attention mechanism. This directly addresses the explainability gap flagged by Paper 3."),
        ...figurePlaceholder("Figure 5. Grad-CAM overlays for representative predictions across all five severity grades (outputs/gradcam_examples.png)."),

        heading("7. Discussion", HeadingLevel.HEADING_1),
        new Paragraph({ children: [fillRun(
          "[FILL: Discuss where HybridDRNet lands relative to Table 2 — which papers it " +
          "matches or exceeds, which per-class grades remain hardest (typically 'Mild' vs " +
          "'Moderate' due to visual similarity), and what the confusion matrix reveals about " +
          "failure modes. Be honest about limitations: single-dataset validation, no external " +
          "test set, no clinical validation.]"
        )] }),

        heading("8. Conclusion and Future Work", HeadingLevel.HEADING_1),
        p("This work introduced HybridDRNet, a CNN-Transformer hybrid with a dual classification/ordinal-regression head, evaluated against the same benchmark used by the strongest papers in a comprehensive 18-paper literature comparison, and reporting every metric used across that literature together with built-in Grad-CAM explainability."),
        bullet("Future work: cross-dataset evaluation (EyePACS, Messidor, IDRID) to test generalization, matching Paper 6 and Paper 8's multi-dataset validation."),
        bullet("Future work: 8-bit quantization and edge deployment benchmarking (Raspberry Pi/Jetson), matching Paper 9's edge-deployment results."),
        bullet("Future work: 5-fold cross-validation with checkpoint ensembling, following Paper 13's ensemble approach."),

        heading("References", HeadingLevel.HEADING_1),
        ...references.map((r) => new Paragraph({ text: r, spacing: { after: 100 }, indent: { left: 360, hanging: 360 } })),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  require("fs").writeFileSync(__dirname + "/Research_Paper_Draft.docx", buffer);
  console.log("Wrote Research_Paper_Draft.docx");
});
