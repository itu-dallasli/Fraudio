// Build the technical presentation for Voice Spoof Detection.
// Run: node build_presentation.js
// Output: Voice_Spoof_Detection_Presentation.pptx

const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3" x 7.5"
pres.author = "Fraudio team";
pres.title = "Voice Spoof Detection — Technical Overview";

// Midnight Executive palette + accents
const NAVY = "1E2761";
const ICE  = "CADCFC";
const WHITE = "FFFFFF";
const GOLD = "FFB627";
const DARK = "1A1A1A";
const MUTED = "64748B";
const LIGHT_BG = "F7F9FC";
const RED = "B85042";
const GREEN = "2C5F2D";

const HEADER_FONT = "Georgia";
const BODY_FONT = "Calibri";

const SLIDE_W = 13.333;
const SLIDE_H = 7.5;

// ---------- helpers ---------- //

function bg(slide, color) { slide.background = { color }; }

function addTopBar(slide, color = NAVY, height = 0.18) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: SLIDE_W, h: height, fill: { color }, line: { color, width: 0 },
  });
}

function addBottomBar(slide, text = "Voice Spoof Detection · WavLM + Wav2Vec2 · ITU 2026", color = NAVY) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: SLIDE_H - 0.35, w: SLIDE_W, h: 0.35, fill: { color }, line: { color, width: 0 },
  });
  slide.addText(text, {
    x: 0.6, y: SLIDE_H - 0.34, w: SLIDE_W - 1.2, h: 0.33,
    fontFace: BODY_FONT, fontSize: 9, color: ICE, valign: "middle", margin: 0,
  });
  slide.addText("github.com/itu-dallasli/Fraudio", {
    x: SLIDE_W - 3.6, y: SLIDE_H - 0.34, w: 3.0, h: 0.33,
    fontFace: BODY_FONT, fontSize: 9, color: ICE, valign: "middle", align: "right", margin: 0,
  });
}

function addSlideTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.6, y: 0.42, w: SLIDE_W - 1.2, h: 0.85,
    fontFace: HEADER_FONT, fontSize: 36, bold: true, color: NAVY, margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.6, y: 1.25, w: SLIDE_W - 1.2, h: 0.45,
      fontFace: BODY_FONT, fontSize: 16, color: MUTED, margin: 0,
    });
  }
}

function numberCircle(slide, n, x, y, d = 0.55, fill = NAVY, fontColor = WHITE) {
  slide.addShape(pres.shapes.OVAL, {
    x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 },
  });
  slide.addText(String(n), {
    x, y, w: d, h: d,
    fontFace: HEADER_FONT, fontSize: 22, bold: true, color: fontColor,
    align: "center", valign: "middle", margin: 0,
  });
}

function statBlock(slide, x, y, w, h, value, label, valueColor = NAVY) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: WHITE }, line: { color: ICE, width: 1 },
    shadow: { type: "outer", color: "000000", blur: 10, offset: 2, angle: 135, opacity: 0.1 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.10, h, fill: { color: valueColor }, line: { color: valueColor, width: 0 },
  });
  slide.addText(value, {
    x: x + 0.20, y: y + 0.15, w: w - 0.30, h: h * 0.55,
    fontFace: HEADER_FONT, fontSize: 44, bold: true, color: valueColor, margin: 0,
  });
  slide.addText(label, {
    x: x + 0.20, y: y + h * 0.55 + 0.05, w: w - 0.30, h: h * 0.35,
    fontFace: BODY_FONT, fontSize: 12, color: MUTED, margin: 0,
  });
}

function cardBlock(slide, x, y, w, h, title, body, accent = NAVY) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: WHITE }, line: { color: ICE, width: 1 },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h: 0.10, fill: { color: accent }, line: { color: accent, width: 0 },
  });
  slide.addText(title, {
    x: x + 0.22, y: y + 0.25, w: w - 0.44, h: 0.45,
    fontFace: HEADER_FONT, fontSize: 16, bold: true, color: NAVY, margin: 0,
  });
  slide.addText(body, {
    x: x + 0.22, y: y + 0.75, w: w - 0.44, h: h - 0.95,
    fontFace: BODY_FONT, fontSize: 12, color: DARK, margin: 0, valign: "top",
  });
}

function arrowRight(slide, x, y, w = 0.5, h = 0.25, color = MUTED) {
  slide.addShape(pres.shapes.RIGHT_TRIANGLE, {
    x, y, w, h, fill: { color }, line: { color, width: 0 }, rotate: 90,
  });
}

// ---------- SLIDE 1: Title ---------- //
{
  const s = pres.addSlide();
  bg(s, NAVY);
  // Accent diagonal bands
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.35, h: SLIDE_H, fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  s.addText("ACADEMIC PoC · 2026", {
    x: 0.95, y: 1.2, w: 6, h: 0.4,
    fontFace: BODY_FONT, fontSize: 12, color: GOLD, charSpacing: 6, bold: true, margin: 0,
  });
  s.addText("Voice Spoof Detection", {
    x: 0.95, y: 1.7, w: SLIDE_W - 1.5, h: 1.4,
    fontFace: HEADER_FONT, fontSize: 60, bold: true, color: WHITE, margin: 0,
  });
  s.addText("WavLM Base+ ve Wav2Vec2 SSL encoder'larıyla late fusion tabanlı audio-deepfake tespiti", {
    x: 0.95, y: 3.2, w: SLIDE_W - 2, h: 1.0,
    fontFace: BODY_FONT, fontSize: 22, color: ICE, italic: true, margin: 0,
  });
  // Hero stats
  statBlock(s, 0.95, 4.7,   3.0, 1.5, "1.02%", "Eval EER (fusion)", GOLD);
  statBlock(s, 4.10, 4.7,   3.0, 1.5, "99.93%", "Eval ROC-AUC",     GOLD);
  statBlock(s, 7.25, 4.7,   3.0, 1.5, "121k",  "ASVspoof19 LA örnek", GOLD);

  s.addText("Bisher/ASVspoof_2019_LA · HuggingFace · Google Colab A100", {
    x: 0.95, y: 6.45, w: SLIDE_W - 2, h: 0.35,
    fontFace: BODY_FONT, fontSize: 12, color: ICE, italic: true, margin: 0,
  });
  s.addText("github.com/itu-dallasli/Fraudio", {
    x: 0.95, y: 6.85, w: 6, h: 0.35,
    fontFace: BODY_FONT, fontSize: 11, color: GOLD, bold: true, margin: 0,
  });
}

// ---------- SLIDE 2: Problem ---------- //
{
  const s = pres.addSlide();
  bg(s, LIGHT_BG); addTopBar(s); addBottomBar(s);
  addSlideTitle(s,
    "Problem: ses spoofing ne kadar gerçek?",
    "TTS ve voice conversion modelleri saniyeler içinde insan ses imitasyonu üretebiliyor.",
  );

  // 3 card row
  const y = 2.1, w = 3.95, h = 4.6;
  cardBlock(s, 0.6,  y, w, h,
    "Tehdit yüzeyi",
    "• Bankaların ses doğrulaması (voice OTP)\n" +
    "• Politik dezenformasyon, sahte demeçler\n" +
    "• Telefon dolandırıcılığı (CEO scam)\n" +
    "• Mahkeme delili olarak kötüye kullanım",
    RED,
  );
  cardBlock(s, 4.7,  y, w, h,
    "Görev tanımı",
    "16 kHz mono konuşmayı ikili sınıflandır:\n\n" +
    "• bonafide — gerçek insan konuşması\n" +
    "• spoof    — TTS / Voice Conversion çıktısı\n\n" +
    "Çıktı: P(spoof) ∈ [0, 1] ve karar etiketi.",
    NAVY,
  );
  cardBlock(s, 8.8,  y, w, h,
    "Neden zor?",
    "• Spoof artefaktları zaman-frekansta çok ince\n" +
    "• Class imbalance ~1:9 (bonafide:spoof)\n" +
    "• Eval'de görülen TTS sistemleri train'de yok\n" +
    "• Gerçek dünya bonafide ≠ studio bonafide\n" +
    "   → ayrı OOD challenge",
    GOLD,
  );
}

// ---------- SLIDE 3: Dataset ---------- //
{
  const s = pres.addSlide();
  bg(s, LIGHT_BG); addTopBar(s); addBottomBar(s);
  addSlideTitle(s,
    "Veri seti: ASVspoof 2019 LA",
    "Hugging Face üzerinden Bisher/ASVspoof_2019_LA — manuel indirme yok, parquet shard streaming.",
  );

  // Stat row
  statBlock(s, 0.6, 2.0, 2.95, 1.5, "25,380", "Train (bona/spoof: 2,580/22,800)", NAVY);
  statBlock(s, 3.7, 2.0, 2.95, 1.5, "24,844", "Dev (2,548/22,296)",               NAVY);
  statBlock(s, 6.8, 2.0, 2.95, 1.5, "71,237", "Eval (7,355/63,882)",              NAVY);
  statBlock(s, 9.9, 2.0, 2.85, 1.5, "16 kHz", "Mono, native sample rate",         GOLD);

  // Schema table
  s.addText("Şema", {
    x: 0.6, y: 3.8, w: 8, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 18, bold: true, color: NAVY, margin: 0,
  });
  const rows = [
    [{ text: "Kolon", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
     { text: "Tip", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
     { text: "Not", options: { bold: true, fill: { color: NAVY }, color: WHITE } }],
    ["speaker_id",      "string",                                "20 unique LA train'de"],
    ["audio_file_name", "string",                                "Örn. LA_T_1000137"],
    ["audio",           "Audio(sampling_rate=16000)",            "Native 16 kHz, FLAC bytes"],
    ["system_id",       "string",                                "'-' bona, 'A01'-'A19' spoof"],
    ["key",             "ClassLabel(['bonafide', 'spoof'])",     "0 = bonafide, 1 = spoof"],
  ];
  s.addTable(rows, {
    x: 0.6, y: 4.25, w: 12.1, colW: [2.3, 4.2, 5.6],
    border: { pt: 1, color: ICE },
    fontFace: BODY_FONT, fontSize: 13, color: DARK,
    rowH: 0.42,
  });

  s.addText(
    "Speaker overlap yok · Eval sistemleri (A07–A19) train'de (A01–A06) görülmedi · zero-shot generalisation",
    { x: 0.6, y: 6.8, w: SLIDE_W - 1.2, h: 0.3,
      fontFace: BODY_FONT, fontSize: 12, italic: true, color: MUTED, margin: 0 });
}

// ---------- SLIDE 4: Architecture ---------- //
{
  const s = pres.addSlide();
  bg(s, LIGHT_BG); addTopBar(s); addBottomBar(s);
  addSlideTitle(s,
    "Mimari: iki SSL encoder, ortak head, late fusion",
    "Encoder farkı dışındaki her şey aynı — adil karşılaştırma için kritik.",
  );

  // Pipeline diagram
  const py = 2.3, ph = 1.0;
  // Audio input
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: py + 0.95, w: 1.9, h: ph, fill: { color: NAVY }, line: { color: NAVY, width: 0 }, rectRadius: 0.1,
  });
  s.addText("16 kHz waveform\n[B, 64000]", {
    x: 0.6, y: py + 0.95, w: 1.9, h: ph,
    fontFace: BODY_FONT, fontSize: 12, color: WHITE, align: "center", valign: "middle", bold: true, margin: 0,
  });

  // Two encoder boxes
  const encX = 3.2, encW = 2.4;
  s.addShape(pres.shapes.RECTANGLE, {
    x: encX, y: py - 0.05, w: encW, h: ph + 0.2, fill: { color: WHITE }, line: { color: ICE, width: 1 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: encX, y: py - 0.05, w: 0.08, h: ph + 0.2, fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  s.addText("WavLM Base+\n12-layer Transformer", {
    x: encX + 0.15, y: py - 0.05, w: encW - 0.2, h: ph + 0.2,
    fontFace: BODY_FONT, fontSize: 12, bold: true, color: NAVY, align: "left", valign: "middle", margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: encX, y: py + 1.85, w: encW, h: ph + 0.2, fill: { color: WHITE }, line: { color: ICE, width: 1 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: encX, y: py + 1.85, w: 0.08, h: ph + 0.2, fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  s.addText("Wav2Vec2 Base\n12-layer Transformer", {
    x: encX + 0.15, y: py + 1.85, w: encW - 0.2, h: ph + 0.2,
    fontFace: BODY_FONT, fontSize: 12, bold: true, color: NAVY, align: "left", valign: "middle", margin: 0,
  });

  // Pool boxes
  const poolX = 6.1, poolW = 1.8;
  s.addShape(pres.shapes.RECTANGLE, {
    x: poolX, y: py, w: poolW, h: ph, fill: { color: ICE }, line: { color: NAVY, width: 1 },
  });
  s.addText("Masked\nMean + Std", {
    x: poolX, y: py, w: poolW, h: ph,
    fontFace: BODY_FONT, fontSize: 11, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: poolX, y: py + 1.9, w: poolW, h: ph, fill: { color: ICE }, line: { color: NAVY, width: 1 },
  });
  s.addText("Masked\nMean + Std", {
    x: poolX, y: py + 1.9, w: poolW, h: ph,
    fontFace: BODY_FONT, fontSize: 11, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0,
  });

  // Head boxes
  const headX = 8.3, headW = 2.0;
  s.addShape(pres.shapes.RECTANGLE, {
    x: headX, y: py, w: headW, h: ph, fill: { color: WHITE }, line: { color: NAVY, width: 1 },
  });
  s.addText("Head\n1536→256→2", {
    x: headX, y: py, w: headW, h: ph,
    fontFace: BODY_FONT, fontSize: 11, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: headX, y: py + 1.9, w: headW, h: ph, fill: { color: WHITE }, line: { color: NAVY, width: 1 },
  });
  s.addText("Head\n1536→256→2", {
    x: headX, y: py + 1.9, w: headW, h: ph,
    fontFace: BODY_FONT, fontSize: 11, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0,
  });

  // Temp scaling small boxes
  const tX = 10.6, tW = 0.8;
  s.addShape(pres.shapes.OVAL, {
    x: tX, y: py + 0.25, w: tW, h: 0.5, fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  s.addText("÷T₁", {
    x: tX, y: py + 0.25, w: tW, h: 0.5,
    fontFace: BODY_FONT, fontSize: 13, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0,
  });
  s.addShape(pres.shapes.OVAL, {
    x: tX, y: py + 2.15, w: tW, h: 0.5, fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  s.addText("÷T₂", {
    x: tX, y: py + 2.15, w: tW, h: 0.5,
    fontFace: BODY_FONT, fontSize: 13, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0,
  });

  // Fusion big box on right
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 11.7, y: py + 0.95, w: 1.5, h: ph,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 }, rectRadius: 0.08,
  });
  s.addText("Late\nFusion\n(logreg)", {
    x: 11.7, y: py + 0.95, w: 1.5, h: ph,
    fontFace: BODY_FONT, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });

  // Caption / formula
  s.addText("• Frame-level [B, T', 768] → masked pool → [B, 1536] → Linear(1536→256) + LayerNorm + GELU + Dropout(0.3) → Linear(256→2)", {
    x: 0.6, y: 6.0, w: SLIDE_W - 1.2, h: 0.35,
    fontFace: BODY_FONT, fontSize: 12, color: DARK, margin: 0,
  });
  s.addText("• Padding frame'leri encoder._get_feat_extract_output_lengths ile mask'lenir, pooling'e dahil edilmez.", {
    x: 0.6, y: 6.35, w: SLIDE_W - 1.2, h: 0.35,
    fontFace: BODY_FONT, fontSize: 12, color: DARK, margin: 0,
  });
}

// ---------- SLIDE 5: SSL encoder comparison ---------- //
{
  const s = pres.addSlide();
  bg(s, LIGHT_BG); addTopBar(s); addBottomBar(s);
  addSlideTitle(s,
    "SSL backbone karşılaştırması",
    "Aynı head, aynı pipeline. Tek değişen: encoder ön-eğitim verisi ve objektifi.",
  );

  const rows = [
    [{ text: "", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
     { text: "WavLM Base+", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
     { text: "Wav2Vec2 Base", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } }],
    ["Yayınlayan",            "Microsoft (2021)",                                  "Facebook AI (2020)"],
    ["Pre-train veri",        "94,000 saat konuşma + denoising",                   "LibriSpeech 960 saat"],
    ["Objektif",              "Masked speech denoising + contrastive",             "Contrastive (vocoded chunks)"],
    ["Parametre",             "~95 M",                                             "~95 M"],
    ["Hidden boyutu",         "768",                                               "768"],
    ["Açılan katman",         "Son 4 transformer + head",                          "Son 4 transformer + head"],
    ["Bizim Eval EER",        { text: "1.77 %",     options: { bold: true, color: GREEN } },
                              { text: "1.43 %",     options: { bold: true, color: GREEN } }],
    ["Bizim Eval AUC",        "99.82 %",                                           "99.80 %"],
  ];
  s.addTable(rows, {
    x: 0.6, y: 2.0, w: 12.1, colW: [3.5, 4.3, 4.3],
    border: { pt: 1, color: ICE }, fontFace: BODY_FONT, fontSize: 13, color: DARK, rowH: 0.46,
  });

  s.addText("İki backbone tek başına yakın performansta — fusion için bu bir avantaj: complementary error patterns.", {
    x: 0.6, y: 6.5, w: SLIDE_W - 1.2, h: 0.4,
    fontFace: BODY_FONT, fontSize: 13, italic: true, color: MUTED, margin: 0,
  });
}

// ---------- SLIDE 6: Pooling + head math ---------- //
{
  const s = pres.addSlide();
  bg(s, LIGHT_BG); addTopBar(s); addBottomBar(s);
  addSlideTitle(s,
    "Masked mean+std pooling: padding sızıntısını önle",
    "Sadece mean ile spoof TTS'in 'düz' zaman seyri ile bonafide'nin doğal varyasyonu ayırt edilemez.",
  );

  // Left: formula card
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 2.0, w: 7.5, h: 4.5,
    fill: { color: WHITE }, line: { color: ICE, width: 1 },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
  });
  s.addText("Formüller", {
    x: 0.85, y: 2.15, w: 7.0, h: 0.45,
    fontFace: HEADER_FONT, fontSize: 18, bold: true, color: NAVY, margin: 0,
  });
  s.addText("M[b, t] = 1   eğer t < length(b),  yoksa 0", {
    x: 0.85, y: 2.65, w: 7.0, h: 0.4,
    fontFace: "Consolas", fontSize: 13, color: DARK, margin: 0,
  });
  s.addText("mean[b]  =  Σ_t H[b,t] · M[b,t]   /   Σ_t M[b,t]", {
    x: 0.85, y: 3.15, w: 7.0, h: 0.4,
    fontFace: "Consolas", fontSize: 13, color: DARK, margin: 0,
  });
  s.addText("std[b]   =  √( Σ_t (H[b,t] − mean[b])² · M[b,t] / Σ_t M[b,t] )", {
    x: 0.85, y: 3.65, w: 7.0, h: 0.4,
    fontFace: "Consolas", fontSize: 13, color: DARK, margin: 0,
  });
  s.addText("pooled[b] = concat(mean[b], std[b])   →  [B, 1536]", {
    x: 0.85, y: 4.15, w: 7.0, h: 0.4,
    fontFace: "Consolas", fontSize: 13, bold: true, color: NAVY, margin: 0,
  });
  s.addText("Head:", {
    x: 0.85, y: 4.75, w: 7.0, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0,
  });
  s.addText("Linear(1536 → 256)  →  LayerNorm  →  GELU  →  Dropout(0.3)  →  Linear(256 → 2)", {
    x: 0.85, y: 5.15, w: 7.0, h: 0.6,
    fontFace: "Consolas", fontSize: 13, color: DARK, margin: 0,
  });
  s.addText("Same head for both backbones → only encoder differs in ablation.", {
    x: 0.85, y: 5.85, w: 7.0, h: 0.5,
    fontFace: BODY_FONT, fontSize: 12, italic: true, color: MUTED, margin: 0,
  });

  // Right: insight cards
  cardBlock(s, 8.4, 2.0, 4.3, 2.15,
    "Neden mean + std?",
    "TTS sistemleri ortalama spektrumu yakalar; vokoder titreşimleri varyans olarak görünür. " +
    "Std pooling bu varyansı ölçer.", GOLD);
  cardBlock(s, 8.4, 4.25, 4.3, 2.25,
    "Neden partial fine-tune?",
    "Encoder'ın son 4 transformer katmanı + feature_projection açık; CNN feature extractor donduruldu. " +
    "Tam fine-tune ASVspoof'ta hızlı overfit eder.", NAVY);
}

// ---------- SLIDE 7: Augmentation ---------- //
{
  const s = pres.addSlide();
  bg(s, LIGHT_BG); addTopBar(s); addBottomBar(s);
  addSlideTitle(s,
    "Augmentation: clean studio'dan gerçek dünyaya",
    "Yalnız train partition'ında uygulanır. Son iki satır OOD robustness için kritik.",
  );

  const rows = [
    [{ text: "Augmentation", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
     { text: "Olasılık", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
     { text: "Parametreler", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
     { text: "Amaç", options: { bold: true, fill: { color: NAVY }, color: WHITE } }],
    ["Gaussian noise",     { text: "0.50", options: { align: "center" } }, "σ = 0.005",                          "Mikrofon noise floor"],
    ["Random gain",        { text: "0.50", options: { align: "center" } }, "±6 dB",                              "Volume distribution"],
    ["Time shift",         { text: "0.50", options: { align: "center" } }, "±10% of length",                     "Konum invariance"],
    ["Single-tap reverb",  { text: "0.20", options: { align: "center" } }, "50–150 ms gecikme",                  "Oda akustiği"],
    [{ text: "Phone-band filter", options: { bold: true, color: GOLD } },
     { text: "0.35", options: { align: "center", bold: true, color: GOLD } },
     { text: "200–400 Hz HP, 3.0–3.8 kHz LP", options: { bold: true, color: GOLD } },
     { text: "Browser / VoIP spektrumu", options: { bold: true, color: GOLD } }],
    [{ text: "μ-law re-encoding", options: { bold: true, color: GOLD } },
     { text: "0.25", options: { align: "center", bold: true, color: GOLD } },
     { text: "8 / 10 / 12-bit, G.711-style", options: { bold: true, color: GOLD } },
     { text: "Codec artefaktları", options: { bold: true, color: GOLD } }],
  ];
  s.addTable(rows, {
    x: 0.6, y: 2.05, w: 12.1, colW: [3.0, 1.5, 4.2, 3.4],
    border: { pt: 1, color: ICE }, fontFace: BODY_FONT, fontSize: 13, color: DARK, rowH: 0.45,
  });

  // bottom callout
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 5.65, w: 12.1, h: 1.0, fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Neden son iki satır altın renkte?", {
    x: 0.8, y: 5.72, w: 12.0, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 14, bold: true, color: GOLD, margin: 0,
  });
  s.addText("Mikrofon/browser üzerinden gelen bonafide audio WebRTC AGC + codec'ten geçer. " +
            "Bu kanalı eğitime sokmazsan model 'codec artefaktı = TTS' sanır → kullanıcının kendi sesini spoof olarak işaretler.",
    { x: 0.8, y: 6.10, w: 12.0, h: 0.55,
      fontFace: BODY_FONT, fontSize: 12, color: ICE, italic: true, margin: 0 });
}

// ---------- SLIDE 8: Training pipeline ---------- //
{
  const s = pres.addSlide();
  bg(s, LIGHT_BG); addTopBar(s); addBottomBar(s);
  addSlideTitle(s,
    "Eğitim pipeline'ı",
    "ASVspoof'un 1:9 class imbalance'ına karşı sampler + label smoothing + erken durma kombinasyonu.",
  );

  // 5 step row
  const steps = [
    { n: 1, title: "Sampler",  body: "WeightedRandomSampler\n(inverse-frequency)\n→ ~50:50 batch dağılımı" },
    { n: 2, title: "Loss",     body: "CE + label_smoothing 0.10\n(P=0.9999 yasak)" },
    { n: 3, title: "Optimiser",body: "AdamW · iki LR grubu:\nencoder 2e-5  ·  head 5e-4" },
    { n: 4, title: "Schedule", body: "Cosine + warmup 5%\nGrad clip 1.0  ·  BF16 (A100)" },
    { n: 5, title: "Erken durma", body: "4 epoch · best.pt = en düşük dev EER\nClass-collapse early warning" },
  ];
  const baseY = 2.0, cardW = 2.32, cardH = 3.0, gap = 0.10;
  let curX = 0.6;
  for (const st of steps) {
    s.addShape(pres.shapes.RECTANGLE, {
      x: curX, y: baseY, w: cardW, h: cardH, fill: { color: WHITE }, line: { color: ICE, width: 1 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
    });
    numberCircle(s, st.n, curX + cardW / 2 - 0.28, baseY + 0.25);
    s.addText(st.title, {
      x: curX + 0.15, y: baseY + 0.95, w: cardW - 0.3, h: 0.4,
      fontFace: HEADER_FONT, fontSize: 15, bold: true, color: NAVY, align: "center", margin: 0,
    });
    s.addText(st.body, {
      x: curX + 0.15, y: baseY + 1.40, w: cardW - 0.3, h: cardH - 1.5,
      fontFace: BODY_FONT, fontSize: 11, color: DARK, align: "center", valign: "top", margin: 0,
    });
    curX += cardW + gap;
  }

  // bottom callout
  s.addText("Önceki 8 epoch run epoch 5–6 civarı OOD overfit'e geçmişti — 4 epoch'a indirildi.",
    { x: 0.6, y: 5.35, w: SLIDE_W - 1.2, h: 0.4,
      fontFace: BODY_FONT, fontSize: 13, italic: true, color: MUTED, margin: 0 });

  // KPI strip
  statBlock(s, 0.6,  5.85, 3.0, 1.2, "4", "Epoch (down from 8)",       NAVY);
  statBlock(s, 3.75, 5.85, 3.0, 1.2, "0.10", "Label smoothing (ε)",    NAVY);
  statBlock(s, 6.90, 5.85, 3.0, 1.2, "BF16", "Mixed precision (A100)", NAVY);
  statBlock(s, 10.05, 5.85, 2.65, 1.2, "~30 min", "A100'de toplam eğitim", GOLD);
}

// ---------- SLIDE 9: Calibration ---------- //
{
  const s = pres.addSlide();
  bg(s, LIGHT_BG); addTopBar(s); addBottomBar(s);
  addSlideTitle(s,
    "Temperature scaling: güven kalibrasyonu",
    "Tek skaler T dev set'te L-BFGS ile fit edilir. Eval seti calibration'a sızmaz.",
  );

  // Formula card
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 2.0, w: 6.0, h: 1.6, fill: { color: WHITE }, line: { color: ICE, width: 1 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 2.0, w: 0.10, h: 1.6, fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  s.addText("Objective", {
    x: 0.85, y: 2.10, w: 5.8, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0,
  });
  s.addText("T* = argmin_T  −Σ_i log softmax(logits_i / T)[y_i],   T > 0", {
    x: 0.85, y: 2.55, w: 5.8, h: 0.45,
    fontFace: "Consolas", fontSize: 13, color: DARK, margin: 0,
  });
  s.addText("Log-parametrised, single scalar per encoder.", {
    x: 0.85, y: 3.05, w: 5.8, h: 0.5,
    fontFace: BODY_FONT, fontSize: 12, italic: true, color: MUTED, margin: 0,
  });

  // Before/after table
  const rows = [
    [{ text: "Metric", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
     { text: "WavLM before", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "right" } },
     { text: "WavLM after",  options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "right" } },
     { text: "W2V before",   options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "right" } },
     { text: "W2V after",    options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "right" } }],
    ["Temperature T*",  { text: "—", options: { align: "right" } },
                        { text: "1.82",  options: { align: "right", bold: true, color: NAVY } },
                        { text: "—", options: { align: "right" } },
                        { text: "2.04",  options: { align: "right", bold: true, color: NAVY } }],
    ["NLL ↓",           { text: "0.0289", options: { align: "right" } },
                        { text: "0.0198", options: { align: "right", bold: true, color: GREEN } },
                        { text: "0.0416", options: { align: "right" } },
                        { text: "0.0251", options: { align: "right", bold: true, color: GREEN } }],
    ["Brier ↓",         { text: "0.00515", options: { align: "right" } },
                        { text: "0.00475", options: { align: "right", bold: true, color: GREEN } },
                        { text: "0.00595", options: { align: "right" } },
                        { text: "0.00555", options: { align: "right", bold: true, color: GREEN } }],
    ["ECE ↓",           { text: "0.0049", options: { align: "right" } },
                        { text: "0.0017", options: { align: "right", bold: true, color: GREEN } },
                        { text: "0.0057", options: { align: "right" } },
                        { text: "0.0017", options: { align: "right", bold: true, color: GREEN } }],
  ];
  s.addTable(rows, {
    x: 6.8, y: 2.0, w: 5.95, colW: [1.65, 1.05, 1.05, 1.05, 1.15],
    border: { pt: 1, color: ICE }, fontFace: BODY_FONT, fontSize: 11, color: DARK, rowH: 0.40,
  });

  // bottom insight
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 5.0, w: 12.1, h: 1.6, fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("T ≈ 2 anlamı:", {
    x: 0.8, y: 5.10, w: 4.0, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 14, bold: true, color: GOLD, margin: 0,
  });
  s.addText(
    "Model softmax çıkışı biraz 'fazla sivri' — gerçek doğruluk oranından daha yüksek confidence basıyor. " +
    "T = 2 ile bölünce kalibrasyon hatası (ECE) 0.0049'dan 0.0017'ye düşüyor. Önceki bozuk run'da " +
    "T = 0.085 çıkıyordu — sinyal yokken calibration'ın çırpınması.",
    { x: 0.8, y: 5.55, w: 12.0, h: 1.0,
      fontFace: BODY_FONT, fontSize: 12, color: ICE, margin: 0 });
}

// ---------- SLIDE 10: Late fusion ---------- //
{
  const s = pres.addSlide();
  bg(s, LIGHT_BG); addTopBar(s); addBottomBar(s);
  addSlideTitle(s,
    "Late fusion: üç strateji, dev'de yarıştır",
    "Hep dev EER'de en iyi olan kazanır. Eval bu seçime karışmaz.",
  );

  const y = 2.0, w = 4.0, h = 3.5;
  // Average
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y, w, h, fill: { color: WHITE }, line: { color: ICE, width: 1 },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y, w, h: 0.10, fill: { color: MUTED }, line: { color: MUTED, width: 0 } });
  s.addText("1. Average", { x: 0.85, y: y + 0.25, w: w - 0.5, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 18, bold: true, color: NAVY, margin: 0 });
  s.addText("P = 0.5·P_wavlm + 0.5·P_w2v2", {
    x: 0.85, y: y + 0.80, w: w - 0.5, h: 0.5,
    fontFace: "Consolas", fontSize: 12, color: DARK, margin: 0,
  });
  s.addText("Dev EER: 0.19%\nEval EER: 1.08%",
    { x: 0.85, y: y + 1.4, w: w - 0.5, h: 1.0,
      fontFace: BODY_FONT, fontSize: 14, color: DARK, margin: 0 });
  s.addText("En basit baseline; baz çizgisi.", {
    x: 0.85, y: y + 2.5, w: w - 0.5, h: 0.7,
    fontFace: BODY_FONT, fontSize: 11, italic: true, color: MUTED, margin: 0,
  });

  // Weighted
  s.addShape(pres.shapes.RECTANGLE, {
    x: 4.85, y, w, h, fill: { color: WHITE }, line: { color: ICE, width: 1 },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 4.85, y, w, h: 0.10, fill: { color: MUTED }, line: { color: MUTED, width: 0 } });
  s.addText("2. Weighted (α sweep)", { x: 5.10, y: y + 0.25, w: w - 0.5, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 18, bold: true, color: NAVY, margin: 0 });
  s.addText("P = α·P_wavlm + (1−α)·P_w2v2", {
    x: 5.10, y: y + 0.80, w: w - 0.5, h: 0.5,
    fontFace: "Consolas", fontSize: 12, color: DARK, margin: 0,
  });
  s.addText("α* = argmin_α dev EER (21 nokta)\nα* = 0.50\nDev EER: 0.19%\nEval EER: 1.08%",
    { x: 5.10, y: y + 1.4, w: w - 0.5, h: 1.4,
      fontFace: BODY_FONT, fontSize: 13, color: DARK, margin: 0 });

  // Logreg winner
  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.10, y: y - 0.15, w, h: h + 0.30, fill: { color: WHITE }, line: { color: GOLD, width: 3 },
    shadow: { type: "outer", color: "000000", blur: 10, offset: 2, angle: 135, opacity: 0.15 },
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 9.10, y: y - 0.15, w, h: 0.10, fill: { color: GOLD }, line: { color: GOLD, width: 0 } });
  s.addText("3. Logistic regression  ★", { x: 9.35, y: y + 0.10, w: w - 0.5, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 18, bold: true, color: NAVY, margin: 0 });
  s.addText("Δz = z_spoof − z_bona  (calibrated)", {
    x: 9.35, y: y + 0.70, w: w - 0.5, h: 0.4,
    fontFace: "Consolas", fontSize: 11, color: DARK, margin: 0,
  });
  s.addText("P = σ( 0.80·Δz_wavlm + 0.76·Δz_w2v2 + 2.31 )", {
    x: 9.35, y: y + 1.10, w: w - 0.5, h: 0.5,
    fontFace: "Consolas", fontSize: 11, color: DARK, margin: 0,
  });
  s.addText("Dev EER: 0.18%",
    { x: 9.35, y: y + 1.70, w: w - 0.5, h: 0.4,
      fontFace: BODY_FONT, fontSize: 14, color: DARK, margin: 0 });
  s.addText("Eval EER: 1.02%",
    { x: 9.35, y: y + 2.10, w: w - 0.5, h: 0.45,
      fontFace: HEADER_FONT, fontSize: 22, bold: true, color: GREEN, margin: 0 });
  s.addText("İki katsayı yakın — encoder'lar complementary.",
    { x: 9.35, y: y + 2.65, w: w - 0.5, h: 0.7,
      fontFace: BODY_FONT, fontSize: 11, italic: true, color: MUTED, margin: 0 });

  // bottom note
  s.addText("Eval = ASVspoof19 LA test split (71,237 örnek). Calibration ve fusion ağırlığı seçimi yalnız dev'de yapılır.",
    { x: 0.6, y: 6.0, w: SLIDE_W - 1.2, h: 0.5,
      fontFace: BODY_FONT, fontSize: 13, italic: true, color: MUTED, margin: 0 });
}

// ---------- SLIDE 11: Results hero ---------- //
{
  const s = pres.addSlide();
  bg(s, LIGHT_BG); addTopBar(s); addBottomBar(s);
  addSlideTitle(s,
    "Sonuçlar: yayın-kalitesi seviyesinde",
    "ASVspoof 2019 LA eval — 71,237 örnek üzerinde.",
  );

  // Big stat row
  statBlock(s, 0.6,  2.0, 3.0, 2.0, "1.77%",  "WavLM Eval EER",         NAVY);
  statBlock(s, 3.75, 2.0, 3.0, 2.0, "1.43%",  "Wav2Vec2 Eval EER",      NAVY);
  statBlock(s, 6.90, 2.0, 3.0, 2.0, "1.02%",  "Fusion Eval EER ★",      GOLD);
  statBlock(s, 10.05, 2.0, 2.65, 2.0, "99.93%", "Fusion ROC-AUC",       GOLD);

  // Comparison table
  s.addText("Karşılaştırma tablosu", {
    x: 0.6, y: 4.25, w: 8, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 18, bold: true, color: NAVY, margin: 0,
  });
  const rows = [
    [{ text: "Model", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
     { text: "EER", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "right" } },
     { text: "ROC-AUC", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "right" } },
     { text: "F1 (spoof)", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "right" } },
     { text: "min t-DCF", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "right" } },
     { text: "Calibration T*", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "right" } }],
    ["WavLM Base+",     { text: "1.77%",  options: { align: "right" } }, { text: "99.82%", options: { align: "right" } }, { text: "0.9551", options: { align: "right" } }, { text: "0.012", options: { align: "right" } }, { text: "1.82", options: { align: "right" } }],
    ["Wav2Vec2 Base",   { text: "1.43%",  options: { align: "right" } }, { text: "99.80%", options: { align: "right" } }, { text: "0.9222", options: { align: "right" } }, { text: "0.016", options: { align: "right" } }, { text: "2.04", options: { align: "right" } }],
    [{ text: "Fusion (logreg) ★", options: { bold: true, color: GREEN } },
     { text: "1.02%",  options: { align: "right", bold: true, color: GREEN } },
     { text: "99.93%", options: { align: "right", bold: true, color: GREEN } },
     { text: "0.9715", options: { align: "right", bold: true, color: GREEN } },
     { text: "—",      options: { align: "right" } },
     { text: "—",      options: { align: "right" } }],
  ];
  s.addTable(rows, {
    x: 0.6, y: 4.70, w: 12.1, colW: [2.7, 1.6, 1.9, 1.9, 1.9, 2.1],
    border: { pt: 1, color: ICE }, fontFace: BODY_FONT, fontSize: 13, color: DARK, rowH: 0.42,
  });

  s.addText("Tipik literatür aralığı SSL-tabanlı sistemlerde 0.5–3% EER — sistem alt sınırda konumlanıyor.",
    { x: 0.6, y: 6.55, w: SLIDE_W - 1.2, h: 0.4,
      fontFace: BODY_FONT, fontSize: 13, italic: true, color: MUTED, margin: 0 });
}

// ---------- SLIDE 12: Inference + decision logic ---------- //
{
  const s = pres.addSlide();
  bg(s, LIGHT_BG); addTopBar(s); addBottomBar(s);
  addSlideTitle(s,
    "Inference akışı ve karar mantığı",
    "Audio her uzunlukta — kısa için 4 sn center crop, uzun için sliding window (4 sn, 2 sn stride).",
  );

  // Top horizontal pipeline
  const steps = ["Audio in", "Mono + 16 kHz + peak norm", "WavLM + Wav2Vec2 forward", "÷T₁, ÷T₂ calibrate", "Logreg fusion", "Decision logic"];
  const y0 = 2.0;
  const cW = 1.95, gap = 0.10;
  let x = 0.6;
  for (let i = 0; i < steps.length; i++) {
    const fillC = i === steps.length - 1 ? NAVY : WHITE;
    const txtC = i === steps.length - 1 ? WHITE : NAVY;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: y0, w: cW, h: 0.85,
      fill: { color: fillC }, line: { color: NAVY, width: 1 }, rectRadius: 0.08,
    });
    s.addText(steps[i], {
      x: x + 0.05, y: y0, w: cW - 0.1, h: 0.85,
      fontFace: BODY_FONT, fontSize: 11, bold: true, color: txtC, align: "center", valign: "middle", margin: 0,
    });
    if (i < steps.length - 1) {
      arrowRight(s, x + cW + 0.001, y0 + 0.3, 0.10, 0.25, MUTED);
    }
    x += cW + gap;
  }

  // Decision tree section
  s.addText("Karar mantığı (öncelik sırasıyla)", {
    x: 0.6, y: 3.1, w: 8, h: 0.5,
    fontFace: HEADER_FONT, fontSize: 18, bold: true, color: NAVY, margin: 0,
  });

  const decY = 3.65;
  // Decision rows
  const rules = [
    ["1", "Encoder'lar farklı + |P₁ − P₂| ≥ 0.30 → ", "UNCERTAIN"],
    ["2", "|P_fused − threshold| < 0.12 → ",          "UNCERTAIN"],
    ["3", "confidence < 0.45 → ",                       "UNCERTAIN"],
    ["4", "P_fused ≥ 0.65 → ",                         "SPOOF"],
    ["5", "Aksi takdirde → ",                          "BONAFIDE"],
  ];
  for (let i = 0; i < rules.length; i++) {
    const r = rules[i];
    const ry = decY + i * 0.55;
    numberCircle(s, r[0], 0.6, ry - 0.05, 0.45, NAVY, WHITE);
    s.addText(r[1], {
      x: 1.20, y: ry, w: 8.0, h: 0.40,
      fontFace: BODY_FONT, fontSize: 14, color: DARK, valign: "middle", margin: 0,
    });
    const lblColor = r[2] === "SPOOF" ? RED : (r[2] === "BONAFIDE" ? GREEN : GOLD);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 9.30, y: ry - 0.02, w: 1.8, h: 0.45, fill: { color: lblColor }, line: { color: lblColor, width: 0 }, rectRadius: 0.05,
    });
    s.addText(r[2], {
      x: 9.30, y: ry - 0.02, w: 1.8, h: 0.45,
      fontFace: BODY_FONT, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
    });
  }

  s.addText("confidence  =  min(1,  2 · |P_fused − threshold|)", {
    x: 0.6, y: 6.55, w: 8.0, h: 0.35,
    fontFace: "Consolas", fontSize: 13, color: DARK, italic: true, margin: 0,
  });
}

// ---------- SLIDE 13: Demo ---------- //
{
  const s = pres.addSlide();
  bg(s, LIGHT_BG); addTopBar(s); addBottomBar(s);
  addSlideTitle(s,
    "Gradio canlı demo",
    "Mikrofon kaydı + dosya upload + hazır bonafide / spoof örnekleri + canlı eşik slider'ı.",
  );

  // Left: feature list
  cardBlock(s, 0.6, 2.0, 6.0, 4.5,
    "Arayüz",
    "• Mikrofon kaydı veya WAV/MP3/FLAC upload\n\n" +
    "• Hazır örnekler: bonafide.flac + spoof.flac\n" +
    "  (ASVspoof19 LA test split'inden çekildi)\n\n" +
    "• Decision threshold slider (0.30–0.95)\n" +
    "• Uncertainty margin slider (0.00–0.30)\n\n" +
    "• Output: final decision + confidence,\n" +
    "  WavLM P, Wav2Vec2 P, fusion P,\n" +
    "  model agreement, gerekçe metni\n\n" +
    "• 4 sn'den uzun ses → per-window\n" +
    "  spoof probability grafiği",
    NAVY,
  );

  // Right: demo flow
  s.addText("Sunum demo akışı", {
    x: 6.9, y: 2.0, w: 6.0, h: 0.5,
    fontFace: HEADER_FONT, fontSize: 18, bold: true, color: NAVY, margin: 0,
  });

  const flow = [
    { n: 1, title: "bonafide.flac → Analyse", body: "Beklenen: BONAFIDE, yüksek confidence. Model in-distribution güçlü." },
    { n: 2, title: "spoof.flac → Analyse",    body: "Beklenen: SPOOF, yüksek confidence. Eğitilen problemi çözüyor." },
    { n: 3, title: "Mikrofonla kendi sesini ver", body: "Olası: UNCERTAIN ya da threshold'a göre BONAFIDE / SPOOF. Slider'la canlı oynat." },
  ];
  for (let i = 0; i < flow.length; i++) {
    const f = flow[i];
    const ry = 2.6 + i * 1.35;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.9, y: ry, w: 6.0, h: 1.2, fill: { color: WHITE }, line: { color: ICE, width: 1 },
    });
    numberCircle(s, f.n, 7.05, ry + 0.30, 0.5, GOLD, NAVY);
    s.addText(f.title, {
      x: 7.65, y: ry + 0.10, w: 5.1, h: 0.40,
      fontFace: HEADER_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0,
    });
    s.addText(f.body, {
      x: 7.65, y: ry + 0.55, w: 5.1, h: 0.55,
      fontFace: BODY_FONT, fontSize: 11, color: DARK, margin: 0,
    });
  }

  s.addText("python app.py  --wavlm-checkpoint ...  --wav2vec2-checkpoint ...  --fusion-bundle ...  --share", {
    x: 0.6, y: 6.65, w: SLIDE_W - 1.2, h: 0.35,
    fontFace: "Consolas", fontSize: 11, color: MUTED, margin: 0,
  });
}

// ---------- SLIDE 14: OOD limitation + future work ---------- //
{
  const s = pres.addSlide();
  bg(s, LIGHT_BG); addTopBar(s); addBottomBar(s);
  addSlideTitle(s,
    "Bilinen sınırlamalar ve gelecek çalışma",
    "Akademik dürüstlük: sistem clean studio'da neredeyse hatasız, gerçek mikrofonda kırılgan.",
  );

  // Left: OOD problem
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 2.0, w: 6.0, h: 4.5, fill: { color: WHITE }, line: { color: RED, width: 2 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 2.0, w: 6.0, h: 0.55, fill: { color: RED }, line: { color: RED, width: 0 },
  });
  s.addText("Olgu: kullanıcı sesi → 99.99% SPOOF", {
    x: 0.8, y: 2.0, w: 5.8, h: 0.55,
    fontFace: HEADER_FONT, fontSize: 15, bold: true, color: WHITE, valign: "middle", margin: 0,
  });
  s.addText("Neden?", {
    x: 0.85, y: 2.7, w: 5.7, h: 0.40,
    fontFace: HEADER_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0,
  });
  s.addText(
    "ASVspoof bonafide = 2018 studio kayıt, native 16 kHz, temiz oda.\n" +
    "Tarayıcı kaydı = WebRTC AGC + noise suppression + codec yeniden örnekleme + donanım frekans cevabı.\n" +
    "Model bu artefaktları eğitimde görmedi → 'TTS artefaktı' sanıyor.",
    { x: 0.85, y: 3.10, w: 5.7, h: 1.5,
      fontFace: BODY_FONT, fontSize: 11, color: DARK, margin: 0 });
  s.addText("Bu run'da uygulanan azaltıcılar", {
    x: 0.85, y: 4.65, w: 5.7, h: 0.40,
    fontFace: HEADER_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0,
  });
  s.addText(
    "• Phone-band bandpass (p=0.35) + μ-law codec (p=0.25) augmentation\n" +
    "• Label smoothing ε=0.10 (confidence saturation engellenir)\n" +
    "• Epoch 8 → 4 (overfit'e zaman vermeme)\n" +
    "• Decision threshold default 0.5 → 0.65, demoda canlı slider",
    { x: 0.85, y: 5.05, w: 5.7, h: 1.4,
      fontFace: BODY_FONT, fontSize: 11, color: DARK, margin: 0 });

  // Right: future work
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.9, y: 2.0, w: 5.8, h: 4.5, fill: { color: WHITE }, line: { color: GREEN, width: 2 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.9, y: 2.0, w: 5.8, h: 0.55, fill: { color: GREEN }, line: { color: GREEN, width: 0 },
  });
  s.addText("Gelecek çalışma", {
    x: 7.1, y: 2.0, w: 5.6, h: 0.55,
    fontFace: HEADER_FONT, fontSize: 15, bold: true, color: WHITE, valign: "middle", margin: 0,
  });
  s.addText("Kısa vadeli", { x: 7.15, y: 2.7, w: 5.5, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0 });
  s.addText(
    "• Küçük gerçek-dünya bonafide set'iyle post-hoc calibration\n" +
    "• Real RIR + background noise dataset (MUSAN, RIRs)\n" +
    "• RawBoost: time/freq masking, convolutive noise",
    { x: 7.15, y: 3.10, w: 5.5, h: 1.2,
      fontFace: BODY_FONT, fontSize: 11, color: DARK, margin: 0 });
  s.addText("Orta vadeli", { x: 7.15, y: 4.4, w: 5.5, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0 });
  s.addText(
    "• ASVspoof5 / WaveFake / ADD2023 ile yeniden değerlendirme\n" +
    "• Domain adaptation: feature-level adversarial alignment\n" +
    "• Neural-vocoder spesifik artefakt detektörü eklentisi\n" +
    "• HF Spaces'te kalıcı public demo + telemetry",
    { x: 7.15, y: 4.80, w: 5.5, h: 1.7,
      fontFace: BODY_FONT, fontSize: 11, color: DARK, margin: 0 });
}

// ---------- SLIDE 15: Closing ---------- //
{
  const s = pres.addSlide();
  bg(s, NAVY);
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.35, h: SLIDE_H, fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  s.addText("TEŞEKKÜRLER · QUESTIONS?", {
    x: 0.95, y: 1.3, w: 10, h: 0.45,
    fontFace: BODY_FONT, fontSize: 13, color: GOLD, charSpacing: 6, bold: true, margin: 0,
  });
  s.addText("Voice Spoof Detection", {
    x: 0.95, y: 1.8, w: SLIDE_W - 2, h: 1.0,
    fontFace: HEADER_FONT, fontSize: 48, bold: true, color: WHITE, margin: 0,
  });
  s.addText("WavLM Base+  +  Wav2Vec2 Base  →  Logistic-regression fusion", {
    x: 0.95, y: 2.95, w: SLIDE_W - 2, h: 0.5,
    fontFace: BODY_FONT, fontSize: 18, italic: true, color: ICE, margin: 0,
  });

  // Repo links + key takeaways
  cardBlock(s, 0.95, 3.85, 5.8, 2.8,
    "Kaynaklar",
    "Repo:       github.com/itu-dallasli/Fraudio\n" +
    "Dataset:    huggingface.co/datasets/Bisher/ASVspoof_2019_LA\n" +
    "Encoders:   microsoft/wavlm-base-plus, facebook/wav2vec2-base\n" +
    "Doküman:    DOCS.md (teknik referans)\n" +
    "Colab:      Voice_Spoof_Detection_Colab.ipynb",
    GOLD,
  );
  cardBlock(s, 7.05, 3.85, 5.7, 2.8,
    "Take-aways",
    "• İki SSL backbone + ortak head → adil ablation\n" +
    "• Balanced sampler + label smoothing class collapse'ı önler\n" +
    "• Temperature scaling + logreg fusion → ECE %0.17 + EER %1.02\n" +
    "• OOD failure modu açıkça raporlandı (real-world mic)\n" +
    "• Tam reprodüksiyon Colab notebook + DOCS.md ile",
    GOLD,
  );

  s.addText("github.com/itu-dallasli/Fraudio  ·  DOCS.md  ·  python app.py --share", {
    x: 0.95, y: 6.95, w: SLIDE_W - 2, h: 0.4,
    fontFace: BODY_FONT, fontSize: 12, italic: true, color: GOLD, margin: 0,
  });
}

// ---------- write ---------- //
pres.writeFile({ fileName: "Voice_Spoof_Detection_Presentation.pptx" })
  .then(name => console.log("[ok] wrote", name))
  .catch(err => { console.error(err); process.exit(1); });
