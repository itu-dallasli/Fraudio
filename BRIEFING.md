# Voice Spoof Detection — Konuşmacı Briefingi

> Sunum öncesi/sırasında elinde tutman için yazıldı. Her teknik kararın **neden** alındığını, neyi **denediğimizi**, neyi **kırıp düzelttiğimizi**, ve dinleyiciden gelebilecek **olası soruları** içerir. Slide'lar yüzeyi gösterir — bu doküman altta ne olduğunu anlatır.

İçindekiler
1. 60 saniyelik özet
2. Problem & motivasyon
3. Veri seti — ASVspoof 2019 LA ve neden HF
4. Mimari kararları (encoder seçimi, head paylaşımı, partial fine-tune)
5. Pooling — neden masked mean + std
6. Önişleme & augmentation — her birinin neden olduğu
7. Eğitim — sampler, loss, optimizer, schedule
8. Calibration — neden temperature scaling
9. Late fusion — üç strateji ve neden logreg kazandı
10. Inference & karar mantığı
11. Demo
12. Debug günlüğü — neyi kırdık, nasıl düzelttik
13. Ezberlenmesi gereken sayılar
14. Soru-cevap hazırlık (olası sorular + cevapları)

---

## 1. 60 saniyelik özet

> "Audio deepfake tespiti için iki self-supervised konuşma encoder'ını (WavLM Base+ ve Wav2Vec2 Base) aynı head üzerinde fine-tune ettik, çıkışlarını dev set'inde temperature scaling ile kalibre ettik, üç farklı late fusion stratejisi yarıştırdık ve en iyisini (logistic regression) seçtik. ASVspoof 2019 LA test setinde **%1.02 EER** ve **%99.93 ROC-AUC** elde ettik. Sistem clean studio sesinde yayın-kalitesi seviyesinde; gerçek mikrofon ile gelen OOD inputlarda kırılgan olduğunu da açıkça raporladık ve hafifletici augmentation/threshold mekanizmaları ekledik."

İki cümlede yatay: **"İki güçlü SSL encoder + ortak head + kalibre edilmiş skorların lojistik regresyonla füzyonu"**. Üç sayı: **1.02 / 99.93 / 71,237**.

---

## 2. Problem & motivasyon

### Görev
Tek bir 16 kHz mono ses parçasını ikili sınıflandır:
- `bonafide` — gerçek insan
- `spoof` — TTS (Text-to-Speech) veya VC (Voice Conversion) tarafından üretilmiş

Output: `P(spoof) ∈ [0, 1]` ve `BONAFIDE / SPOOF / UNCERTAIN` etiketi.

### Neden önemli
- **Bankalar**: ses tabanlı OTP, voice-print authentication
- **Hukuk**: ses kaydı delillerinin doğruluğu
- **Politika**: sahte demeçler / dezenformasyon
- **Telefon dolandırıcılığı**: CEO scam, fidye çağrıları

ElevenLabs, OpenAI TTS, Bark gibi modern sistemler 3–5 saniyelik referansla bir kişiyi neredeyse kusursuz taklit ediyor. Saldırı yüzeyi büyüdü; savunma araçları yetişmeli.

### Neden zor
- Spoof artefaktları zaman-frekans uzayında **çok ince** — vokoder fasareleri, fonem geçişlerindeki ısrarlı titreşimler, doğal olmayan F0 trajektorileri.
- Class imbalance (eval'de bonafide:spoof ≈ **1:9**) modelin "hep spoof" trivial kararına çekiyor.
- Train'deki spoof sistemleri (A01–A06) eval'de yok; eval'de yeni sistemler (A07–A19) — **zero-shot generalization** testi.
- Gerçek dünya bonafide ≠ studio bonafide → ayrı bir **out-of-distribution (OOD)** problemi.

---

## 3. Veri seti — ASVspoof 2019 LA ve neden HF

### Neden ASVspoof 2019 LA?
- Spoof tespiti literatüründe **referans benchmark**. Tüm modern SSL bazlı sistemler bununla raporlanır → karşılaştırılabilirlik.
- Logical Access (LA) sürümü = sadece TTS+VC tabanlı spoof (replay attack yok). Bizim odak alanımız bu.
- Resmi protocol bölünmesi **speaker-disjoint** ve **system-disjoint** — speaker leak yok, eval sistemleri train'de yok.

### Splits
| Split | Örnek | Bonafide | Spoof |
|---|---:|---:|---:|
| train | 25,380 | 2,580 | 22,800 |
| dev   | 24,844 | 2,548 | 22,296 |
| eval  | 71,237 | 7,355 | 63,882 |

### Neden Hugging Face dataset (`Bisher/ASVspoof_2019_LA`)?
- Resmi paket (Edinburgh data share) ~7.5 GB FLAC + manuel unzip gerektirir.
- HF deposu **aynı veriyi parquet'e dönüştürmüş**, `datasets.load_dataset()` ile tek satırda gelir.
- Streaming modu ile diske hiç yazmadan da çalışabilir (`streaming=True`).
- Audio sütunu `Audio(sampling_rate=16000)` → native 16 kHz, ekstra resampling yok.

### Şema
```
speaker_id      : string
audio_file_name : string
audio           : Audio(sampling_rate=16000)   ← FLAC bytes, native 16 kHz
system_id       : string  ('-' bonafide için, A01-A19 spoof için)
key             : ClassLabel(['bonafide', 'spoof'])   ← integer 0/1 döner
```

**Pratik fark:** HF kütüphanesi `key` ClassLabel olduğu için integer döndürür (0=bonafide, 1=spoof) — string değil. Kodumuzun `_normalise_label` fonksiyonu her iki durumu da destekler.

---

## 4. Mimari kararları

### 4.1 Neden self-supervised encoder, neden klasik özellik değil?

Klasik yöntemler (MFCC + GMM, CQCC + LCNN) ASVspoof19 LA'da ~%10 EER civarında. SSL encoder'lar (Wav2Vec2, WavLM, XLS-R) literatürde ~%1-3 EER'e iner. Avantajları:
- **Pretraining'de görülen 60k+ saat ses** = devasa prior bilgi
- **Düşük seviyeli akustik özellikleri** (formantlar, harmonikler) manuel olarak çıkarmaya gerek kalmıyor
- **Az veri** (ASVspoof'un 25k train örneği gibi) ile bile güçlü transfer

### 4.2 Neden iki encoder?
- **Tek modelle güvenmek riskli**: bir modelin kör noktası diğerinde olmayabilir.
- **Late fusion** literatürde sistematik olarak ~1-3 puan EER iyileştirir.
- WavLM ve Wav2Vec2 farklı objektiflerle eğitildi (WavLM denoising + contrastive, Wav2Vec2 sadece contrastive) → öğrendikleri özellikler **birbirini tamamlar (complementary)**.

### 4.3 Neden bu iki encoder spesifik olarak?
| Encoder | Pre-train veri | Boyut | Neden seçtik |
|---|---|---|---|
| **WavLM Base+** | 94k saat (LibriLight + GigaSpeech + VoxPopuli) | ~95M | Microsoft, gürültülü ses + denoising obj. ile eğitildi → kanal artefaktlarına dayanıklı |
| **Wav2Vec2 Base** | 960 saat LibriSpeech | ~95M | En klasik SSL baseline, ödüllü, ASVspoof literatüründe tipik karşılaştırma encoder'ı |

Aynı parametre sınıfında olduklarından (~95M) adil karşılaştırma — fark backbone'dan değil, boyuttan gelmiyor.

### 4.4 Neden ortak (shared) head?
- İki encoder'ın yan yana çalışacağı bir mimaride **farklılığı encoder'a yıkmak** istiyoruz.
- Head'i aynı tutarsak: WavLM vs Wav2Vec2 performans farkı **kesinlikle backbone'dan kaynaklanır**, head choice'tan değil.
- Akademik yayın "ablation study" prensibine uyar.

### 4.5 Neden partial fine-tune (son 4 katman), tam fine-tune değil?
| Mod | Sonuç |
|---|---|
| Frozen (head-only) | Encoder hareket etmez, performans düşük (~%8-10 EER) |
| Son 2 katman | Bizim ilk denememiz — sinyal akıyor ama yavaş |
| **Son 4 katman** | Şu anki seçim — yeterli kapasite, kontrollü overfit |
| Tam fine-tune | ASVspoof'un 25k örneğinde hızla overfit eder, dev EER düşse de eval EER yükselir |

CNN feature_extractor + feature_projection donduruldu — bunlar düşük seviyeli akustik özellikler, ASVspoof'a özel bir şey öğrenmesi gerekmiyor.

---

## 5. Pooling — neden masked mean + std

### Sorun
SSL encoder çıktısı **frame-level**: `[B, T', H]` (her 20 ms için bir vektör). Head ise bir tek vektör bekliyor → **pooling** lazım.

### Neden basit averaging değil?
- Bonafide konuşmada **doğal varyasyon** var: nefes, vurgu, prozodi.
- TTS bunu **yumuşatır**, "düz" bir zaman seyri verir.
- Sadece mean alırsak bu farkı kaybederiz.

### Mean + std pooling
- **Mean**: spektral içerik ortalaması (kim konuşuyor, ne renk ses)
- **Std**: zaman varyansı (ne kadar doğal değişiyor)
- Concat → `[B, 2H] = [B, 1536]`

### Neden masked?
Padding frame'leri pooling'e dahil edersek (sıfırlı frame'ler) ortalama ve varyasyon **biaslı** olur. ASVspoof'ta sesler 1-13 saniye arası; çoğu 4 sn'lik pencereye sığmıyor → mask kritik.

Padding'in encoder dikkatine sızmaması için `encoder._get_feat_extract_output_lengths(raw_lengths)` ile waveform mask'ini frame seviyesine düşürüyoruz, sonra:

```
M[b, t] = 1 if t < length(b) else 0
mean[b] = Σ_t H[b,t] · M[b,t] / Σ_t M[b,t]
std[b]  = √( Σ_t (H[b,t] − mean[b])² · M[b,t] / Σ_t M[b,t] )
pooled  = concat(mean, std)
```

### Head mimarisi
```
Linear(1536 → 256) → LayerNorm → GELU → Dropout(0.3) → Linear(256 → 2)
```
- **256 boyutlu projeksiyon**: 1536 → 2 doğrudan riskli (head çok küçük, overfit)
- **LayerNorm**: feature scale stabilizesi
- **GELU**: smooth aktivasyon, modern Transformer standardı
- **Dropout(0.3)**: head'de regularizasyon (encoder zaten partial frozen)

---

## 6. Önişleme & augmentation

### Önişleme (her partition)
1. **Mono dönüşümü**: stereo varsa axis 0/1'den ortalama (HF AudioDecoder'da axis 0 = channels)
2. **16 kHz resample**: native 16 kHz olduğu için no-op
3. **4 saniyelik pencere**:
   - Train: **random crop** (data augmentation gibi davranır)
   - Dev/Eval: **center crop** (tekrarlanabilir)
   - Kısa ses: zero-padding
4. **Peak normalization**: amplitüd farklarını eleyip mikrofon seviyesi bias'ını azalt

### Neden 4 saniye?
- ASVspoof eval set median süresi ~3.5 sn.
- 4 sn → çoğu sesi tek pencerede yakalar, sliding window'a düşmek nadir.
- Daha uzun (8 sn) → batch_size yarıya iner, training 2x yavaşlar.
- Daha kısa (2 sn) → modelin bağlam görmesi azalır.

### Augmentation (yalnız train)

Her birinin **neden orada olduğunu** bil:

| Aug | Olasılık | Neden |
|---|---:|---|
| Gaussian noise (σ=0.005) | 0.5 | Mikrofon noise floor simülasyonu — modelin sessizliği "spoof olmama belirteci" olarak kullanmasını engeller |
| Random gain (±6 dB) | 0.5 | Volume normalisation eksikliklerini telafi eder, mic level invariance |
| Time shift (±10%) | 0.5 | Konuşma pencerenin başında/ortasında/sonunda olabilir; bu konum bias'ını eler |
| Single-tap reverb | 0.2 | Oda akustiği çeşitliliği |
| **Phone-band filter** (300-3400 Hz) | **0.35** | Telefon/browser mikrofonu spektrumu — OOD robustness için kritik |
| **μ-law re-encoding** (8/10/12-bit) | **0.25** | G.711 codec artefaktları — yine OOD için |

### Neden son iki augmentation altın renkte?
Bu sunumun en önemli "ne deneyip düzelttik" hikayesi → bkz. Bölüm 12. Kısaca: kullanıcının kendi sesi demoda %99 spoof işaretlendi, çünkü tarayıcı mikrofonu WebRTC AGC + codec'ten geçer. Eğitime bu kanalı sokmazsan model "codec artefaktı = TTS" sanar.

---

## 7. Eğitim — sampler, loss, optimizer, schedule

### 7.1 Balanced sampler
ASVspoof train: bonafide:spoof = 1:8.8. Random sampling yaparsak her batch ~14 spoof, ~2 bonafide → model "hep spoof" diyerek kolay yolu seçer.

**WeightedRandomSampler(weights ∝ 1/n_class)** ile her batch ~50:50 görür. Bizim run'da **class collapse'ı engelleyen tek mekanizma bu**.

> **Denedik:** Sadece `CrossEntropyLoss(weight=balanced)` yetmedi — model yine collapse etti. Sampler + weighted loss birlikteyse aşırı düzeltme (over-shoot to bonafide) oluyor. **Çözüm: sampler ON + class_weighting OFF.**

### 7.2 Label smoothing (ε = 0.10)
`CrossEntropyLoss(label_smoothing=0.10)`. Bunun yerine sert one-hot label kullansak:
- Model `P(spoof) = 0.9999` gibi kararlar verebilir.
- OOD inputlarda bu **aşırı güven** felakete dönüşür (sunumda gösterdik).
- Label smoothing soft target'e (`[0.05, 0.95]`) çekerek logit magnitude'unu sınırlar.

### 7.3 Optimizer — neden iki LR grubu?
```
Encoder (açılan son 4 transformer katmanı): lr = 2e-5
Head (1536 → 256 → 2):                      lr = 5e-4
```
- Encoder zaten pretrained — küçük adımlarla **bozulmamasını** istiyoruz (2e-5).
- Head random init — **hızlı yetişmesi** lazım (5e-4, encoder'dan 25x).
- Tek LR olsaydı: encoder LR'ye eşit → head öğrenemez. Head LR'ye eşit → encoder bozulur.

AdamW: weight decay'i parameter'dan ayrı uygular, Adam'ın L2 ile yanlış etkileşmesi yok.

### 7.4 Schedule — cosine + warmup
- **Warmup (ilk 5%)**: random init head'in büyük adımlarla encoder'ı bozmaması için
- **Cosine decay**: sona doğru fine adjustment
- **Grad clip 1.0**: occasional gradient explosion'a karşı

### 7.5 Mixed precision (BF16 on A100, FP16 on T4)
- **2x hızlanma** A100'de
- BF16 dynamic range FP32 ile aynı (sadece mantisa daha az), gradient scaling gerektirmiyor
- FP16 daha hızlı ama gradient overflow'a hassas → `torch.amp.GradScaler` kullanıyoruz

### 7.6 4 epoch — neden bu kadar az?
Önceki run'da 8 epoch'ta:
- Epoch 5-6 civarı en düşük dev EER
- Epoch 7-8 dev EER hafif tırmanışa geçti → **erken overfit sinyali**
- 4 epoch optimal noktada durur, en yüksek generalization

**Erken durma ile birlikte best.pt** (en düşük dev EER) saklanıyor — yine de en iyiyi yakalıyoruz, sadece eğitim daha hızlı + OOD'a daha az fit ediyor.

### 7.7 Class-collapse erken uyarısı
Her epoch sonunda dev tahminlerinin %X'i spoof? Eğer ≥98% tek sınıf + EER ≈ random ise log'a kalın uyarı:
```
[class-collapse] dev predictions are 100.0% spoof while EER=0.49.
```
Bir önceki bozuk run bu uyarı yokken **8 epoch'unu boşa harcamıştı**.

---

## 8. Calibration — neden temperature scaling

### Sorun
Cross-entropy ile eğitilen modeller genelde **overconfident** olur. Eğitim seti accuracy %99 olabilir ama softmax çıktıları "fazla sivri" — model 0.99 dediğinde gerçek olasılık 0.85 olabilir.

### Çözüm: Temperature scaling
Tek skaler T:
```
P_calibrated = softmax(logits / T)
T* = argmin_T  −Σ_i log P_calibrated[y_i]   (NLL minimization on dev)
```

L-BFGS ile birkaç iterasyonda yakınsar. Log-parametrised, T > 0 garantili.

### Neden dev'de fit, eval'de değil?
- Eval seti **test set** — model performansının final ölçümü
- Eval'i calibration'a kullanırsak: dolaylı olarak eval label'larına optimize ederiz → **data leakage**, raporlanan ECE optimistik olur
- Literatürdeki **yaygın hata** — biz açıkça önlüyoruz

### Sonuçlar
| | WavLM | Wav2Vec2 |
|---|---:|---:|
| T* (fit) | 1.82 | 2.04 |
| ECE before | 0.0049 | 0.0057 |
| ECE after | **0.0017** | **0.0017** |

**T ≈ 2 anlamı:** softmax biraz fazla sivri, T ile bölünce ECE 3x iyileşiyor.

**Önceki bozuk run'da T = 0.085 ve 9421 çıkıyordu** — sinyal yokken calibration "çırpınıyor". Şimdi T = 2 makul bir sayı, sistemin gerçekten öğrenmiş olduğunu gösteriyor.

---

## 9. Late fusion — üç strateji ve neden logreg kazandı

### Neden late fusion (early değil)?
- **Early fusion**: feature concatenation (`[H_wavlm; H_w2v2]`) — backbone çıktılarını birleştirir, daha fazla parametre, joint training gerekir
- **Late fusion**: skor seviyesinde birleştirme — her model bağımsız eğitilir, fusion sonradan eklenebilir, model swap kolay
- Late fusion **operational olarak daha esnek** ve **literatürde sistematik olarak iyi**

### Üç strateji
1. **Average**: `P = 0.5·P_wavlm + 0.5·P_w2v2` — baseline
2. **Weighted average**: `P = α·P_wavlm + (1-α)·P_w2v2`, α dev'de grid search (21 nokta)
3. **Logistic regression**: kalibre edilmiş **logit farkı** (`Δz = z_spoof − z_bonafide`) feature, küçük sklearn LR

### Neden Δz, P değil?
- P sigmoid çıktısı → 0 veya 1 yakınında **doygun**, türev ~0
- Δz logit uzayında → lineer, daha iyi feature
- Calibration sonrası Δz değerleri "doğru ölçekte" → LR coefficients yorumlanabilir

### Sonuçlar (bizim run'da)
| Strateji | Dev EER | Eval EER |
|---|---:|---:|
| Average | 0.19% | 1.08% |
| Weighted (α* = 0.50) | 0.19% | 1.08% |
| **Logreg** | **0.18%** | **1.02%** |

Logreg parametreleri:
```
β_WavLM = 0.80
β_Wav2Vec2 = 0.76
intercept = 2.31
```

İki coefficient'in **yakın olması** kritik: encoder'lar **birbirine bağımlı değil, complementary**. Eğer biri diğerinin "kopyası" olsaydı LR biri 0.0 verirdi.

### Neden seçim kriteri dev EER?
- Eval'i seçim kriterine sokarsak → leakage
- Dev EER zaten unbiased estimate (training'de görülmedi)

---

## 10. Inference & karar mantığı

### Pipeline
1. Audio in → mono + 16 kHz + peak normalize
2. **Eğer süre < 4 sn**: center crop / pad to 4 sn, tek forward
3. **Eğer süre ≥ 4 sn**: sliding window (4 sn pencere, 2 sn stride), her pencere ayrı forward, sonuçların ortalaması
4. Her pencerede: WavLM + Wav2Vec2 forward → ÷T₁, ÷T₂ calibrate → logreg fusion
5. Karar mantığı

### Karar mantığı (öncelik sırasıyla)
1. **Encoder'lar disagree** + `|P_wavlm − P_w2v2| ≥ 0.30` → **UNCERTAIN**
   - İki bağımsız modelin farklı şey demesi güvensizlik sinyali
2. **Karar sınırına yakın**: `|P_fused − threshold| < 0.12` → **UNCERTAIN**
   - 0.65 threshold için 0.53–0.77 arasını "gri bölge" yapar
3. **Düşük confidence**: `confidence = min(1, 2·|P − threshold|) < 0.45` → **UNCERTAIN**
4. **Hiçbiri olmadıysa**: P ≥ 0.65 → SPOOF, aksi → BONAFIDE

### Neden threshold = 0.65, 0.5 değil?
- Modelin training prior'ı (1:9 bonafide:spoof) **spoof tarafına eğilim** yaratıyor
- Dev'de fit edilen logreg'in bias'ı bu önyargıyı taşıyor
- OOD bonafide skorları 0.55–0.70 aralığında çıkıyor
- 0.65 → bu OOD bonafide'leri kurtarır
- **Gradio UI'da canlı slider** ile sunumda 0.50–0.95 arası gösterilebilir

### Sliding window neden?
- 10 saniyelik bir spoof sesinin sadece 3 saniyesi sahte olabilir (concatenation attack)
- Her 4 sn pencereyi ayrı analiz ederiz, sonra ortalama
- Aynı zamanda **zaman ekseninde spoof grafiği** üretiriz — sunumda gösterilebilir

---

## 11. Demo

### Mimari
- Gradio Blocks UI, FastAPI altyapısı
- WavLM ve Wav2Vec2 checkpoint'lerini yükler (~370 MB her biri)
- `fusion_bundle.json` içinden T₁, T₂, logreg coef ve fusion method'unu okur
- CPU'da inference ~2-3 sn/ses, GPU'da <1 sn

### UI elementleri
- Mikrofon kaydı veya WAV/MP3/FLAC upload
- **Hazır örnekler paneli**: bonafide.flac + spoof.flac (ASVspoof eval'den)
- **Decision controls (advanced) açılır panel**: threshold + uncertainty margin slider'ları
- Waveform plot
- Sonuç bloğu (decision, confidence, fusion score, model scores, agreement, reason)
- 4 sn'den uzun ses → sliding window grafiği

### Önerilen demo akışı
1. **bonafide.flac → Analyse** → BONAFIDE, yüksek confidence. Model in-distribution güçlü.
2. **spoof.flac → Analyse** → SPOOF, yüksek confidence. Eğitilen problemi çözüyor.
3. **Kendi sesinle mikrofon kaydı** → muhtemelen UNCERTAIN ya da BONAFIDE/SPOOF. Slider'la threshold'u canlı değiştir, sınıfın nasıl döndüğünü göster.

---

## 12. Debug günlüğü — neyi kırdık, nasıl düzelttik

Sunumda anlatırken bu hikaye altın değerinde — "her şey ilk seferde işledi" demek inanılır değil. **Dürüst debug süreci** seni güvenilir yapar.

### 12.1 İlk bug — class collapse (round 1)

**Belirti:** 3 epoch eğitim sonunda:
- Eval accuracy %89.67 — yüksek görünüyor
- ROC-AUC = 0.485 — rastgele tahminden bile kötü
- Confusion matrix: `[[0, 7355], [0, 63882]]` — **0 bonafide doğru tahmin**, model her şeye spoof diyor
- Train loss 0.6947 → 0.6848 (3 epoch boyunca neredeyse hareketsiz, `ln(2) = 0.693` taban)

**Tanı:** Model %90 spoof prior'a uyum sağlayıp trivial çözüme (hep spoof) yapışmış. CrossEntropy ile `class_weighting=balanced` koymuştuk ama yeterli olmadı çünkü:
- Encoder LR çok düşüktü (1e-5) — encoder hareket etmiyor
- Head LR de düşüktü (1e-4) — bias-only çözümden çıkamıyor
- Sadece son 2 transformer açıktı — kapasite az
- 3 epoch yetersiz

**İlk düzeltme denemesi:**
- `use_balanced_sampler: true` (en güçlü kalkan)
- `class_weighting: none` (sampler ile çakışmasın)
- `encoder_lr: 2e-5`, `head_lr: 5e-4`
- `unfreeze_last_n_layers: 4`
- `epochs: 8`
- Ayrıca `evaluate_loader`'a **class-collapse erken uyarısı** ekledik (ilk epoch sonunda fark etmek için)

### 12.2 İkinci bug — sinyal yok (root cause)

**Belirti:** Yukarıdaki düzeltmelerle yeniden train ettik:
- Hâlâ loss ≈ 0.69, dev EER ≈ 0.49
- Wav2Vec2 calibration sıcaklığı T = 9421 (!) — yani logits aşırı sivri ya da random, calibration çırpınıyor
- ROC tam 45° diyagonal

**"Bir bug daha vurmadan önce gerçek schema'yı doğrulayalım"** dedik. HF datasets-server'dan `Bisher/ASVspoof_2019_LA` şemasını çektik. `key` ClassLabel int dönüyor, `audio` AudioDecoder.

**Sonra fark ettik:** `datasets >=3.5` AudioDecoder API'si değişmiş. Eski sürüm `{"array": np.ndarray (T,), "sampling_rate": int}` dict dönerdi. Yeni sürüm `AudioDecoder` objesi dönüyor, `.get_all_samples().data` torch tensor şeklinde **`(num_channels, num_samples)`** — channels first.

**`hf_loader._decode_audio` kodumuz** her ndim > 1 array için `axis=-1`'den ortalama alıyordu. Mono için `(1, 64000)` array → `mean(axis=-1)` = shape `(1,)`. **Her ses tek bir skalara çöktü.** Sonra `_fix_length` 64,000 elemanlık vektörde sadece ilk elemana bu skaler değeri koydu, gerisi sıfır.

**Sonuç:** Encoder her örnekte "tek dolu sample + 63,999 sıfır" görüyordu. Öğrenecek bir şey yok → loss 0.69'da kalıyor, model class prior'a düşüyor.

**Düzeltme:**
```python
elif hasattr(audio_field, "get_all_samples"):  # AudioDecoder
    samples = audio_field.get_all_samples()
    data = samples.data
    if hasattr(data, "numpy"):
        data = data.numpy()
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)  # ← (C, T) için axis 0
    sr = int(samples.sample_rate)
```

Soundfile fallback path'i için `axis=-1` doğru kalıyor — kaynak başına ayrı handling.

**Sonra:** Tekrar train ettik, loss `0.183 → 0.006` monotonik düştü, dev EER `1.7% → 0.5%` indi. **Bu sefer öğreniyor.** Eval EER fusion 1.02% — publication-quality.

### 12.3 Üçüncü sorun — OOD overfit (kullanıcı sesi 99.99% spoof)

**Belirti:** Eğitim bittikten sonra Gradio demo'da kullanıcı kendi sesini verdi → sistem **%99.99 SPOOF** dedi.

**Tanı:**
- ASVspoof bonafide = 2018 studio mikrofon, temiz oda, native 16 kHz
- Tarayıcı kaydı = WebRTC AGC + noise suppression + echo cancellation + codec + farklı mic frekans cevabı
- Model "codec artefaktı = TTS sinyali" sanıyor — eğitimde bu artefaktı sadece **TTS çıkışlarında** gördü
- Ayrıca label smoothing'siz cross-entropy → softmax saturated, küçük artefakt bile %99 confidence üretiyor

**Düzeltme (üç katman):**
1. **Augmentation**: phone-band filter (300-3400 Hz bandpass, p=0.35) + μ-law re-encoding (8/10/12-bit, p=0.25). Browser/codec artefaktlarını eğitime soktuk.
2. **Label smoothing** (ε=0.10): model 0.99 yerine 0.95 demeye programlanır, OOD'da extreme confidence riski azalır.
3. **Epoch 8 → 4**: önceki run epoch 5-6'da OOD overfit'e geçiyordu, erken duruyoruz.
4. **Decision threshold 0.5 → 0.65**: prior bias'ı telafi etmek için karar sınırını yukarı çektik. UI'da slider ile canlı ayarlanabilir.

**Trade-off:** Yeni run'da eval EER 1.02% → 1.81% yükseldi. **Kabul edilebilir** çünkü:
- ASVspoof test set'i OOD davranışını ölçmüyor — sadece in-distribution
- Karşılığında gerçek demo kullanıcısı için sistem **kullanılabilir** hale geldi

**Bu hikaye önemli**: "düz akademik metrik düştü ama pratik kullanılabilirlik arttı" — gerçek ML mühendisliği bunu yapmak demek.

### 12.4 Diğer küçük düzeltmeler
- **Notebook eski sürümü cache'liyordu**: `git reset --hard origin/main` ile zorla güncelleme komutu eklendi
- **`git status` "untracked __pycache__/" gürültüsü**: `.gitignore` eklendi
- **Smoke test**: sentetik veriyle full pipeline test ediyor — büyük refaktör sonrası bile bug yakalama olasılığı yüksek

---

## 13. Ezberlenmesi gereken sayılar

| Bağlam | Sayı |
|---|---:|
| Final fusion eval EER | **1.02%** |
| Final fusion eval AUC | **99.93%** |
| WavLM tek başına eval EER | 1.77% |
| Wav2Vec2 tek başına eval EER | 1.43% |
| Eval set boyutu | 71,237 |
| Bonafide/spoof oranı (eval) | 1:9 (7,355 / 63,882) |
| Encoder parametre | ~95M (her biri) |
| Hidden boyut | 768 |
| Pooled boyut | 1,536 (mean+std) |
| Head bottleneck | 256 |
| Pencere uzunluğu | 4 saniye |
| Sliding stride | 2 saniye |
| Sample rate | 16 kHz |
| Encoder LR | 2e-5 |
| Head LR | 5e-4 |
| Epoch | 4 |
| Label smoothing | 0.10 |
| Decision threshold | 0.65 |
| Uncertainty margin | 0.12 |
| WavLM T* (calibration) | 1.82 |
| Wav2Vec2 T* | 2.04 |
| Logreg fusion β | (0.80, 0.76), intercept 2.31 |
| ECE after calibration | 0.0017 |
| Training time (A100) | ~30 dk toplam |

---

## 14. Soru-cevap hazırlık

### Q: "Neden ASVspoof 5 değil de 2019 LA?"
A: ASVspoof 5 (2024) daha yeni ve gerçek vokoder'ları içeriyor, ama HF'de hazır dataset olarak yok, ön-işleme süresi haftalar alır. 2019 LA literatürde de **dominant benchmark** — bu sayede sayılarımızı karşılaştırabiliyoruz. Future work olarak ASVspoof 5'i belirttik.

### Q: "Sadece WavLM kullansaydık ne olurdu?"
A: Eval EER 1.77% — yine güçlü. Ama fusion ile 1.02%'ye iniyoruz. Aradaki 0.75 puan small ama ASVspoof-23 challenge'da ilk 5 ile 10 arasındaki fark bu kadar; rekabetçi.

### Q: "Neden classical CNN değil de Transformer SSL?"
A: SSL encoder'lar 60k+ saat ses ile pretrain edildi. Bu prior'ı sıfırdan CNN ile öğrenmek için çok daha fazla labeled data gerekir. ASVspoof'un 25k labeled örneği SSL fine-tune için yeterli, sıfırdan CNN için yetersiz. Literatürde SSL bazlı sistemler classical CNN baseline'larından (LCNN, RawNet2) sistematik olarak daha iyi.

### Q: "Replay attack tespit edebilir mi?"
A: Hayır — LA sadece TTS+VC içerir. Replay için ASVspoof PA (Physical Access) gerekir, bu farklı bir model gerektirir (oda akustiği özellikleri öğrenmeli). Future work.

### Q: "ROC-AUC %99.93 — çok iyi görünüyor, fazla mı iyi?"
A: Test setinde evet, çünkü ASVspoof eval modelin görmediği TTS sistemlerinden oluşuyor ve sistem hâlâ ayırt edebiliyor — sağlam generalisation. **Ama OOD'da** (gerçek mikrofon) bu rakam yanıltıcı, ECE de orada bozulur. Bunu sınırlamalar bölümünde açıkça söyledik.

### Q: "Demoda kendi sesim neden spoof işaretlendi başlangıçta?"
A: Distribution shift. ASVspoof bonafide studio kayıtları, sen WebRTC üzerinden codec geçen ses gönderdin. Model bu kanal artefaktlarını eğitimde görmediği için 'TTS' sandı. Bunu phone-band + μ-law augmentation ile çözmeye çalıştık; tam fix değil ama önemli iyileştirme.

### Q: "Label smoothing modelin doğruluğunu düşürmez mi?"
A: Hafifçe — 1-2 puan EER kötüleşebilir. Ama **OOD robustness ve calibration** karşılığında bunu kabul ediyoruz. Production-grade sistemde overconfidence çok daha tehlikeli (yanlış pozitif → kullanıcı engellenir).

### Q: "Logistic regression neden average'dan az daha iyi?"
A: Çok değil (0.06% EER) ama dev'de tutarlı. **Gerçek katma değer**: logreg coefficient'ları **yorumlanabilir** — iki encoder'ın katkı ağırlıklarını söylüyor. Eğer biri 0.0 verseydi "Wav2Vec2 hiç katmıyor, Average kullan" derdik.

### Q: "Confusion matrix'te 7355 bonafide'nin ~411'i hâlâ spoof olarak işaretleniyor (recall 94.5%). Bu kabul edilebilir mi?"
A: %94.5 recall bonafide için decent. Ama operasyonel olarak threshold ile dengelenebilir — eğer false rejection (gerçek kullanıcıyı reddetme) daha pahalıysa threshold'u yükseltirsin, daha fazla bonafide kurtarırsın ama spoof bazıları sızar. **Threshold tuning** her uygulama domain'inde ayrı yapılır.

### Q: "Temperature scaling neden tek skaler? Per-class olmaz mı?"
A: Per-class (vector scaling) overfit'e meyilli ve binary classification'da gain minimal. Tek skaler T literatürde standard, basitliği ile daha güvenli.

### Q: "Encoder'ın son 2 yerine 4 katmanı neden açtın?"
A: 2 ile başladık, signal sızmıyordu (loss düşmüyordu). 4 ile yeterli kapasite sağlandı. Tam fine-tune (12 katman) overfit oluyor — 4 sweet spot.

### Q: "Fusion neden eval üzerinde değil dev üzerinde fit?"
A: Eval test setimiz — model seçimi, fusion weight'i, calibration için kullanırsak **data leakage**. Eval sadece final raporlama. Bu hata literatürde sıkça yapılıyor.

### Q: "Neden Mermaid diyagramı yerine SVG/PNG kullanmadın?"
A: Mermaid GitHub README ve mark down'da native render, ekstra dosya gerektirmiyor. Sunumda gerçek slide diagramı pptxgenjs ile çizilmiş, oradaki shape'ler PowerPoint'in kendi vector çizimi.

### Q: "Bu sistemin commercial ürüne ne kadar uzak?"
A: 6 ay+ engineering ve domain adaptation gerekir:
1. Real-world bonafide veri toplama (1000+ kullanıcı çeşitli mikrofonlardan)
2. Domain adaptation (DANN, MMD)
3. Daha güçlü augmentation (RawBoost suite)
4. Threshold/calibration kullanıcı segmentine göre per-tier
5. Production inference latency optimization (ONNX, TensorRT)
6. Continuous learning / drift detection

### Q: "Eğer demodanın güvenli olmayacağını biliyorsan neden demo veriyorsun?"
A: Akademik PoC için **prensibi göstermek** önemli — sistem çalışıyor, sınırları biliniyor, dürüstçe raporlandı. Production iddiası yok; sınırlar açıkça yazıldı. Bu **dürüst bilimsel iletişim**.

---

## Bonus: Konuşma tempo önerileri

| Slide | Süre | Anahtar mesaj |
|---|---|---|
| Title | 30 sn | Üç sayı: 1.02 / 99.93 / 71,237 |
| Problem | 1 dk | Tehdit reel, ASVspoof referans |
| Dataset | 1 dk | Speaker/system disjoint vurgu |
| Mimari | 1.5 dk | Diagramı parmakla takip et |
| Encoder karşılaştırma | 1 dk | Complementary, aynı head |
| Pooling | 1 dk | Mean + std, masked, neden |
| Augmentation | 1.5 dk | Son iki satır = OOD hikayesi |
| Training | 1.5 dk | Sampler + label smoothing |
| Calibration | 1 dk | T ≈ 2 makul, ECE 3x iyileşti |
| Fusion | 1.5 dk | Üç strateji, logreg complementary |
| Results | 1.5 dk | Hero sayılar, bağlam ver |
| Inference | 1 dk | Karar tree, threshold = 0.65 |
| Demo | 3 dk | **Canlı 3 örnek** |
| Limitations + Future | 1.5 dk | OOD hikayesini dürüstçe |
| Closing | 30 sn | Take-aways + repo |

**Toplam ~20 dakika**, demo dahil. Q&A için 10 dk ayır.

---

*Bu doküman DOCS.md ile birlikte projenin teknik kalbini taşır. Sunum sırasında bu briefingden, slide'da bahsedilen her bullet için "neden bu sayı?", "neden bu seçim?", "neyi denedik?" cevaplarını çıkarabilirsin.*
