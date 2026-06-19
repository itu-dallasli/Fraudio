# Terimler & Kavramlar — Bu Projeye Özel Sözlük

> Sunumda her terim için "bu ne anlama geliyor?" diye sorulduğunda **bir cümlede sezgisel cevap + bir cümlede formal tanım** verebilmen için hazırlandı.

İçindekiler
1. Transformer mimarisi — bizim için ne yapıyor
2. Self-supervised learning (SSL)
3. Sınıflandırma metrikleri (EER, AUC, F1, accuracy, confusion matrix)
4. Calibration metrikleri (NLL, Brier, ECE)
5. Eğitim teknikleri (fine-tune, cross-entropy, sampler, mixed precision, schedule)
6. Calibration & fusion (temperature scaling, logistic regression)
7. Spoof detection özel terimleri (bonafide, TTS, VC, min t-DCF, OOD)
8. Kod & altyapı terimleri (logit, softmax, label smoothing, gradient clipping)

---

## 1. Transformer mimarisi — bu projeye ne katıyor

### 1.1 Sezgisel açıklama

Transformer'in temel operasyonu **self-attention**: bir dizideki her elemanın, dizinin geri kalanıyla nasıl ilişkili olduğunu öğrenir.

Konuşma için bu şu demek: 4 saniyelik ses 199 frame'e bölünür (her frame ~20 ms). Self-attention sayesinde **199. frame, 1. frame'le doğrudan ilişki kurabilir** — uzun mesafe bağlantı kurmak için 199 adım RNN'in içinden geçmesi gerekmiyor.

**Spoof detection için niye önemli:**
- TTS artefaktları zaman içinde **tekrarlayan** desenler oluşturur (vokoder her 80 ms'de aynı titreşimi atar)
- Bonafide konuşmada nefes, tonlama, prozodi **uzun-vadeli yapı** içerir
- Bu iki örüntüyü ayırt etmek için modelin uzak frame'leri karşılaştırabilmesi gerekir → self-attention bunu doğal olarak yapar

### 1.2 CNN ve RNN ile karşılaştırma

| Mimari | Uzak frame'lere erişim | Konuşma için zayıflığı |
|---|---|---|
| CNN | Yalnız komşu pikseller, kademeli genişler | 4 sn = 16,000 sample arasında bağlantı için derin stack gerekir |
| RNN/LSTM | Soldan sağa, son hidden state'te birikir | Uzun mesafede bilgi kaybı, paralelizasyon zor |
| **Transformer** | Her token her tokenı doğrudan görür | Tek sınırlama: hesap karmaşıklığı O(T²) |

WavLM ve Wav2Vec2 12-layer transformer kullanır. Yani 12 farklı seviyede self-attention uygulanır — alt katmanlar düşük seviyeli özellikleri (formant, harmonik), üst katmanlar yüksek seviyeli özellikleri (fonem, prozodi) modeller.

### 1.3 Bizim akışta nerede

```
ham waveform [B, 64000]
       ↓ (CNN feature extractor, donduruldu)
[B, T'=199, 768]
       ↓
12-layer transformer encoder
  ↓ son 4 katman açık (eğitiliyor)
  ↑ ilk 8 katman donduruldu (pretrained kalır)
       ↓
[B, 199, 768]  ← frame-level temsiller
       ↓ masked mean+std pool
[B, 1536]
       ↓ classification head
[B, 2]  ← bonafide / spoof logits
```

Bizim asıl **fine-tune'ladığımız parça transformer'ın son 4 katmanı**. Bunlar ASVspoof'a özel "spoof artefaktı detektörü" haline geliyor.

### 1.4 Attention mask — neden önemli

Self-attention varsayılan olarak **tüm frame'lere** bakar. Bizim sesimiz 4 sn'den kısaysa sıfırlarla padding'lendi → 199 frame'in son N'i sahte. Padding'i attention'a sokarsak:
- Hesaplama gereksiz yere artar
- Daha önemlisi, **pooling biaslanır** (sıfır frame'ler ortalamayı aşağı çeker)

Bu yüzden `attention_mask = [1, 1, ..., 1, 0, 0, ..., 0]` geçiriyoruz. Padding frame'leri hem dikkatten hem pooling'den dışlanır.

### 1.5 "Pretrained" ne anlama geliyor

WavLM ve Wav2Vec2 daha önce **çok büyük etiketsiz ses verisi** ile self-supervised eğitildi:
- WavLM: 94,000 saat (LibriLight + GigaSpeech + VoxPopuli)
- Wav2Vec2: 960 saat (LibriSpeech)

Bu eğitimde model "etiket görmedi" — sadece sesin bir kısmını maskeleyip "ne maskelendi?" sorusunu cevaplamayı öğrendi. Sonuç olarak modelin transformer katmanları **konuşma hakkında genel bilgi** öğrendi (fonemler, hangi sesler birlikte gelir, vb.). Biz bu bilgiyi alıp ASVspoof'a fine-tune ediyoruz — sıfırdan başlamaktansa.

---

## 2. Self-supervised learning (SSL)

**Tanım:** Labeled data olmadan, verinin kendisini "etiket" olarak kullanan eğitim paradigması.

**WavLM/Wav2Vec2 örneği:** Sesin %15'i maskelenir (gri kutuyla örtülür), model maskelenenleri tahmin etmeye çalışır. Hiçbir insan "burası fonemdir / şuradan ses başlar" diye etiket koymaz — model kendi keşfeder.

**Bizim için katma değeri:** ASVspoof'un 25 bin örneği SSL pretrain'in 94 bin saatinin yanında çok küçük. Pretrain ile gelen genel ses bilgisini kullanmasaydık, sıfırdan bu task'ı öğrenmek için yüz binlerce labeled spoof/bonafide örneği gerekirdi.

---

## 3. Sınıflandırma metrikleri

### 3.1 Accuracy (doğruluk) — ve neden yanıltıcı

`accuracy = doğru tahmin / toplam`

**Tuzak:** Eval setimizde bonafide:spoof = 7,355:63,882 = 1:9. Eğer model her şeye "spoof" derse:
- Doğru tahmin: 63,882
- Yanlış: 7,355
- Accuracy: **89.67%** — yüksek görünüyor

Ama sistem **işe yaramaz** — hiçbir bonafide'i doğru tanımıyor. Class imbalance olan datasette accuracy tek başına misleading. Bizim metriklerimizde **EER ve F1** başroldedir.

### 3.2 Precision (kesinlik), Recall (duyarlılık), F1

Spoof'u pozitif sınıf alalım (1 = spoof, 0 = bonafide):

| Terim | Formül | Anlamı |
|---|---|---|
| **True Positive (TP)** | — | Doğru spoof olarak işaretlenmiş spoof |
| **False Positive (FP)** | — | Yanlışlıkla spoof denilmiş bonafide |
| **False Negative (FN)** | — | Kaçırılmış spoof (bonafide denilmiş) |
| **Precision** | TP / (TP + FP) | "Spoof dediklerimin yüzde kaçı gerçekten spoof?" |
| **Recall** | TP / (TP + FN) | "Tüm spoof'ların yüzde kaçını yakaladım?" |
| **F1** | 2·P·R / (P + R) | Precision ve recall'un harmonik ortalaması — tek skor |

**Bizim sonuçlar (fusion):** Precision 99.97%, Recall 94.5%, F1 0.9715.

Yorumla: spoof dediğimde neredeyse her zaman haklıyım (precision yüksek), ama %5.5 spoof beni atlatıyor (recall biraz düşük). Bu trade-off threshold ile ayarlanabilir.

### 3.3 Confusion matrix

```
                Predicted
                bonafide  spoof
True bonafide     6,956    399
True spoof        3,524   60,358
```

(Yaklaşık sayılar — kesin değerler `outputs/evaluation/fusion/confusion_matrix.png`'de.)

Köşegen = doğru tahminler. Off-diagonal = hatalar. Spoof detection için **sol üst köşedeki bonafide-bonafide hücresi** sistemin gerçek kullanıcıları kabul etme oranı.

### 3.4 EER — Equal Error Rate ⭐

Bu **bizim ana metriğimiz**.

**Sezgisel:** "Sistemim ne kadar iyi?" sorusunu **eşikten bağımsız** bir tek sayıyla cevaplar.

**Formal tanım:** İki tip hata var:
- **False Positive Rate (FPR)** = yanlışlıkla spoof denilen bonafide oranı
- **False Negative Rate (FNR)** = kaçırılan spoof oranı

Karar eşiğini yükseltirsen FPR düşer (daha az bonafide yanlış işaretlersin) ama FNR yükselir (daha çok spoof kaçar). Tersi de doğru.

**EER**, FPR = FNR olduğu noktadaki ortak hata oranı.

```
EER = ((FPR + FNR) / 2)  where  FPR == FNR
```

### Görsel olarak

ROC eğrisi (FPR vs TPR) düşünelim. Diyagonal **y = 1 − x** çizgisi (FPR + FNR = 1) ROC ile kesişir → o noktada FPR = FNR = EER.

```
TPR
 1 ┤    ___________   ← ideal sistem: AUC=1.0
   │   /
   │  /  ◇  ← bizim ROC eğrisi
   │ /
   │/   ╲   ╲
   │     ╲ ╲ ← y = 1 − x diyagonali
   │      ◯ ← kesişim noktası: FPR = FNR = EER
 0 ╶─────╲────╲─→  FPR
   0           1
```

**Bizim sayılar:**
- WavLM tek başına: EER = 1.77%
- Wav2Vec2: EER = 1.43%
- Fusion (logreg): **EER = 1.02%**

Bu, "iki tip hata da eşit olacak şekilde eşik koyarsam, modelin her birini %1'in altında yapar" demek.

**Neden EER spoof detection'ta tercih edilir:**
- Threshold-independent (tek sayı, eşik seçimi gerekmez)
- İki hata tipini eşit önemde sayar (asla "false positive'in maliyeti şu, false negative'in bu" tartışmasına girmiyor — bu çoğu zaman bilinmiyor)
- ASVspoof literatürünün **standart metriği** — yayınlanmış sistemlerle direkt karşılaştırma

### 3.5 ROC-AUC

**ROC** (Receiver Operating Characteristic) eğrisi: tüm olası eşikler için (FPR, TPR) noktalarını çizer.

**AUC** (Area Under the Curve) bu eğrinin altındaki alan, 0–1 arası:
- AUC = 0.5 → rastgele tahmin (eğri diyagonal)
- AUC = 1.0 → kusursuz ayırma
- AUC = 0.9 → "model genelde ayırt edebiliyor"
- AUC = 0.999 → "neredeyse her threshold için iyi"

**Sezgisel yorum:** AUC, rastgele seçilen bir spoof'un skoru rastgele seçilen bir bonafide'in skorundan **yüksek olma olasılığıdır**.

**Bizim:** Fusion ROC-AUC = 99.93%. Yani bir bonafide ve bir spoof'u rastgele alsam, sistemin spoof'a daha yüksek skor verme olasılığı 99.93%.

### 3.6 Min t-DCF (tandem Detection Cost Function)

ASVspoof yarışmasının resmi metriği. Spoof tespit sistemini bir ASV (otomatik konuşmacı doğrulama) sistemiyle birlikte değerlendirir — sadece spoof tespiti değil, spoof'un ASV sistemini kandırma maliyeti de hesaba katılır.

**Basitleştirilmiş formül:**
```
DCF(τ) = C_miss · π_target · P_miss(τ) + C_fa · (1 − π_target) · P_fa(τ)
min t-DCF = min_τ DCF(τ)
```

Biz tam tDCF (ASV-coupled) değil, basitleştirilmiş versiyonu hesaplıyoruz (C_miss=1, C_fa=10, π_target=0.05). Bizim: 0.012 (WavLM) — düşük = iyi.

---

## 4. Calibration metrikleri

Calibration: modelin söylediği olasılık ne kadar **gerçek** olasılığa karşılık geliyor?

**Örnek:** Model 100 örneğe "90% spoof" dediyse, gerçekten yaklaşık 90'ı spoof olmalı. Eğer 70'i spoofsa model **overconfident**.

### 4.1 NLL — Negative Log-Likelihood

```
NLL = -Σ_i log P(y_i)
```

Modelin doğru sınıfa verdiği olasılığın logaritmasının negatifi. Düşük = iyi.

**Sezgi:** Model doğru sınıfa 0.99 verdiyse log(0.99) ≈ 0, NLL'e az katkı. Yanlış sınıfa 0.99 verdiyse log(0.01) ≈ -4.6, NLL'e büyük ceza.

**Bizim:** WavLM NLL before/after = 0.0289 / 0.0198.

### 4.2 Brier score

```
Brier = (1/N) · Σ_i (P(spoof|x_i) − y_i)²
```

Olasılık tahminleriyle gerçek 0/1 etiketler arası MSE. Düşük = iyi.

**Sezgi:** Model "0.95 spoof" deyip gerçek spoofsa → (0.95-1)² = 0.0025. Eğer bonafide olsaydı → (0.95-0)² = 0.9025 büyük ceza.

NLL'den farkı: Brier outliers'a daha az hassas (logaritma exponential cezalar verir, kare polynomial).

**Bizim:** WavLM 0.00515 → 0.00475 (kalibrasyon sonrası).

### 4.3 ECE — Expected Calibration Error ⭐

**En önemli calibration metriği.**

Sezgisel: Modelin **dediği güven** ile **gerçek doğruluk** arasındaki ortalama fark.

**Algoritma:**
1. Tüm tahminleri 15 bin'e böl (confidence 0–0.067, 0.067–0.133, ..., 0.933–1.0)
2. Her bin için: o bin'deki tahminlerin **ortalama confidence**'ı ve **gerçek accuracy**'sini hesapla
3. Aralarındaki farkı, bin'in büyüklüğüne göre ağırlıklı ortalamasını al

```
ECE = Σ_b (|B_b|/N) · |accuracy(B_b) − confidence(B_b)|
```

**Sezgisel yorum:**
- ECE = 0.10 → "model %10 ortalama hatalı kalibre"
- ECE = 0.01 → "model neredeyse mükemmel kalibre, dediği güven gerçek"

**Bizim:** Temperature scaling öncesi/sonrası
- WavLM: 0.0049 → **0.0017** (3x iyileşme)
- Wav2Vec2: 0.0057 → **0.0017**

ECE < 0.05 production-grade kabul edilir, biz 0.002 civarındayız — mükemmel.

---

## 5. Eğitim teknikleri

### 5.1 Fine-tune vs frozen

- **Frozen**: encoder ağırlıkları sabit, sadece üstüne eklenen head eğitilir. Hızlı, az kapasite.
- **Full fine-tune**: tüm encoder eğitilir. Yüksek kapasite, overfit riski yüksek.
- **Partial fine-tune (bizim seçimimiz)**: encoder'ın bir kısmı (son 4 katman) açılır, gerisi donar. Denge.

### 5.2 Cross-entropy loss

İkili sınıflandırma için standart loss:

```
CE = − Σ_i [y_i · log P(spoof|x_i) + (1−y_i) · log P(bonafide|x_i)]
```

Pratikte `nn.CrossEntropyLoss(logits, labels)` — softmax + NLL bir arada.

**Neden bu loss?** Olasılıksal yorumlanabilir, sınıflandırma için Bayes-optimal, gradyanı temiz.

### 5.3 Label smoothing (ε = 0.10)

Standart CE hard target kullanır: spoof için `[0, 1]`. Label smoothing soft target'e çevirir: `[0.05, 0.95]` (ε=0.10 için).

**Etkisi:**
- Model artık 1.0 (= sonsuz logit) hedefine doğru itilmiyor
- Logit magnitude'ları sınırlı kalıyor
- Aşırı güven (overconfidence) engelleniyor
- OOD inputlarda saturated output (P=0.9999) felaketi azalır

**Bizim için neden kritik:** Demo'da kullanıcı sesi %99.99 spoof işaretlendi. Label smoothing'le bu maksimum belki %95'e iner, slider'la threshold'u 0.85'e çekersen UNCERTAIN'e düşer.

### 5.4 Class imbalance, balanced sampler

**Sorun:** Train'de bonafide:spoof = 1:8.8. Rastgele batch'te 14 spoof, 2 bonafide → model kolay yol seçer ("hep spoof" der, %88 accuracy alır).

**Çözüm 1 — class-weighted loss:**
```python
CrossEntropyLoss(weight=[w_bonafide, w_spoof])
# w_class ∝ 1 / count_class
```
Loss'ta her yanlış bonafide'a 8.8x ceza verilir.

**Çözüm 2 — WeightedRandomSampler:**
```python
sampler_weights = [1/count_class for class in labels]
WeightedRandomSampler(sampler_weights, num_samples=N, replacement=True)
```
Her batch'te bonafide ve spoof eşit olasılıkla seçilir → batch dağılımı ~50:50.

**Bizim deneyim:** Sadece weighted loss yetmedi (model yine collapsed). Sadece sampler yetti. İkisi birlikte aşırı düzeltme yaratıyordu, sampler ON + weight=none kombinasyonu en iyi.

### 5.5 Mixed precision (BF16 / FP16)

Modelin ileri/geri geçişini 32-bit yerine 16-bit floating-point'le yap. Hız ~2x, bellek ~yarıya iner.

| | FP16 | BF16 |
|---|---|---|
| Sample size | 16 bit | 16 bit |
| Mantissa (precision) | 10 bit | 7 bit |
| Exponent (range) | 5 bit | 8 bit (FP32 ile aynı) |
| Gradient overflow riski | Yüksek | Düşük |
| Hangi GPU? | T4, V100 (FP16 native) | A100, RTX 4090 (BF16 native) |

BF16'da exponent FP32 ile aynı → gradient overflow yok, GradScaler gerekmez. A100'de varsayılan tercih.

### 5.6 Cosine schedule + warmup

LR (learning rate) zamanla değişir:

```
        LR
         │     ___
peak ────┤   /   \___        ← peak'ten sonra cosine ile düşer
         │  /        \___
         │ /             \_
warmup ──┤/____________________ steps
         0   ↑              max
             warmup_steps (5% of total)
```

**Warmup**: random init head'in büyük adımlarla encoder'ı bozmaması için ilk %5'te LR sıfırdan peak'e çıkarılır.

**Cosine decay**: peak'ten sonra LR yumuşak şekilde 0'a iner. Eğitim sonunda fine adjustment.

Alternatif: linear decay, step decay, constant. Cosine SSL fine-tune'da empirik olarak iyi sonuç verir.

### 5.7 Gradient clipping (max_norm = 1.0)

Gradient'ların büyüklüğünü kontrol eder:
```python
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
```

Tüm gradient vektörünün normu 1.0'ı aşarsa orantılı olarak küçültür. **Gradient explosion**'a karşı sigorta — özellikle mixed precision'da kritik.

### 5.8 AdamW

Standart Adam optimizer'ın weight decay'i ayrı uygulayan versiyonu. Modern transformer eğitiminin de-facto seçimi.

Adam: adaptif öğrenme oranı (her parametre için ayrı), momentum kullanır.
AdamW: weight decay'i gradient'a karıştırmaz, ayrı çıkartır — daha temiz regularizasyon.

---

## 6. Calibration & fusion

### 6.1 Logit vs probability

- **Logit**: ham model çıktısı, herhangi bir gerçek sayı (negatif/pozitif). Sınırlı değil.
- **Probability**: softmax sonrası, 0–1 arası, toplamı 1.

```
P(spoof) = softmax(logits)[1] = exp(z_spoof) / (exp(z_bonafide) + exp(z_spoof))
```

**Logit farkı (Δz):**
```
Δz = z_spoof − z_bonafide
P(spoof) = sigmoid(Δz) = 1 / (1 + exp(-Δz))
```

Bizim logreg fusion'ı **Δz** üzerinde çalışır çünkü logit uzayı lineer ve doygun değil.

### 6.2 Softmax

İki çıktı logitini olasılığa çevirir. Toplamı 1, her ikisi de pozitif.

```python
import torch.nn.functional as F
prob = F.softmax(logits, dim=-1)  # [P_bonafide, P_spoof]
```

### 6.3 Temperature scaling

```
P_calibrated = softmax(logits / T)
```

- T = 1 → değişiklik yok
- T > 1 → softmax yumuşar, olasılıklar 0.5'e yaklaşır (model daha az güvenli)
- T < 1 → softmax sertleşir, olasılıklar 0/1'e yapışır (model daha güvenli)

**Bizim T ≈ 2 anlamı:** Model raw çıktıda biraz fazla güvenli, T = 2 ile bölünce kalibre oluyor.

### 6.4 Logistic regression — fusion için

Tek katmanlı sinir ağı (sigmoid output):
```
P(spoof) = σ(β₁ · f₁ + β₂ · f₂ + b)
```

Features (`f₁, f₂`) = iki encoder'ın kalibre edilmiş logit farkları.

**Neden lineer model yeterli?** İki encoder skoru zaten "iyi feature" — non-lineer kombinasyon ekstra fayda getirmiyor, overfit riski getiriyor. Occam'ın usturası: en basit çalışan model.

---

## 7. Spoof detection özel terimleri

### 7.1 Bonafide
"Gerçek insan konuşması". Latince "iyi niyetli" — ASVspoof literatüründe authentic / genuine speech için kullanılır.

### 7.2 Spoof
"Sahte ses" — yapay olarak üretilmiş veya manipüle edilmiş konuşma.

### 7.3 TTS — Text-to-Speech
Metinden konuşma üretme. Modern sistemler (Tacotron 2, FastSpeech, VITS) bir kişinin sesini 3-5 sn referansla taklit edebilir.

### 7.4 VC — Voice Conversion
Bir konuşmacının sesini başka bir konuşmacının sesine dönüştürme. Konuşmanın içeriği aynı kalır, sadece "kim söylüyor" değişir.

ASVspoof 2019 LA'da hem TTS hem VC sistemleri var (A01-A19), train'de 6 sistem, eval'de 13 farklı sistem.

### 7.5 OOD — Out-of-Distribution
Modelin training dağılımı dışından gelen veri.

**Bizim örneğimiz:**
- In-distribution: ASVspoof19 LA test'in studio kayıtları → EER 1%
- Out-of-distribution: kullanıcının laptop mikrofonu üzerinden gelen ses → model bunu spoof sanabilir

OOD generalization ayrı bir problem alanı; domain adaptation, augmentation, data collection ile saldırılır.

### 7.6 ASV — Automatic Speaker Verification
"Bu ses gerçekten X kişisinin mi?" sorusunu cevaplayan sistem. Spoof detection ASV'nin kandırılmasını engellemeyi hedefler.

ASVspoof yarışmasının full ismi: "Automatic Speaker Verification Spoofing And Countermeasures Challenge".

---

## 8. Kod & altyapı terimleri

### 8.1 Padding & attention mask
Variable-length input'ları sabit-uzunlukta batch'e koymak için kısa olanları sıfırla doldurmak. Attention mask hangi pozisyonların gerçek hangi padding olduğunu söyler — modeli padding'i hesaba katmaktan korur.

### 8.2 Pooling
Variable-length sequence'i tek vektöre indirme. Mean pooling, max pooling, attention pooling vb. Bizim seçimimiz: **masked mean + std pooling**.

### 8.3 Dropout
Eğitim sırasında neuronların bir kısmını rastgele "kapatma" — overfit'i azaltır. Bizim head'imizde dropout=0.3 (her forward'ta head neuronlarının %30'u sıfırlanır).

### 8.4 LayerNorm
Her batch örneği için feature vektörünün ortalamasını sıfır, varyansını bir yapar. Eğitim stabilizesi. Transformer'ın her katmanında zaten var, bizim head'imizde de bir tane var.

### 8.5 GELU — Gaussian Error Linear Unit
Aktivasyon fonksiyonu, ReLU'nun daha yumuşak versiyonu:
```
GELU(x) = x · Φ(x)   where Φ is standard normal CDF
```
Modern transformer'ların standardı (BERT, GPT).

### 8.6 Checkpoint
Eğitim sırasında modelin durumunu (ağırlıklar + optimizer state + epoch) diske yazma. Eğitim çökse de devam edebilir, en iyi modeli geri yükleyebiliriz. Bizim `best.pt` = en düşük dev EER veren epoch'un ağırlıkları.

### 8.7 Inference
Eğitilen modeli kullanarak yeni veriler üzerinde tahmin yapma. Eğitimden farkı: gradient hesaplaması yok (`torch.no_grad()`), dropout kapalı, batch norm running stats kullanır.

### 8.8 Calibration leakage
Test setini calibration parametresini (T) bulmak için kullanma. Bu yapıldığında raporlanan calibration metrikleri optimistik olur, gerçek out-of-sample performansı yansıtmaz. **Yapılmaması gereken** ama literatürde sıkça yapılan hata. Biz açıkça dev'de fit ediyoruz.

### 8.9 Data leakage
Genel olarak: test setinin herhangi bir şekilde model seçimine/training'e sızması. Bizim önlemlerimiz:
- Calibration: dev'de fit
- Fusion weight'leri: dev'de seç
- Best epoch: dev EER ile seç
- Test: sadece final raporlama

### 8.10 Reproducibility (seed fixing)
Aynı kodla aynı sonucu almak için tüm random number generator'ları seed'leme:
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### 8.11 Smoke test
Pipeline'ın baştan sona çalıştığını kontrol eden hızlı test — gerçek model performansını değil, **kodun patlamadığını** doğrular. Bizim `tests/test_smoke.py` sentetik veriyle ve mock encoder ile 4 aşamayı test eder, ~5 saniyede koşar.

### 8.12 Class collapse
Modelin tüm input'lar için aynı sınıfı tahmin etmesi — class imbalance'ın klasik failure mode'u. Bizim ilk run'da yaşandı, balanced sampler ile çözüldü, sonradan early-warning log'u eklendi.

---

## Hızlı referans — soru gelirse tek cümle cevaplar

| Soru | Tek cümlelik cevap |
|---|---|
| "EER nedir?" | "False positive ve false negative oranlarının eşit olduğu noktadaki ortak hata oranı — eşikten bağımsız, spoof detection'ın standart metriği." |
| "AUC = 0.999 ne demek?" | "Rastgele bir bonafide ile rastgele bir spoof seçsem, modelin spoof'a daha yüksek skor verme olasılığı %99.9." |
| "Transformer neden uygun?" | "Self-attention sayesinde uzak frame'ler doğrudan etkileşebiliyor — TTS'in tekrarlayan artefaktlarını ve bonafide'nin uzun-vadeli prozodisini ayırt etmek için ideal." |
| "Label smoothing ne için?" | "Modelin P=0.9999 gibi aşırı güvenli kararlar vermesini engelliyor, OOD inputlarda saturated output felaketini azaltıyor." |
| "ECE neden önemli?" | "Modelin söylediği güven gerçek mi? — bir banka 'sistem %95 güvenli' diyorsa, gerçekten %95 doğru olmalı, %50 değil." |
| "Temperature 1.82 ne demek?" | "Modelin çıkışları biraz fazla sivri — T=1.82 ile bölünce kalibre oluyor, ECE 3x iyileşiyor." |
| "Logreg coef 0.80 ve 0.76 ne anlama geliyor?" | "İki encoder neredeyse eşit ağırlıkta katkı yapıyor — biri diğerinin kopyası olmadığını, complementary olduğunu gösterir." |
| "Min t-DCF ne?" | "ASVspoof'un resmi metriği, miss ve false alarm maliyetlerini birlikte tartar — düşük = iyi, biz 0.012'deyiz." |
| "OOD ne demek?" | "Out-of-distribution — modelin eğitimde görmediği tipte veri, bizde gerçek mikrofon kaydı olduğunda yaşanan dağılım kayması." |
| "Self-supervised learning?" | "Label olmadan veriden öğrenme — WavLM 94 bin saat etiketsiz konuşmadan genel ses bilgisi öğrendi, biz onun üstüne ASVspoof'u fine-tune ediyoruz." |

---

*Bu doküman BRIEFING.md ve DOCS.md ile birlikte projenin tam teknik anlatımını oluşturur. Her terim sunumda 30 saniyede açıklanabilecek şekilde yazıldı.*
