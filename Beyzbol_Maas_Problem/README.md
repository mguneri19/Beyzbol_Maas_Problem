### Beyzbol Maaş Tahmin Projesi

Beyzbol oyuncularının 1986 sezonu ve kariyer istatistiklerinden yola çıkarak maaş tahmini yapan bir makine öğrenmesi çalışması.

- **Veri kümesi**: `Data/hitters.csv`
- **Ana betik**: `Maaş Tahmin.py`
- **Çıktılar**: Grafik ve önem görselleri `cat_plots/`, `num_plots/`, `outlier_plots/`, `plots/`, `feature_importance_plots/`, `special_outlier_plots/` klasörlerine kaydedilir.

### Kurulum

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

CatBoost/macOS kullanıyorsanız ve derleme hatası alırsanız, `xcode-select --install` ile komut satırı araçlarının kurulu olduğundan emin olun.

### Çalıştırma

```bash
python "Maaş Tahmin.py"
```

Betik, EDA çıktıları ve model karşılaştırmalarını üretir; görseller ilgili klasörlere otomatik kaydedilir.

### Proje Yapısı

```
Beyzbol_Maas_Problem/
  Data/
    hitters.csv
  Maaş Tahmin.py
  requirements.txt
  cat_plots/                 # otomatik üretilir (git ignore)
  num_plots/                 # otomatik üretilir (git ignore)
  outlier_plots/             # otomatik üretilir (git ignore)
  plots/                     # otomatik üretilir (git ignore)
  feature_importance_plots/  # otomatik üretilir (git ignore)
  special_outlier_plots/     # otomatik üretilir (git ignore)
```

### Notlar
- Veri seti kaynak anlatımı betik içinde yer almaktadır.
- Büyük/otomatik üretilen dosyalar `.gitignore` ile hariç tutulmuştur.
- Sonuçlar örnek olarak GBM, RF ve CatBoost ile karşılaştırılmıştır.

### Lisans
MIT. Ayrıntı için `LICENSE` dosyasına bakınız.
