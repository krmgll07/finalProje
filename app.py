#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TARIM PERSONELİ ANALİZ SİSTEMİ - WEB API VE DASHBOARD
================================================================================
Bu script, derin öğrenme modelleri ile yapılan personel norm tahminlerini
web üzerinden sunan bir FastAPI uygulamasıdır.

AMAÇ:
    - Personel analiz sonuçlarını REST API ile sunmak
    - İnteraktif bir dashboard ile görselleştirmek
    - İlçe bazlı arama ve filtreleme imkanı sağlamak

KULLANIM:
    python app.py

    Tarayıcıda açın: http://localhost:8080/dashboard

API ENDPOİNTLERİ:
    GET /                       - API bilgileri
    GET /api/summary            - Genel özet istatistikler
    GET /api/districts          - İlçe listesi (filtreli)
    GET /api/districts/{il}/{ilce} - Tek ilçe detayı
    GET /api/search?query=xxx   - İlçe arama
    GET /api/top-deficits/{meslek} - En çok eksik olan ilçeler
    GET /api/top-surpluses/{meslek} - En çok fazla olan ilçeler
    GET /dashboard              - Web arayüzü

Yazar: Kerim GÜLLÜ
Tarih: Aralık 2024
Ders: AIE521 - Derin Öğrenme
================================================================================
"""

# =============================================================================
# KÜTÜPHANE İMPORTLARI
# =============================================================================

# FastAPI - Modern, hızlı web framework
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Pydantic - Veri doğrulama ve şema tanımlama
from pydantic import BaseModel

# Veri işleme
import pandas as pd
import json
from typing import Optional, List, Dict, Any

# Web sunucu
import uvicorn

# Dosya işlemleri
from pathlib import Path
import os

# =============================================================================
# TEMEL YAPILANDIRMA
# =============================================================================

# Bu dosyanın bulunduğu dizini al
# Tüm dosya yolları bu dizine göre belirlenir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(BASE_DIR, 'dashboard')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# =============================================================================
# FASTAPI UYGULAMA OLUŞTURMA
# =============================================================================

# FastAPI uygulamasını oluştur
# title: API dokümantasyonunda görünecek başlık
# description: API açıklaması
# version: API versiyonu
app = FastAPI(
    title="Personel Norm Analiz Sistemi",
    description="Yapay zeka tabanlı tarım personeli norm analizi - Türkiye ilçeleri için optimal personel tahsisi",
    version="1.0.0"
)


# =============================================================================
# VERİ YÜKLEME FONKSİYONU
# =============================================================================

def load_analysis_data():
    """
    CSV dosyalarından analiz verilerini yükle.

    YÜKLENEN DOSYALAR:
        - personnel_analysis_results.csv: Tüm mesleklerin birleşik sonuçları
        - veteriner_analysis_results.csv: Veteriner hekim analizi
        - gida_analysis_results.csv: Gıda mühendisi analizi
        - ziraat_analysis_results.csv: Ziraat mühendisi analizi

    Returns:
        dict: Her meslek için DataFrame içeren sözlük
              veya None (hata durumunda)
    """
    try:
        # Ana sonuç dosyasını yükle (tüm meslekler birleşik)
        main_results = pd.read_csv(os.path.join(RESULTS_DIR, 'personnel_analysis_results.csv'))

        # Meslek bazlı sonuç dosyalarını yükle
        veteriner_results = pd.read_csv(os.path.join(RESULTS_DIR, 'veteriner_analysis_results.csv'))
        gida_results = pd.read_csv(os.path.join(RESULTS_DIR, 'gida_analysis_results.csv'))
        ziraat_results = pd.read_csv(os.path.join(RESULTS_DIR, 'ziraat_analysis_results.csv'))

        return {
            'main': main_results,          # Tüm veriler
            'veteriner': veteriner_results, # Veteriner hekim
            'gida': gida_results,           # Gıda mühendisi
            'ziraat': ziraat_results        # Ziraat mühendisi
        }
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None


# Uygulama başlatıldığında veriyi yükle
# Global değişken olarak tutulur (her istekte tekrar yüklememek için)
data = load_analysis_data()


# =============================================================================
# PYDANTIC MODELLER (API ŞEMALARI)
# =============================================================================
# Pydantic modelleri, API'nin döndüreceği veri yapılarını tanımlar.
# Bu şemalar otomatik olarak API dokümantasyonunda gösterilir.

class DistrictSummary(BaseModel):
    """
    İlçe özet bilgileri şeması.

    Bu model, tek bir ilçenin tüm meslek grupları için
    temel bilgilerini içerir.
    """
    il_adi: str                           # İl adı
    ilce_adi: str                         # İlçe adı
    nufus_18plus: float                   # 18 yaş üstü nüfus
    yuzolcum: float                       # Yüzölçümü (km²)
    toplam_hayvan_2024: float             # 2024 yılı hayvan sayısı

    # Veteriner hekim verileri
    veteriner_hekim: float                # Mevcut veteriner sayısı
    veteriner_tahmini_norm_yuvarlak: float # Tahmin edilen ideal sayı
    veteriner_norm_farki: float           # Fark (mevcut - ideal)
    veteriner_durumu: str                 # Durum (fazla/eksik/dengede)

    # Gıda mühendisi verileri
    gida_muhendisi: float
    gida_tahmini_norm_yuvarlak: float
    gida_norm_farki: float
    gida_durumu: str

    # Ziraat mühendisi verileri
    ziraat_muhendisi: float
    ziraat_tahmini_norm_yuvarlak: float
    ziraat_norm_farki: float
    ziraat_durumu: str


class ProfessionStats(BaseModel):
    """
    Meslek istatistikleri şeması.

    Bir meslek grubu için tüm ilçelerdeki durumların
    özet istatistiklerini içerir.
    """
    profession: str      # Meslek adı
    total_districts: int # Toplam ilçe sayısı
    balanced: int        # Dengede olan ilçe sayısı
    deficit: int         # Eksik personeli olan ilçe sayısı
    surplus: int         # Fazla personeli olan ilçe sayısı
    belirsiz: int        # Belirsiz durumda olan ilçe sayısı
    mae: float           # Ortalama mutlak hata
    r2: float            # R² skoru


class DistrictDetail(BaseModel):
    """
    İlçe detay bilgileri şeması.

    Tek bir ilçe için tüm meslek gruplarının
    detaylı analizini içerir.
    """
    il_adi: str                          # İl adı
    ilce_adi: str                        # İlçe adı
    current_personnel: Dict[str, float]  # Mevcut personel sayıları
    predicted_norm: Dict[str, float]     # Tahmin edilen norm değerleri
    difference: Dict[str, float]         # Fark değerleri
    status: Dict[str, str]               # Durum bilgileri
    details: Dict[str, Any]              # Ek detaylar


# =============================================================================
# API ENDPOİNTLERİ
# =============================================================================

@app.get("/")
async def root():
    """
    Kök endpoint - API bilgilerini döndürür.

    Bu endpoint, API'nin temel bilgilerini ve
    kullanılabilir endpoint'lerin listesini verir.

    Returns:
        dict: API bilgileri ve endpoint listesi
    """
    return {
        "message": "Personel Norm Analiz Sistemi",
        "version": "1.0.0",
        "endpoints": {
            "summary": "/api/summary",              # Özet istatistikler
            "districts": "/api/districts",          # İlçe listesi
            "profession_stats": "/api/stats/{profession}",  # Meslek istatistikleri
            "search": "/api/search",                # Arama
            "dashboard": "/dashboard"               # Web arayüzü
        }
    }


@app.get("/api/summary")
async def get_summary():
    """
    Genel özet istatistikleri döndür.

    Bu endpoint, tüm meslekler için:
        - Toplam ilçe sayısı
        - Dengede/eksik/fazla ilçe sayıları
        - Yüzde dağılımları

    bilgilerini döndürür.

    Returns:
        dict: Özet istatistikler

    Raises:
        HTTPException: Veri yüklenemezse 500 hatası
    """
    # Veri yüklenmiş mi kontrol et
    if data is None:
        raise HTTPException(status_code=500, detail="Veri yüklenemedi")

    try:
        main_df = data['main']

        # Her meslek için istatistikleri hesapla
        stats = {}
        professions = ['veteriner', 'gida', 'ziraat']

        for prof in professions:
            # Durum sütunu adı (örn: veteriner_durumu)
            status_col = f"{prof}_durumu"

            if status_col in main_df.columns:
                # Her durumdan kaç ilçe var say
                status_counts = main_df[status_col].value_counts()
                total = len(main_df)

                stats[prof] = {
                    "total_districts": total,
                    "balanced": int(status_counts.get('dengede', 0)),
                    "deficit": int(status_counts.get('norm_eksigi', 0)),
                    "surplus": int(status_counts.get('norm_fazlasi', 0)),
                    "unclear": int(status_counts.get('belirsiz', 0)),
                    # Yüzde hesapla
                    "percentages": {
                        "balanced": round(status_counts.get('dengede', 0) / total * 100, 1),
                        "deficit": round(status_counts.get('norm_eksigi', 0) / total * 100, 1),
                        "surplus": round(status_counts.get('norm_fazlasi', 0) / total * 100, 1),
                        "unclear": round(status_counts.get('belirsiz', 0) / total * 100, 1)
                    }
                }

        return {
            "total_districts": len(main_df),
            "professions": stats,
            "last_updated": "2024-12-07"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Özet hesaplama hatası: {str(e)}")


@app.get("/api/districts")
async def get_districts(
    profession: Optional[str] = Query(None, description="Meslek filtresi: veteriner, gida, ziraat"),
    status: Optional[str] = Query(None, description="Durum filtresi: dengede, norm_eksigi, norm_fazlasi, belirsiz"),
    limit: int = Query(50, ge=1, le=1000, description="Döndürülecek sonuç sayısı"),
    offset: int = Query(0, ge=0, description="Atlanacak sonuç sayısı (sayfalama için)")
):
    """
    Filtrelenebilir ilçe listesi döndür.

    PARAMETRELER:
        profession: Meslek türüne göre filtrele
        status: Duruma göre filtrele (dengede/eksik/fazla)
        limit: Sayfa başına sonuç sayısı (max 1000)
        offset: Sayfalama için atlama değeri

    ÖRNEK KULLANIM:
        /api/districts?profession=veteriner&status=norm_eksigi&limit=20

    Returns:
        dict: Filtrelenmiş ilçe listesi ve sayfalama bilgileri
    """
    if data is None:
        raise HTTPException(status_code=500, detail="Veri yüklenemedi")

    try:
        main_df = data['main']

        # Filtreleme için kopya oluştur
        filtered_df = main_df.copy()

        # Meslek ve durum filtresini uygula
        if profession and status:
            status_col = f"{profession}_durumu"
            if status_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[status_col] == status]

        # Toplam sonuç sayısı (sayfalama öncesi)
        total_count = len(filtered_df)

        # Sayfalama uygula
        result_df = filtered_df.iloc[offset:offset + limit]

        # DataFrame'i JSON uyumlu listeye çevir
        districts = []
        for _, row in result_df.iterrows():
            district_dict = {}
            for key, value in row.items():
                # NaN değerleri None'a çevir (JSON uyumluluğu için)
                if pd.isna(value):
                    district_dict[key] = None
                else:
                    district_dict[key] = value
            districts.append(district_dict)

        return {
            "total_count": total_count,  # Toplam sonuç sayısı
            "limit": limit,               # İstenen limit
            "offset": offset,             # Atlama değeri
            "districts": districts        # İlçe listesi
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İlçe listesi hatası: {str(e)}")


@app.get("/api/districts/{il_adi}/{ilce_adi}")
async def get_district_detail(il_adi: str, ilce_adi: str):
    """
    Belirli bir ilçenin detaylı bilgilerini döndür.

    URL PARAMETRELERI:
        il_adi: İl adı (örn: "Ankara")
        ilce_adi: İlçe adı (örn: "Çankaya")

    ÖRNEK KULLANIM:
        /api/districts/Ankara/Çankaya

    Returns:
        dict: İlçenin tüm meslek grupları için detaylı analizi

    Raises:
        HTTPException: İlçe bulunamazsa 404 hatası
    """
    if data is None:
        raise HTTPException(status_code=500, detail="Veri yüklenemedi")

    try:
        main_df = data['main']

        # İlçeyi bul (büyük/küçük harf duyarsız)
        district = main_df[
            (main_df['il_adi'].str.lower() == il_adi.lower()) &
            (main_df['ilce_adi'].str.lower() == ilce_adi.lower())
        ]

        # İlçe bulunamadıysa 404 döndür
        if len(district) == 0:
            raise HTTPException(status_code=404, detail="İlçe bulunamadı")

        row = district.iloc[0]

        # Detaylı yanıt oluştur
        result = {
            "il_adi": row['il_adi'],
            "ilce_adi": row['ilce_adi'],
            "demographics": {
                "nufus_18plus": row['nufus_18plus'],        # Nüfus
                "yuzolcum": row['yuzolcum'],                # Yüzölçümü
                "toplam_hayvan_2024": row['toplam_hayvan_2024']  # Hayvan sayısı
            },
            "analysis": {}
        }

        # Her meslek için analiz bilgilerini ekle
        professions = ['veteriner', 'gida', 'ziraat']
        for prof in professions:
            # Sütun adlarını belirle
            current_col = f"{prof}_hekim" if prof == 'veteriner' else f"{prof}_muhendisi"
            predicted_col = f"{prof}_tahmini_norm_yuvarlak"
            diff_col = f"{prof}_norm_farki"
            status_col = f"{prof}_durumu"

            # Tüm sütunlar varsa ekle
            if all(col in row.index for col in [current_col, predicted_col, diff_col, status_col]):
                result["analysis"][prof] = {
                    "current_personnel": row[current_col],      # Mevcut sayı
                    "predicted_norm": row[predicted_col],       # Tahmin edilen
                    "difference": row[diff_col],                # Fark
                    "status": row[status_col],                  # Durum
                    "status_description": get_status_description(row[status_col])  # Açıklama
                }

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İlçe detay hatası: {str(e)}")


@app.get("/api/search")
async def search_districts(
    query: str = Query(..., min_length=2, description="Arama sorgusu (en az 2 karakter)"),
    profession: Optional[str] = Query(None, description="Meslek filtresi")
):
    """
    İlçe adına göre arama yap.

    PARAMETRELER:
        query: Aranacak metin (il veya ilçe adında)
        profession: Opsiyonel meslek filtresi

    ÖRNEK KULLANIM:
        /api/search?query=ankara
        /api/search?query=merkez&profession=veteriner

    Returns:
        dict: Arama sonuçları (maksimum 20 sonuç)
    """
    if data is None:
        raise HTTPException(status_code=500, detail="Veri yüklenemedi")

    try:
        main_df = data['main']

        # İl veya ilçe adında arama yap
        search_results = main_df[
            main_df['il_adi'].str.contains(query, case=False, na=False) |
            main_df['ilce_adi'].str.contains(query, case=False, na=False)
        ]

        # Meslek filtresi uygula (opsiyonel)
        if profession:
            status_col = f"{profession}_durumu"
            if status_col in search_results.columns:
                # Filtreleme istenirse burada uygulanabilir
                pass

        # Sonuçları JSON uyumlu listeye çevir (max 20 sonuç)
        districts = []
        for _, row in search_results.head(20).iterrows():
            district_dict = {}
            for key, value in row.items():
                if pd.isna(value):
                    district_dict[key] = None
                else:
                    district_dict[key] = value
            districts.append(district_dict)

        return {
            "query": query,                        # Arama sorgusu
            "results_count": len(search_results),  # Toplam bulunan
            "districts": districts                  # İlk 20 sonuç
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arama hatası: {str(e)}")


@app.get("/api/top-deficits/{profession}")
async def get_top_deficits(profession: str, limit: int = Query(10, ge=1, le=50)):
    """
    En fazla personel eksiği olan ilçeleri döndür.

    Bu endpoint, belirli bir meslek için personel
    eksikliği en yüksek olan ilçeleri listeler.

    PARAMETRELER:
        profession: Meslek türü (veteriner/gida/ziraat)
        limit: Döndürülecek sonuç sayısı (max 50)

    ÖRNEK KULLANIM:
        /api/top-deficits/veteriner?limit=10

    Returns:
        dict: En çok eksik olan ilçeler listesi
    """
    if data is None:
        raise HTTPException(status_code=500, detail="Veri yüklenemedi")

    try:
        prof_df = data[profession]

        # Sadece eksik durumundaki ilçeleri al
        deficit_districts = prof_df[prof_df[f'{profession}_durumu'] == 'norm_eksigi']

        # En çok eksik olana göre sırala (en negatif değer = en çok eksik)
        top_deficits = deficit_districts.nsmallest(limit, f'{profession}_norm_farki')

        # JSON uyumlu listeye çevir
        districts = []
        for _, row in top_deficits.iterrows():
            district_dict = {}
            for key, value in row.items():
                if pd.isna(value):
                    district_dict[key] = None
                else:
                    district_dict[key] = value
            districts.append(district_dict)

        return {
            "profession": profession,
            "limit": limit,
            "districts": districts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eksik listesi hatası: {str(e)}")


@app.get("/api/top-surpluses/{profession}")
async def get_top_surpluses(profession: str, limit: int = Query(10, ge=1, le=50)):
    """
    En fazla personel fazlası olan ilçeleri döndür.

    Bu endpoint, belirli bir meslek için personel
    fazlalığı en yüksek olan ilçeleri listeler.

    PARAMETRELER:
        profession: Meslek türü (veteriner/gida/ziraat)
        limit: Döndürülecek sonuç sayısı (max 50)

    ÖRNEK KULLANIM:
        /api/top-surpluses/gida?limit=10

    Returns:
        dict: En çok fazla olan ilçeler listesi
    """
    if data is None:
        raise HTTPException(status_code=500, detail="Veri yüklenemedi")

    try:
        prof_df = data[profession]

        # Sadece fazla durumundaki ilçeleri al
        surplus_districts = prof_df[prof_df[f'{profession}_durumu'] == 'norm_fazlasi']

        # En çok fazla olana göre sırala (en pozitif değer = en çok fazla)
        top_surpluses = surplus_districts.nlargest(limit, f'{profession}_norm_farki')

        # JSON uyumlu listeye çevir
        districts = []
        for _, row in top_surpluses.iterrows():
            district_dict = {}
            for key, value in row.items():
                if pd.isna(value):
                    district_dict[key] = None
                else:
                    district_dict[key] = value
            districts.append(district_dict)

        return {
            "profession": profession,
            "limit": limit,
            "districts": districts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fazla listesi hatası: {str(e)}")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """
    Ana dashboard HTML sayfasını sun.

    Bu endpoint, index.html dosyasını döndürerek
    web arayüzünü kullanıcıya sunar.

    Dashboard özellikleri:
        - Genel istatistikler kartları
        - İl bazlı grafik görselleştirme
        - Pasta grafik (durum dağılımı)
        - Detaylı personel listesi tablosu
        - Arama ve filtreleme

    Returns:
        FileResponse: index.html dosyası
    """
    html_path = os.path.join(DASHBOARD_DIR, "index.html")
    return FileResponse(html_path)




@app.get("/dashboard/test-results", response_class=HTMLResponse)
async def test_results_dashboard():
    """Test sonuclari dashboard sayfasini sun."""
    html_path = os.path.join(DASHBOARD_DIR, "test_results.html")
    return FileResponse(html_path)


@app.get("/api/model-results")
async def get_model_results():
    """Model test sonuclarini dondur."""
    try:
        results = {}
        external_test_path = os.path.join(RESULTS_DIR, 'external_test_results.json')
        if os.path.exists(external_test_path):
            with open(external_test_path, 'r', encoding='utf-8') as f:
                results['external_test'] = json.load(f)
        augmentation_path = os.path.join(RESULTS_DIR, 'augmentation_comparison_summary.json')
        if os.path.exists(augmentation_path):
            with open(augmentation_path, 'r', encoding='utf-8') as f:
                results['augmentation'] = json.load(f)
        professions = ['veteriner', 'gida', 'ziraat']
        models = ['mlp', 'resnet', 'attention']
        results['model_comparison'] = {}
        for prof in professions:
            results['model_comparison'][prof] = {}
            for model in models:
                file_path = os.path.join(RESULTS_DIR, f'{prof}_{model}_test_results.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        results['model_comparison'][prof][model] = json.load(f)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model sonuclari hatasi: {str(e)}")

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def get_status_description(status):
    """
    Durum kodunun Türkçe açıklamasını döndür.

    Args:
        status: Durum kodu (dengede/norm_eksigi/norm_fazlasi/belirsiz)

    Returns:
        str: Türkçe açıklama
    """
    descriptions = {
        'dengede': 'Personel sayısı ideal seviyede',
        'norm_eksigi': 'Personel eksikliği var',
        'norm_fazlasi': 'Personel fazlalığı var',
        'belirsiz': 'Durum belirsiz'
    }
    return descriptions.get(status, 'Bilinmeyen durum')


# =============================================================================
# STATİK DOSYA SUNUMU
# =============================================================================

# Statik dosyaları sun (CSS, JS, vb.)
# ÖNEMLİ: Bu mount, tüm route tanımlamalarından SONRA yapılmalı
# Aksi halde statik dosya sunumu API route'larını ezer
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")


# =============================================================================
# ANA GİRİŞ NOKTASI
# =============================================================================

if __name__ == "__main__":
    """
    Script doğrudan çalıştırıldığında web sunucuyu başlat.

    KULLANIM:
        python app.py

    ERIŞIM:
        - Dashboard: http://localhost:8080/dashboard
        - API Docs: http://localhost:8080/docs (Swagger UI)
        - ReDoc: http://localhost:8080/redoc

    AYARLAR:
        - host: "0.0.0.0" = tüm ağ arayüzlerinden erişilebilir
        - port: 8080 (8000 portu başka uygulama kullanıyor olabilir)
        - reload: False = production modunda çalış
    """
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
