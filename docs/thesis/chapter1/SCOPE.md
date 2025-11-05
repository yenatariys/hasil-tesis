# 1.4 Batasan Masalah (Scope)

Batasan masalah dalam penelitian ini adalah:

1. **Data Review**
   - Sumber data dari ulasan aplikasi Disney+ Hotstar pada:
     - App Store
     - Play Store
   - Data diambil pada tanggal 7 April 2025
   - Total data: 1.676 ulasan (838 per platform)
   - Hanya menggunakan ulasan berbahasa Indonesia

2. **Analisis Sentimen**
   - Klasifikasi tiga kelas sentimen:
     - Negatif
     - Netral
     - Positif
   - Perbandingan terbatas pada dua platform:
     - Distribusi sentimen App Store vs Play Store
     - Permasalahan spesifik tiap platform
     - Pola umum antar platform

3. **Implementasi**
   - Tahap preprocessing teks:
     - Case folding
     - Cleaning
     - Stopword removal
     - Stemming bahasa Indonesia
   - Pelabelan sentimen menggunakan leksikon
   - Model klasifikasi menggunakan SVM
   - Dua metode ekstraksi fitur:
     - TF-IDF
     - IndoBERT embeddings
   - Analisis frekuensi kata
   - Visualisasi wordcloud

4. **Batasan Penelitian**
   - Data terbatas pada satu waktu pengambilan
   - Hanya ulasan berbahasa Indonesia
   - Terbatas pada dua platform (tidak termasuk ulasan web)
   - Hanya menggunakan tiga kelas sentimen
   - Klasifikasi terbatas pada model SVM
   - Tidak membahas aspek teknis platform streaming