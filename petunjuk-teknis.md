# Petunjuk Teknis BDC Satria Data 2026

***Waktu penyelesaian soal mulai tanggal 1  s.d. 30 Juli 2026***

Selamat datang di kompetisi Big Data Challenge (BDC) Satria Data 2026. BDC Satria Data adalah sebuah ajang tahunan yang mengumpulkan talenta-talenta terbaik dalam bidang *data science*, memberikan kesempatan kepada para peserta untuk menerapkan pengetahuan dan keterampilan mereka dalam menyelesaikan masalah nyata menggunakan *big data*.

## 1. Pendahuluan

**Big Data Challenge (BDC)** adalah kompetisi yang dirancang untuk mendorong peserta mengembangkan solusi berbasis data terhadap permasalahan nyata yang dihadapi berbagai organisasi. Permasalahan yang diangkat berasal dari kebutuhan bisnis, operasional, maupun strategis milik mitra penyelenggara, yang dapat berupa perusahaan, instansi pemerintah, lembaga penelitian, organisasi nonpemerintah, institusi pendidikan, ataupun organisasi lainnya. Penyelesaian kasus dilakukan melalui penerapan metode statistika, analitika data, dan *machine learning* dengan memanfaatkan dataset yang telah disediakan oleh panitia.

Dalam pelaksanaannya, panitia tidak menetapkan platform analitik tertentu. Setiap tim bebas menggunakan perangkat keras, perangkat lunak, serta lingkungan komputasi yang dimiliki, dengan dukungan dari perguruan tinggi asal masing-masing. Seluruh hasil analisis kemudian disusun dan diserahkan kepada panitia dalam format yang telah ditentukan melalui platform pengumpulan yang disediakan.

**Kompetisi Big Data Challenge (BDC) Satria Data 2026** diselenggarakan untuk mendorong eksplorasi dan pengembangan solusi berbasis kecerdasan artifisial dalam mendukung pengelolaan limbah yang berkelanjutan. Pada kompetisi ini, peserta ditantang untuk mengembangkan model *machine learning* yang mampu mengklasifikasikan gambar limbah padat (sampah) berdasarkan jenis materialnya, khususnya untuk mengidentifikasi material yang memiliki potensi untuk didaur ulang. Tantangan kompetisi dirancang secara bertahap dengan tingkat kompleksitas yang semakin meningkat, sehingga peserta tidak hanya dituntut menghasilkan model yang akurat, tetapi juga mampu membangun solusi yang tangguh, adaptif, dan dapat diterapkan pada berbagai kondisi data.

## 2. Problem

Pengelolaan sampah merupakan salah satu tantangan utama dalam mewujudkan pembangunan yang berkelanjutan. Pertumbuhan jumlah penduduk, urbanisasi, serta meningkatnya pola konsumsi masyarakat menyebabkan volume sampah yang dihasilkan terus bertambah setiap tahunnya. Apabila tidak dikelola dengan baik, sampah dapat menimbulkan berbagai dampak negatif, seperti pencemaran lingkungan, gangguan kesehatan masyarakat, kerusakan ekosistem, hingga peningkatan emisi gas rumah kaca. Oleh karena itu, diperlukan upaya pengelolaan sampah yang efektif mulai dari tahap pengumpulan, pemilahan, hingga pengolahan.

Salah satu langkah penting dalam pengelolaan sampah adalah melakukan pemilahan berdasarkan karakteristik materialnya. Proses pemilahan memungkinkan material yang masih memiliki nilai ekonomi untuk didaur ulang, sehingga dapat mengurangi jumlah sampah yang berakhir di tempat pembuangan akhir (TPA). Selain itu, pemilahan juga meningkatkan efisiensi proses daur ulang, mengurangi biaya pengelolaan, serta mendukung implementasi ekonomi sirkular (*circular economy*). Keberhasilan proses ini memerlukan partisipasi aktif dari pemerintah, dunia industri, dan seluruh lapisan masyarakat.

Dalam praktiknya, pemilahan sampah secara manual masih menghadapi berbagai kendala, antara lain keterbatasan sumber daya manusia, tingginya volume sampah yang harus diproses, serta potensi terjadinya kesalahan identifikasi jenis material. Seiring berkembangnya teknologi kecerdasan artifisial (*Artificial Intelligence*) dan *computer vision*, proses identifikasi jenis sampah dapat diotomatisasi menggunakan model pembelajaran mesin (*machine learning*) yang mampu mengenali objek berdasarkan citra digital. Pendekatan ini diharapkan dapat meningkatkan kecepatan, konsistensi, dan akurasi proses pemilahan sehingga mendukung sistem pengelolaan sampah yang lebih efisien.

Pada **Big Data Challenge (BDC) Satria Data 2026**, peserta ditantang untuk mengembangkan model klasifikasi citra yang mampu mengidentifikasi objek berdasarkan gambar yang diberikan. Setiap gambar harus diklasifikasikan ke dalam salah satu dari tiga kategori berikut.

- **Recyclable**: Objek yang termasuk ****sampah non-elektronik**** dan memiliki potensi untuk ****didaur ulang****, seperti botol plastik, kaleng, kertas, kardus, atau kaca.

- **Electronic**: Objek yang termasuk ****perangkat elektronik**** atau ****limbah elektronik (e-waste)****, baik yang masih berfungsi maupun sudah rusak, seperti telepon genggam, laptop, keyboard, mouse, charger, kabel, dan perangkat elektronik lainnya.

- **Organic**: Objek yang berasal dari ****bahan hayati**** dan mudah terurai secara alami, seperti daun, buah, sayuran, sisa makanan, ranting kecil, atau bagian tumbuhan lainnya.

Untuk membangun model klasifikasi, panitia menyediakan sekumpulan citra yang telah diberi label sebagai ****Data Latih (train)****. Dataset ini dapat digunakan peserta untuk melakukan eksplorasi data, prapemrosesan (*preprocessing*), pengembangan dan pelatihan model, validasi, serta penyempurnaan model hingga diperoleh performa terbaik.

Dalam pengembangan model, peserta ****hanya diperbolehkan memanfaatkan informasi visual yang terkandung pada citra**** yang disediakan oleh panitia. Penggunaan informasi tambahan di luar konten gambar, seperti metadata, label eksternal, maupun sumber data lain yang dapat memberikan keuntungan di luar ruang lingkup kompetisi, ****tidak diperkenankan****. Ketentuan ini diterapkan untuk menjamin keadilan kompetisi sehingga seluruh peserta membangun model berdasarkan data dan informasi yang sama.

Setelah memperoleh model terbaik, peserta diminta menggunakannya untuk melakukan prediksi terhadap ****Data Uji (test)**** yang disediakan oleh panitia. Hasil prediksi harus diserahkan dalam format ****CSV**** sesuai berkas ****template.csv****. Seluruh nama file citra pada folder ****test**** telah diurutkan berdasarkan nomor ****1 hingga 1458****, dan urutan tersebut harus dipertahankan sesuai dengan urutan pada ****template.csv****. Label sebenarnya (*ground truth*) pada Data Uji tidak dipublikasikan selama kompetisi sehingga kemampuan model dalam melakukan generalisasi terhadap data yang belum pernah dilihat sebelumnya menjadi aspek utama dalam proses penilaian.

Melalui kompetisi ini, peserta diharapkan tidak hanya mampu menghasilkan model dengan tingkat akurasi yang tinggi, tetapi juga mengembangkan solusi yang ****robust****, efisien, dan mampu menggeneralisasi dengan baik pada data baru. Selain menjadi ajang kompetisi, kegiatan ini diharapkan dapat mendorong inovasi pemanfaatan *machine learning*, *computer vision*, dan analitika data untuk mendukung pengembangan sistem pemilahan sampah otomatis yang cerdas dan berkelanjutan.

Model klasifikasi dibangun menggunakan ****Data Latih**** yang telah disediakan oleh panitia, kemudian digunakan untuk memprediksi label pada ****Data Uji****. Performa model akan dievaluasi menggunakan metrik ****Macro-averaged F1-Score****, yaitu rata-rata aritmetika dari nilai ****F1-Score**** yang dihitung secara terpisah untuk setiap kelas.

Peringkat peserta akan ditentukan berdasarkan nilai ****Macro-averaged F1-Score**** pada Data Uji. Semakin tinggi nilai yang diperoleh, semakin baik kemampuan model dalam mengidentifikasi seluruh kategori objek secara konsisten dan seimbang.

## 3. Dataset

Dataset yang digunakan dalam kompetisi ini berupa sekumpulan citra (*image*) yang dibagi menjadi dua bagian, yaitu ****Data Latih (train)**** dan ****Data Uji (test)****.

- ****Data Latih (train)**** terdiri atas ****26.527**** citra yang diberikan informasi label kelasnya. Seluruh citra disimpan pada folder ****train****, dengan setiap kelas ditempatkan pada subfolder yang terpisah sesuai kategorinya.

- ****Data Uji (test)**** terdiri atas ****1.458**** citra tanpa label yang harus diprediksi oleh peserta menggunakan model yang telah dibangun. Seluruh citra disimpan pada folder ****test****, dengan nama file yang telah diurutkan secara berurutan. Urutan nama file tersebut sesuai dengan urutan pada berkas ****template.csv**** yang digunakan untuk pengumpulan hasil prediksi.

Tautan untuk mengunduh dataset adalah sebagai berikut: [https://bit.ly/datasetbdc2026](https://bit.ly/datasetbdc2026)

## 4. Mekanisme Pengumpulan Hasil

Peserta diwajibkan memprediksi data test tidak secara manual  dan mengumpulkan hasil prediksi dalam format CSV standar internasional (pemisah koma) dengan menggunakan template **submission.csv** yang telah disediakan oleh panitia. File tersebut berisi dua kolom: **id** dan **predicted**. Kolom predicted diisi dengan hasil prediksi berupa kode numerik sesuai ketentuan berikut:

- 0 = Recyclable

- 1 = Electronic

- 2 = Organic

Metrik pengukuran yang digunakan oleh Tim Juri BDC pada tahap penyisihan adalah Macro Averaged F1-Score. File submission.csv dapat diunduh pada tautan berikut: [https://bit.ly/submissionbdc2026](https://bit.ly/submissionbdc2026)

### Ketentuan Submission

- Urutan data pada file submission ****tidak boleh diubah**** dan harus sesuai dengan urutan pada berkas **submission.csv** yang disediakan oleh panitia.

- Setiap tim diperbolehkan melakukan ****maksimal tiga kali submission****. Apabila sebuah tim melakukan lebih dari satu submission, maka ****nilai tertinggi**** yang diperoleh akan digunakan sebagai nilai akhir untuk penilaian.

- **Tidak tersedia perpanjangan waktu** untuk pengumpulan submission pada Problem 1. Submission yang diterima setelah batas waktu yang telah ditentukan tidak akan diproses.

- Periode submission ****dibuka mulai tanggal 8 Juli 2026.**** 

## 5. Batasan

Batasan yang berlaku dalam penyelesaian permasalahan pada kompetisi ini adalah sebagai berikut.

1. Model klasifikasi harus dibangun menggunakan **Data Latih** yang telah disediakan oleh panitia.
2. Peserta **hanya diperbolehkan menggunakan informasi visual (image)** yang terdapat pada dataset yang disediakan. Penggunaan metadata, label tambahan, maupun informasi lain di luar konten citra sebagai fitur prediksi tidak diperkenankan.
3. Peserta diperbolehkan melakukan tahapan *preprocessing* dan *data augmentation* terhadap **Data Latih**, sepanjang tidak memanfaatkan informasi dari **Data Uji**.
4. **Data Uji** hanya boleh digunakan untuk menghasilkan prediksi akhir. Peserta dilarang menggunakan **Data Uji** dalam proses pelatihan (*training*), validasi, pemilihan model (*model selection*), maupun penyesuaian parameter (*hyperparameter tuning*).
5. Peserta diperbolehkan menggunakan perangkat lunak, pustaka (*library*), dan kerangka kerja (*framework*) yang tersedia secara bebas (*open source*) untuk membangun model.
6. Peserta diperbolehkan menggunakan model *pre-trained* yang tersedia secara publik untuk keperluan *transfer learning* (seperti *EfficientNet, ResNet, ConvNeXt, Vision Transformer*), sepanjang model tersebut tidak pernah dilatih menggunakan **Data Latih** maupun **Data Uji** kompetisi ini. Peserta wajib mendokumentasikan model dasar (*backbone*) yang digunakan pada laporan akhir.
7. Peserta tidak diperkenankan menambahkan data citra berlabel dari sumber lain untuk proses pelatihan model, sehingga seluruh peserta menggunakan sumber data yang sama.
8. File hasil prediksi (*submission*) harus mengikuti format yang telah ditentukan oleh panitia. Peserta dilarang mengubah nama kolom, format file, maupun urutan data pada file *submission*.
9. Seluruh proses pengembangan model harus dilakukan secara mandiri oleh peserta tanpa memanfaatkan informasi label **Data Uji** maupun hasil evaluasi yang tidak disediakan oleh panitia.
10. Panitia berhak melakukan verifikasi terhadap metode yang digunakan peserta. Apabila ditemukan pelanggaran terhadap ketentuan kompetisi, panitia berhak mendiskualifikasi peserta.

## 6. Aturan Tambahan

Berikut merupakan aturan tambahan pada BDC Satria Data 2026 yaitu:

1. Jawaban untuk problem pertama ini diunggah hingga tanggal 30 Juli 2026 pukul 16:00 WIB.

2. Berdasarkan metrik pengukuran, Tim Juri BDC akan mengambil tim terpilih dengan metrik pengukuran tertinggi dengan ketentuan maksimal tiga tim dari setiap perguruan tinggi. Apabila terdapat tim dengan nilai metrik yang sama, tim yang mengunggah jawaban lebih dahulu akan mendapat peringkat lebih tinggi.

3. Tim terpilih akan diminta untuk membuktikan bahwa pekerjaannya tidak dikerjakan secara manual, menggunakan video. Jika tim tidak dapat membuktikan atau tidak mengirimkan bukti tersebut, maka tim dengan nilai terbaik di bawahnya akan naik menjadi tim terpilih.

4. Jumlah tim terpilih adalah sebanyak 22 tim, yang kemudian dinyatakan lolos ke tahap semifinal.

## 7. Diskualifikasi atau sanksi

Panitia berhak memberikan sanksi, termasuk diskualifikasi, kepada peserta atau tim yang terbukti melanggar ketentuan kompetisi. Bentuk pelanggaran yang dimaksud meliputi, namun tidak terbatas pada, hal-hal berikut.

1. **Pelanggaran terhadap ketentuan kompetisi**, termasuk pengumpulan hasil yang tidak sesuai dengan panduan yang telah ditetapkan, penggunaan data atau informasi yang tidak diizinkan, penggunaan metode yang bertentangan dengan aturan kompetisi, maupun ketidaksesuaian format berkas yang dipersyaratkan.
2. **Pelanggaran integritas akademik**, seperti plagiarisme, penjiplakan sebagian atau seluruh karya pihak lain, manipulasi hasil, pemalsuan informasi, atau tindakan lain yang melanggar prinsip kejujuran akademik.
3. **Kecurangan dalam proses kompetisi**, termasuk tetapi tidak terbatas pada penggunaan label Data Uji, pemanfaatan data yang tidak diperbolehkan, rekayasa hasil prediksi, kolusi dengan peserta lain, maupun bentuk kecurangan lain yang bertujuan memperoleh keuntungan secara tidak sah.
4. **Pelanggaran etika kompetisi**, termasuk tindakan yang mengganggu jalannya kompetisi, memberikan informasi yang menyesatkan, menyampaikan konten yang tidak pantas, atau melakukan tindakan yang merugikan peserta lain, mitra penyelenggara, maupun panitia.
5. **Tidak dapat menunjukkan proses pengembangan model** apabila diminta oleh panitia untuk keperluan verifikasi, termasuk kode program, dokumentasi, atau bukti lain yang mendukung keaslian hasil yang dikumpulkan.

Panitia berhak melakukan proses verifikasi terhadap seluruh karya yang dikumpulkan. Keputusan panitia mengenai pemberian sanksi maupun diskualifikasi bersifat final dan tidak dapat diganggu gugat.

Sanksi ini diterapkan untuk menjaga integritas dan keadilan kompetisi, memastikan bahwa semua peserta berkompetisi dalam kondisi yang setara dan menghormati aturan yang telah ditetapkan.
