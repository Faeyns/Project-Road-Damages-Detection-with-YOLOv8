# Project-Road-Damages-Detection-with-YOLOv8

by Farhan Faturahman, Gevin Latifan Abduldjalil, and Rizal Hidayat Putra

Dataset Road Damages Detection versi 7, disediakan oleh Tam, E., dirancang untuk mendeteksi berbagai jenis kerusakan jalan seperti retakan, lubang, atau deformasi lainnya. Dataset ini tersedia di Roboflow Universe dan memiliki anotasi yang sesuai untuk tugas deteksi objek. Dataset ini mendukung pengembangan sistem berbasis deep learning yang dapat digunakan untuk mendukung pemeliharaan jalan secara otomatis.

Dalam eksperimen ini, YOLOv8 digunakan sebagai framework deteksi objek karena performanya yang tinggi dalam mendeteksi objek dengan cepat dan akurat, meskipun menggunakan dataset yang kompleks seperti kerusakan jalan.

**Road Damages Detection v7**

*  **Jumlah kelas (7):** Beberapa kategori kerusakan seperti D00 (Crack), D10 (Pothole),  D20 (Manhole), D40 (Other Deformations).
*  **Jumlah gambar:** 3506 dengan anotasi bounding box untuk setiap kategori kerusakan.
*  **Format dataset:** YOLO format, dengan file data.yaml yang mendefinisikan struktur dataset.
*  **Sumber:** https://universe.roboflow.com/eric-tam-oz6si/road-damages-detection/dataset/7
*  **Pre-processing:** Dataset ini telah melalui tahap pre-processing, di mana objek kerusakan jalan diisolasi dari gambar latar belakang.

---

**Pre-processing:** Isolated Objects

Pre-processing dataset ini bertujuan untuk meningkatkan kualitas pelatihan dengan:

*  **Pemotongan Objek:** Setiap kerusakan jalan diisolasi menggunakan bounding box sehingga latar belakang yang tidak relevan dihilangkan.
*  **Normalisasi:** Resolusi gambar diseragamkan, misalnya 640x640 piksel, untuk memastikan kompatibilitas dengan YOLOv8.


yolov8n.pt: Model YOLOv8 Nano yang telah di-pretrained pada dataset COCO.
Nano: Versi yang ringan dan cepat dari YOLOv8, cocok untuk perangkat dengan sumber daya terbatas.

Ultralytics 8.3.55  Python-3.10.0 torch-2.5.1+cpu CPU (11th Gen Intel Core(TM) i7-11800H 2.30GHz)
Model summary (fused): 168 layers, 3,007,013 parameters, 0 gradients, 8.1 GFLOPs

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 22/22 [00:49

                   all        689        689      0.844      0.887      0.926      0.911
                   D00        149        149      0.837      0.793      0.883      0.858
                   D10        102        102      0.873      0.947      0.974      0.964
                   D20        143        143      0.826      0.853      0.907      0.906
                   D40         45         45      0.586        0.8      0.788      0.762
                   D43         22         22      0.973          1      0.995      0.995
                   D44        112        112      0.851      0.911      0.961      0.961
                   D50        116        116       0.96      0.905      0.976      0.927
Speed: 1.4ms preprocess, 61.9ms inference, 0.0ms loss, 0.3ms postprocess per image

![Untitled](https://github.com/user-attachments/assets/e931755e-d504-4c97-8da1-e5e8541e9151)

![Untitled-1](https://github.com/user-attachments/assets/24c2c6a2-e6d3-4884-8d79-1002611d75a8)

![Untitled](https://github.com/user-attachments/assets/2a3256b2-8468-4535-a53e-00bd970683b2)

Untuk detail training lanjutnya, serta validation, testing, dan juga inference beserta evaluasi dan kesimpulannya terdapat di file Ipynb.
Source_Code_Project_VK__IF_45_DSIS_03_Kelompok_7_Road_Damages_Object_Detection
