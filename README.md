# Ensemble Learning Lab (Iris Classification)

โปรเจกต์ทดลองและเปรียบเทียบเทคนิค **Ensemble Learning** ด้วย Python บนชุดข้อมูล Iris (150 ตัวอย่าง, 4 features, 3 classes, แบ่ง Train 80% / Test 20% แบบ stratified ที่ `random_state=42`) โดยสร้างอัลกอริทึม Bagging และ Multi-class AdaBoost (SAMME) ขึ้นเองจากพื้นฐานและวัดผลทั้ง **Accuracy**, **Training Time**, และ **Inference Time** (ค่ามัธยฐานจาก 5 รอบ)

## ภาพรวมโมเดลที่เปรียบเทียบ (Models Compared)

1. **Single Decision Tree (`DecisionTreeClassifier`)**: โมเดลต้นไม้เดี่ยวใช้เป็น baseline
2. **Custom Bagging — Parallel (`CustomBaggingClassifier`)**: สุ่มตัวอย่างแบบ Bootstrap (sampling with replacement) แล้วฝึก Decision Tree หลายต้นแบบขนานด้วย `joblib.Parallel` (`delayed` + `SeedSequence`) และตัดสินผลด้วย Majority Voting
3. **Custom AdaBoost SAMME — Sequential (`CustomAdaBoostSAMME`)**: ฝึก Decision Stump (`max_depth=1`) ตามลำดับ พร้อมปรับน้ำหนักตัวอย่างด้วยสูตร SAMME ($\alpha = \ln\frac{1 - \text{err}}{\text{err}} + \ln(K - 1)$) และตัดสินผลด้วย Weighted Voting

## โครงสร้างไฟล์ (Project Structure)

```text
.
├── ensemble_lab.py          # โมดูลหลัก: โหลดข้อมูล Iris, CustomBaggingClassifier, CustomAdaBoostSAMME, benchmark และ formatter
├── plot_results.py          # สคริปต์รัน scaling benchmark (n=100..500), สร้างกราฟ 3 ไฟล์ (.png) และซิงก์ out.txt
├── test_ensemble_lab.py     # ชุดทดสอบ unittest (unit, regression และ characterization tests; รันได้แบบออฟไลน์ 100%)
├── requirements.txt         # รายการ Python dependencies พร้อม pin เวอร์ชันตรงตามสภาพแวดล้อมจริง
├── out.txt                  # ตารางสรุปผลการทดลองล่าสุดที่ n_estimators=500 (median of 5 runs)
├── estimators_scaling.png   # กราฟ 2x2 แสดง Scaling ของ Accuracy, Train/Inference Time และ Parallel Speedup (n=100..500)
├── ensemble_comparison.png  # กราฟแท่งเปรียบเทียบ 3 โมเดลหลักที่ n_estimators=500
├── decision_boundaries.png  # กราฟ Decision Boundaries (2D Petal slice จากโมเดล 4 features)
├── .gitignore               # รายการไฟล์ cache/environment ที่ไม่ต้องติดตามใน Git
└── README.md                # เอกสารอธิบายโปรเจกต์และคำสั่งใช้งาน
```

## การติดตั้งและคำสั่งใช้งาน (Setup & Commands)

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. รันการทดลองหลัก (CLI Benchmark ที่ `n_estimators=500`)
```bash
python ensemble_lab.py
```

### 3. สร้างกราฟเปรียบเทียบ (`.png`) และอัปเดต `out.txt`
```bash
python plot_results.py
```
รองรับตัวเลือกเสริม:
- `--show`: เปิดหน้าต่างแสดงกราฟ `estimators_scaling.png` แบบ interactive
- `--repeats <N>`: กำหนดจำนวนรอบในการวัดเวลา median (ค่าเริ่มต้น `5`)

### 4. รันชุดทดสอบ (Unit & Regression Tests)
```bash
python -m unittest -v test_ensemble_lab.py
```
รองรับการรันแบบออฟไลน์ 100% (PowerShell):
```powershell
$env:HF_HUB_OFFLINE="1"; $env:HF_DATASETS_OFFLINE="1"; python -m unittest -v test_ensemble_lab.py
```

### 5. ตรวจสอบ Syntax
```bash
python -m py_compile ensemble_lab.py plot_results.py test_ensemble_lab.py
```

## สรุปผลการทดลองล่าสุด (`out.txt`, `n_estimators=500`, median of 5 runs)

- **Dataset Source:** Hugging Face datasets (`scikit-learn/iris`) — fallback อัตโนมัติไป `sklearn.datasets.load_iris()` หากออฟไลน์
- **Hardware / Warmup:** 12 CPU cores (Joblib pool warmup `4.5915 s` across 12 workers, แยกออกจากเวลา `fit`)

| Model | Estimators | Accuracy | Training Time (s) | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Single DecisionTree | 1 | 93.33% | 0.001835 | 0.000275 |
| Custom Bagging (Parallel) | 500 | 96.67% | 0.290992 | 0.096874 |
| Custom AdaBoost SAMME (Sequential) | 500 | 93.33% | 1.093950 | 0.109270 |

