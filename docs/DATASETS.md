# Datasets: provenance, licences, and how to obtain them

**This repository does not redistribute any image data.** All four datasets remain
under their original licences and must be downloaded from their original sources.
What this repository provides is the **manifest** (`data/manifest.json`), the
**frozen splits**, and the **audit records**, which together specify exactly which
images were used and how they were partitioned.

---

## 1. Summary

| Key | Dataset | Images used | Classes | Native size | Imbalance | Licence |
|---|---|---|---|---|---|---|
| `neu_cls` | NEU Surface Defect Database | 1,799 | 6 | 200x200 | 1.0x | Free for research |
| `casting` | Casting Product Image Data | 7,284 | 2 | 300x300 | 1.4x | **CC BY-NC-ND 4.0** |
| `gc10det` | GC10-DET | 2,312 | 10 | 2048x1000 | 21.0x | Free for research |
| `magnetic_tile` | Magnetic Tile Defect | 2,611 | 6 | ~120-600 px | 28.5x | Free for research |

Counts are **after global deduplication**, which removed 142 duplicate images in
total (1 from `neu_cls`, 64 from `casting`, 77 from `magnetic_tile`, 0 from
`gc10det`). See `data/audit/audit_report.json` for the duplicate groups.

---

## 2. NEU-CLS (`neu_cls`)

- **Full name:** NEU Surface Defect Database (hot-rolled steel strip)
- **Source:** Northeastern University, China
- **URL:** http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
- **Reference:** Song, K., Yan, Y. *A noise robust method based on completed local
  binary patterns for hot-rolled steel strip surface defects.* Applied Surface
  Science 285, 858-864 (2013). doi:10.1016/j.apsusc.2013.09.002
- **Classes (6):** `crazing`, `inclusion`, `patches`, `pitted_surface`,
  `rolled-in_scale`, `scratches`
- **Used here:** 1,799 images of 200x200 px, greyscale. Level cap 180.

**Deduplication note.** The audit found 1 duplicate. More importantly, the
NEU-CLS classification set and the `neu_surface` set circulating in the literature
are **the same 1,800 images**. They must not be treated as two independent datasets.
This repository ships only one copy under the key `neu_cls`.

---

## 3. Casting (`casting`)

- **Full name:** Casting Product Image Data for Quality Inspection
- **Source:** Kaggle, by Ravirajsinh Dabhi
- **URL:** https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
- **Year:** 2020 (initial release 2020-01-23)
- **Licence:** **CC BY-NC-ND 4.0** - Attribution-NonCommercial-NoDerivatives
- **Classes (2):** `def_front` (defective), `ok_front` (ok)
- **Used here:** 7,284 images of 300x300 px, greyscale. Level cap 512.

**Licence caution.** This dataset is **NonCommercial** and **NoDerivatives**. We
therefore publish only the manifest and results, never the images, and we do not
redistribute any modified copy. If your intended use is commercial, or if you need
to redistribute a derived version of the images, check the licence terms first.

**Construction note.** The original Kaggle directory holds two copies of the same
product distribution, `casting_data/` (453 + 262 + 3758 + 2875 = 7,348 images) and
`casting_512x512/` (781 + 519 = 1,300 images). The latter is a resampled copy rather
than additional samples and is excluded in full. After deduplication of the
`casting_data/` tree (3,137 - 64 = 3,073 `ok_front`), 7,284 images remain.

---

## 4. GC10-DET (`gc10det`)

- **Full name:** GC10-DET, a metallic surface defect benchmark
- **Reference:** Lv, X., Duan, F., Jiang, J., Fu, X., Gan, L. *Deep Metallic Surface
  Defect Detection: The New Benchmark and Detection Network.* Sensors 20(6), 1562
  (2020). doi:10.3390/s20061562
- **Classes (10):** `1_chongkong`, `2_hanfeng`, `3_yueyawan`, `4_shuiban`,
  `5_youban`, `6_siban`, `7_yiwu`, `8_yahen`, `9_zhehen`, `10_yaozhe`
- **Used here:** 2,312 images, native 2048x1000 px (2:1), 21x imbalance. Level cap 18.

**Resolution note.** This dataset uses 256x256 network input rather than 224x224
because its native aspect ratio is 2:1, and a square 224 crop would discard the
peripheral regions where many defects lie. The processing is "short side to 256,
then centre crop". This makes `gc10det` not entirely comparable in resolution with
the other three datasets; the paper records this as limitation L2.

---

## 5. Magnetic Tile Defect (`magnetic_tile`)

- **Full name:** Magnetic Tile Defect dataset
- **Reference:** Huang, Y., Qiu, C., Yuan, K. *Surface defect saliency of magnetic
  tile.* The Visual Computer 36(1), 85-96 (2020). doi:10.1007/s00371-018-1588-5
- **Classes (6):** `MT_Blowhole`, `MT_Break`, `MT_Crack`, `MT_Fray`, `MT_Free`,
  `MT_Uneven`
- **Used here:** 2,611 images, variable native size (~120-600 px), 28.5x imbalance.
  Level cap 38.

**Construction note.** The raw audit reports seven labels; we use six. The seventh,
`Magnetic-Tile-Defect`, contains only 2 images, which is below the
database-construction threshold `MIN_CLASS = 20`, and is removed wholesale. This
accounts for exactly those 2 images (2,690 - 77 - 2 = 2,611).

---

## 6. Datasets considered and excluded

Two widely used industrial benchmarks were deliberately excluded:

- **KolektorSDD** (Tabernik et al., 2020, doi:10.1007/s10845-019-01476-x)
- **MVTec AD** (Bergmann et al., 2019, doi:10.1109/CVPR.2019.00982)

Both are predominantly oriented toward **segmentation or unsupervised anomaly
detection**, and therefore fall outside the scope of a **class-level** data-efficiency
study, which requires per-class labelled classification targets.

---

## 7. How to reconstruct the exact image set

1. Download each dataset from the source above.
2. Arrange each under a single root directory.
3. Edit the `root` field for each dataset in `data/manifest.json` to point at your
   local copy. The `items` lists - `[relative_path, label_index]` - are the
   authoritative record of which images were used and what label each has.
4. To redo deduplication and auditing from scratch:

```bash
python code/audit_datasets.py --root <your_root> --out audit_out/
python code/build_dataset.py  --root <your_root> --out <bench_dir>
```

`build_dataset.py` performs the MD5-content deduplication described in the paper and
writes the manifest. `audit_datasets.py` produces the counts, size histograms,
duplicate groups and corrupt-file list.

**Important.** Deduplication must happen **before** splitting. If two copies of the
same image land in the training and test sets respectively, the measured test
accuracy is contaminated. This is not hypothetical here: the duplicates in `casting`
cross the dataset's official train/test boundary.
