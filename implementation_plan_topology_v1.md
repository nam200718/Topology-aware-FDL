# Implementation Plan v2 (Updated)

Dựa trên [kết quả thực nghiệm post-midterm](file:///c:/Users/admin/.gemini/antigravity-ide/brain/5c0ca6a1-99fe-4b71-b0d7-34bf984f04f5/experiment_results.md) và câu trả lời cho Open Questions.

> **Quyết định đã xác nhận:**
> - Defense sweep test trên **Star + Ring + HE** (không chỉ HE) để so sánh tối ưu.
> - `hierarchical_partitioner.py` **giữ nguyên ở `temp/`**, không di chuyển vào `src/`.
> - Tất cả experiment chạy **50 rounds** trên CIFAR-10.

---

## Tổng quan Experiment Matrix

Toàn bộ thí nghiệm được chia thành 2 trục chính:
- **Trục 1**: Cách chia data — Random (1-Phase Dirichlet α=0.1) vs Hierarchical (2-Phase Dirichlet β=0.1, x=1.0)
- **Trục 2**: Có/Không có Byzantine attack + Defense

| # | Partition | Topologies | Defense | Attack | Output Folder |
|---|-----------|------------|---------|--------|---------------|
| **E1** ✅ done | Random (α=0.1) | Star FedAvg, Star Ditto, Star APFL, Ring, Gossip | None | None | `idea4_full_baseline/` |
| **E2** | Random (α=0.1) | Star FedAvg, Ring, Gossip, **HE Agg-Only, HE Ensemble** | None | None | `exp2_random_with_HE/` |
| **E3** | Hierarchical (β=0.1, x=1.0) | Star FedAvg, Ring, Gossip, HE Agg-Only, HE Ensemble | None | None | `exp3_hierarchical_all/` |
| **E4** | Random (α=0.1) | Star FedAvg, Ring, HE Ensemble | None vs Soft Cosine | label_flip 0-30% | `exp4_defense_random/` |
| **E5** | Hierarchical (β=0.1, x=1.0) | Star FedAvg, Ring, HE Ensemble | None vs Soft Cosine | label_flip 0-30% | `exp5_defense_hier/` |

> [!NOTE]
> E1 đã chạy xong nhưng **thiếu HE topology** (config có nhưng output chỉ có Star/Ring/Gossip). E2 bổ sung HE vào cùng setting Random partition.

---

## Phase 1: Baseline Comparison hoàn chỉnh + Accuracy Curve PNG

### Mục tiêu
1. Bổ sung HE topology vào baseline (E2).
2. Chạy cùng topologies trên Hierarchical partition (E3).
3. Export đồ thị accuracy curves (PNG) cho mỗi experiment.

---

### Bước 1.1: Fix export đồ thị PNG

#### [MODIFY] [run_baseline_comparison.py](file:///d:/UROP/Topology-aware-FDL/temp/run_baseline_comparison.py)

Sửa hàm `run_all()` phần plot (dòng 123-136) để tạo 2-subplot PNG:

```python
# Thay thế khối plot hiện tại bằng:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Subplot 1: Global Test Accuracy
for label, metrics in all_metrics.items():
    rounds = [m["round"] for m in metrics]
    accs = [m.get("test_accuracy", 0) for m in metrics]
    ax1.plot(rounds, accs, label=label, marker='o', markersize=2)
ax1.set_xlabel("Round"); ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Global Model Accuracy"); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

# Subplot 2: Ensemble/Personalized Accuracy (chỉ plot nếu có)
for label, metrics in all_metrics.items():
    ens_accs = [m.get("ensemble_test_accuracy", None) for m in metrics]
    if any(a is not None and a > 0 for a in ens_accs):
        rounds = [m["round"] for m in metrics]
        ax2.plot(rounds, [a or 0 for a in ens_accs], label=label, marker='s', markersize=2)
ax2.set_xlabel("Round"); ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Ensemble / Personalized Accuracy"); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "comparison_accuracy_curves.png"), dpi=150)
plt.close()
```

---

### Bước 1.2: Tích hợp Hierarchical Partitioner vào run_baseline_comparison.py

Thay vì sửa `BaseEngine` (ảnh hưởng code teammate), ta sẽ **inject partition data từ bên ngoài** bằng cách:

#### [MODIFY] [run_baseline_comparison.py](file:///d:/UROP/Topology-aware-FDL/temp/run_baseline_comparison.py)

Thêm logic vào `run_single_topology()`: sau khi engine được khởi tạo, **ghi đè** `engine.client_indices` bằng kết quả từ `partition_data_hierarchical()` nếu config yêu cầu.

```python
# Sau dòng: engine = EngineClass(main_config, topology, aggregator, device)
# Thêm:
if common_config_dict.get("use_hierarchical_partition", False):
    from temp.hierarchical_partitioner import partition_data_hierarchical
    hp_config = common_config_dict.get("hierarchical_partition_config", {})
    num_clusters = hp_config.get("num_clusters", 3)
    intra_alpha = hp_config.get("intra_alpha", 1.0)
    inter_alpha = hp_config.get("inter_alpha", 0.1)
    
    engine.client_indices, _ = partition_data_hierarchical(
        dataset=engine.train_dataset,
        num_clients=main_config.clients.num_clients,
        num_clusters=num_clusters,
        intra_alpha=intra_alpha,
        inter_alpha=inter_alpha,
        seed=main_config.env.seed,
    )
    engine.client_test_indices, _ = partition_data_hierarchical(
        dataset=engine.test_dataset,
        num_clients=main_config.clients.num_clients,
        num_clusters=num_clusters,
        intra_alpha=intra_alpha,
        inter_alpha=inter_alpha,
        seed=main_config.env.seed + 1,
    )
    # Rebuild ClientDataset caches and tensor indices
    from src.data.dataset import ClientDataset
    engine.client_train_datasets = {
        cid: ClientDataset(engine.train_dataset, idxs)
        for cid, idxs in engine.client_indices.items()
    }
    engine.client_test_datasets = {
        cid: ClientDataset(engine.test_dataset, idxs)
        for cid, idxs in engine.client_test_indices.items()
    }
    engine.client_test_indices_t = {
        cid: torch.tensor(idxs, dtype=torch.long, device=engine.device)
        for cid, idxs in engine.client_test_indices.items()
    }
    print(f"  [Hierarchical Partition] clusters={num_clusters}, x={intra_alpha}, β={inter_alpha}")
```

**Ưu điểm của cách này:** Không sửa bất kỳ file nào trong `src/` — chỉ thay đổi file trong `temp/`. Code teammate không bị ảnh hưởng.

---

### Bước 1.3: Tạo configs cho E2 và E3

#### [NEW] `temp/exp2_random_with_HE.yaml`

Giống `baseline_comparison.yaml` nhưng output vào folder riêng và **bao gồm HE**:

```yaml
experiment_type: comparison
num_rounds: 50

env:
  seed: 42
  dataset: cifar10
  output_dir: ./outputs/exp2_random_with_HE

clients:
  model_name: resnet9
  num_clients: 30
  local_lr: 0.01
  local_steps: 2
  batch_size: 32
  compute_optimization_mode: shared_backbone
  ensemble_weighting_mode: dynamic_confidence
  ensemble_distillation: true
  distillation_lambda: 0.5
  total_local_steps: 5
  loss_weight_beta: 1.0

non_iid:
  enabled: true
  alpha: 0.1

robustness:
  byzantine_type: none

# Không dùng hierarchical partition → dùng Random mặc định
use_hierarchical_partition: false

topologies:
  - type: star
    label: "Star (FedAvg)"
    params: { personalization_method: none }
  - type: ring
    label: "Ring"
    params: { personalization_method: none }
  - type: gossip
    label: "Gossip (k=3)"
    params: { degree_k: 3, personalization_method: none }
  - type: hier_agg
    label: "HE Aggregation-Only"
    params:
      num_clusters: 5
      cluster_by_label_dist: true
      use_ensemble: false
      hierarchical_ensemble: false
      compute_optimization_mode: "none"
      personalization_method: none
  - type: hier_ens
    label: "HE Ensemble (3-Tier)"
    params:
      num_clusters: 5
      cluster_by_label_dist: true
      personalization_method: none
```

#### [NEW] `temp/exp3_hierarchical_all.yaml`

**Giống hệt E2** nhưng bật hierarchical partition:

```yaml
# ... (giống E2, chỉ thay đổi 3 dòng)
env:
  output_dir: ./outputs/exp3_hierarchical_all

use_hierarchical_partition: true
hierarchical_partition_config:
  num_clusters: 3      # 30 clients / 3 = 10 clients per cluster
  intra_alpha: 1.0     # Moderate non-IID nội cụm (xem Phase 3 giải thích)
  inter_alpha: 0.1     # Strongly non-IID giữa cụm

topologies:
  # (giống E2)
```

---

### Bước 1.4: Cập nhật notebook

#### [MODIFY] [urop-topology-aware-fl.ipynb](file:///d:/UROP/Topology-aware-FDL/baseline/urop-topology-aware-fl.ipynb)

Thêm 2 cell mới cho E2 và E3:

```python
# Cell E2: Random Partition + All Topologies (including HE)
!python temp/run_baseline_comparison.py --config temp/exp2_random_with_HE.yaml
!mkdir -p outputs/exp2_random_with_HE
import shutil; from IPython.display import FileLink, display
shutil.make_archive('exp2_random_with_HE', 'zip', 'outputs/exp2_random_with_HE')
display(FileLink('exp2_random_with_HE.zip'))
```

```python
# Cell E3: Hierarchical Partition + All Topologies
!python temp/run_baseline_comparison.py --config temp/exp3_hierarchical_all.yaml
!mkdir -p outputs/exp3_hierarchical_all
import shutil; from IPython.display import FileLink, display
shutil.make_archive('exp3_hierarchical_all', 'zip', 'outputs/exp3_hierarchical_all')
display(FileLink('exp3_hierarchical_all.zip'))
```

### Verification Phase 1
- Chạy E2 locally (fast debug 2 rounds) → confirm HE topology chạy được + PNG xuất ra.
- Chạy E3 locally → confirm hierarchical partition inject thành công.

---

## Phase 2: Byzantine Defense trên Star / Ring / HE

### Mục tiêu
Test defense layer (`SoftRejectionAggregator`) trên 3 topology đại diện, với 2 cách chia data, sweep byzantine rate 0% → 30%.

### Thay đổi cần thực hiện

#### Bước 2.1: Mở rộng `run_baseline_comparison.py` hỗ trợ defense

#### [MODIFY] [run_baseline_comparison.py](file:///d:/UROP/Topology-aware-FDL/temp/run_baseline_comparison.py)

Thêm logic trong `run_single_topology()`: nếu topology params chứa `defense_mode != "none"`, sử dụng `SoftRejectionAggregator` thay vì `FedAvgAggregator`:

```python
# Thay thế đoạn init aggregator:
defense_mode = topo_config.get("params", {}).get("defense_mode", "none")
if defense_mode != "none":
    from src.defense.config import DefenseConfig
    from src.defense.aggregator import SoftRejectionAggregator
    defense_cfg = DefenseConfig(defense_mode=defense_mode)
    aggregator = SoftRejectionAggregator(defense_cfg)
else:
    aggregator = FedAvgAggregator()
```

Thêm hỗ trợ `byzantine_rates` sweep trong `run_all()`:

```python
def run_all(config_path: str):
    config = load_yaml_config(config_path)
    topologies = config.pop("topologies")
    byzantine_rates = config.pop("byzantine_rates", [0.0])
    
    for byz_rate in byzantine_rates:
        config["robustness"]["byzantine_rate"] = byz_rate
        for topo in topologies:
            res = run_single_topology(topo, config)
            # ... save with byz_rate in filename
```

#### Bước 2.2: Tạo configs cho E4 và E5

#### [NEW] `temp/exp4_defense_random.yaml`

```yaml
experiment_type: comparison
num_rounds: 50

env:
  seed: 42
  dataset: cifar10
  output_dir: ./outputs/exp4_defense_random

clients:
  model_name: resnet9
  num_clients: 30
  local_lr: 0.01
  local_steps: 2
  batch_size: 32
  compute_optimization_mode: shared_backbone
  ensemble_weighting_mode: dynamic_confidence
  ensemble_distillation: true
  total_local_steps: 5

non_iid:
  enabled: true
  alpha: 0.1

robustness:
  byzantine_type: label_flip

use_hierarchical_partition: false
byzantine_rates: [0.0, 0.1, 0.2, 0.3]

topologies:
  # No Defense
  - type: star
    label: "Star (No Defense)"
    params: { personalization_method: none, defense_mode: none }
  - type: ring
    label: "Ring (No Defense)"
    params: { personalization_method: none, defense_mode: none }
  - type: hier_ens
    label: "HE (No Defense)"
    params: { num_clusters: 5, cluster_by_label_dist: true, defense_mode: none }
  # With Defense
  - type: star
    label: "Star (Soft Cosine)"
    params: { personalization_method: none, defense_mode: soft_cosine }
  - type: ring
    label: "Ring (Soft Cosine)"
    params: { personalization_method: none, defense_mode: soft_cosine }
  - type: hier_ens
    label: "HE (Soft Cosine)"
    params: { num_clusters: 5, cluster_by_label_dist: true, defense_mode: soft_cosine }
```

#### [NEW] `temp/exp5_defense_hier.yaml`

Giống E4 nhưng bật hierarchical partition:

```yaml
# ... giống E4, chỉ thay:
env:
  output_dir: ./outputs/exp5_defense_hier

use_hierarchical_partition: true
hierarchical_partition_config:
  num_clusters: 3
  intra_alpha: 1.0
  inter_alpha: 0.1
```

### Verification Phase 2
- Chạy E4 locally (fast debug: 2 rounds, 1 byz rate) → confirm defense aggregator tích hợp + byzantine clients hoạt động.
- Chạy full E4, E5 trên Kaggle.

---

## Phase 3: Khuyến nghị Hyper-parameter

### Chọn `x` (intra_alpha) nào là "thực tế nhất"?

Dựa trên phân tích heatmap từ [Idea 1](file:///d:/UROP/Topology-aware-FDL/outputs/idea1_partitioner):

| x | Đặc tính | Tình huống thực tế |
|---|----------|--------------------|
| 0.1 | Client chỉ có 1-2 classes | 🔴 Quá cực đoan. Hiếm khi xảy ra ngoài lab. |
| **1.0** | Client thiên vị vài class nhưng vẫn nhìn thấy nhiều class khác | ✅ **Phù hợp nhất.** Giống camera giám sát: mỗi camera thiên vị vài loại xe/người nhưng vẫn thấy các loại khác. |
| 5.0 | Gần đồng đều trong cụm | 🟡 Quá dễ. Không đủ challenge cho mô hình. |
| 50.0 | Hoàn toàn IID trong cụm | ❌ Phi thực tế cho non-IID research. |

> [!IMPORTANT]
> **Khuyến nghị chính: `x = 1.0`** cho tất cả experiment Hierarchical Partition (E3, E5).
> 
> Lý do:
> 1. **Thực tế**: Mô phỏng đúng kịch bản FL trong y tế (bệnh viện cùng thành phố thấy bệnh tương tự nhưng không giống hệt) hoặc IoT (thiết bị cùng khu vực thu thập data tương tự).
> 2. **Phân biệt rõ**: Với `x=1.0`, clients cùng cluster vẫn có bias chung nhưng đủ khác để personalization có ý nghĩa.
> 3. **Fair comparison**: `β=0.1` (inter-cluster) giống hệt `α=0.1` (Random partition) → cùng mức "non-IID tổng thể", chỉ khác ở cấu trúc phân cấp.

### Có cần tinh chỉnh config khác không?

| Parameter | Giá trị hiện tại | Khuyến nghị | Lý do |
|-----------|-------------------|-------------|-------|
| `num_rounds` | 50 | **50** (giữ nguyên) | Đủ cho Star hội tụ. Ring/Gossip chưa hội tụ nhưng vẫn thấy trend. |
| `num_clients` | 30 | **30** (giữ nguyên) | Đủ cho 3 clusters × 10 clients, hoặc 5 clusters × 6 clients. |
| `num_clusters` (HE) | 5 | **3** khi dùng Hierarchical Partition | 30/3 = 10 clients/cluster (cân đối). Nếu 5 clusters thì chỉ 6 clients/cluster, hơi ít cho statistical significance. |
| `local_lr` | 0.01 | 0.01 (giữ nguyên) | Hoạt động ổn với ResNet9 trên CIFAR-10. |
| `byzantine_type` | label_flip | `label_flip` | Đơn giản, dễ hiểu, phù hợp cho demo defense. |

> [!WARNING]
> **Lưu ý khi chuyển `num_clusters` từ 5 → 3**: Config HE trong E2 (Random partition) vẫn dùng `num_clusters: 5` (giống baseline), nhưng E3/E5 (Hierarchical partition) nên dùng `num_clusters: 3` để match với `partition_data_hierarchical(num_clusters=3)`. Nếu muốn fair comparison giữa E2 và E3, nên thống nhất `num_clusters = 3` cho cả hai.

---

## Thứ tự thực hiện

```mermaid
graph LR
    A["Phase 1.1<br/>Fix plot PNG"] --> B["Phase 1.2<br/>Inject Hierarchical Partition"]
    B --> C["Phase 1.3<br/>Create E2, E3 configs"]
    C --> D["Phase 1.4<br/>Update notebook"]
    D --> E["Verify locally<br/>(fast debug)"]
    E --> F["Phase 2.1<br/>Add defense to script"]
    F --> G["Phase 2.2<br/>Create E4, E5 configs"]
    G --> H["Verify locally"]
    H --> I["Push to GitHub<br/>Run on Kaggle"]
```

| Phase | Độ khó | Thời gian code | GPU time (Kaggle) |
|-------|--------|----------------|-------------------|
| 1.1: Fix plot | 🟢 | ~10 phút | 0 |
| 1.2: Inject partition | 🟡 | ~20 phút | 0 |
| 1.3-1.4: Configs + notebook | 🟢 | ~15 phút | 0 |
| Verify locally | 🟢 | ~10 phút | 0 |
| 2.1: Defense integration | 🟡 | ~30 phút | 0 |
| 2.2: E4, E5 configs | 🟢 | ~10 phút | 0 |
| **Kaggle E2** | — | — | ~2-3 giờ (5 topologies × 50 rounds) |
| **Kaggle E3** | — | — | ~2-3 giờ |
| **Kaggle E4** | — | — | ~8-12 giờ (6 topologies × 4 byz_rates × 50 rounds) |
| **Kaggle E5** | — | — | ~8-12 giờ |

> [!CAUTION]
> E4 và E5 rất nặng (6 topologies × 4 byzantine_rates = 24 runs × 50 rounds mỗi run). Cần chia thành nhiều Kaggle session hoặc giảm `byzantine_rates` xuống `[0.0, 0.2]` nếu thiếu quota.
