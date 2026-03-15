# Work Log

## 2026-03-13

### HOT3D テストデータ修正

**問題**: `hot3d_test_raw.pt` が空 (テスト参加者 P0004/P0005/P0006/P0008/P0016/P0020 の `mano_hand_pose_trajectory.jsonl` が全て空。HOT3DベンチマークではテストGTが非公開)

**対応**: train cache (32054クリップ) を 90/10 ランダム分割 (seed=42)
- `hot3d_train_cache_split.pt`: 28849 クリップ (train用)
- `hot3d_test_cache.pt`: 3205 クリップ (val用)
- `config/motion_vqvae_hot3d.yaml` の `cache_pt` を `hot3d_train_cache_split.pt` に変更

---

### データセット間スケール調査

**raw kp3d のスケール比較** (joint 37→48 = RH wrist→mid-finger tip, 500サンプル平均):

| データセット | raw joints | wrist→midtip | pelvis→head | 推定単位 |
|---|---|---|---|---|
| **EE4D** | (T, 154, 3) | 0.310 | 0.582 | 不明 (154 joints全非ゼロ) |
| **A101** | (T, 154, 3) | 0.279 | 0.560 | 不明 (62 非ゼロ, 92 ゼロ) |
| **HOT3D** | (T, 42, 3) | 0.122 | N/A (hand-only) | m (MANOそのまま) |

**precompute 後 (cache)**: `process_file()` で tgt_offsets 骨格正規化 (`scale_rt = tgt_leg_len / src_leg_len`) が入るため、EE4D と A101 はスケール統一される。ただし HOT3D は hand-only パイプラインで body 骨格正規化なし → MANO 生スケール (m) のまま。

**結論**: 調査の結果、HOT3D hand scale は問題なし。
- tgt_offsets の hand bone lengths と MANO の hand bone lengths はほぼ同スケール (ratio ≈ 1.03)
- cache 後の RIC (wrist-relative finger positions) も EE4D と HOT3D で mean abs ≈ 0.054 で一致 (ratio 0.998)
- range の違い (EE4D [-1,1] vs HOT3D [-0.2,1]) は rot6d identity placeholder と body=zeros の影響

### cache 再生成 (clip21, base15, handroot)

- val (完了): ee4d_val 82,410件 (4.8G) + a101_val 41,319件 (2.4G)
- train (実行中): ee4d_train 268,567件 (~46%), a101_train 122,340件 (~99%)
- val cache を結合して `mix_val_cache_clip21_base15_handroot_new.pt` (7.2G, 123,729件) を作成
- val データで仮学習開始: `config/mix3_valtest.yaml`, GPU 0, wandb run `u2rfhd12`

### Hand Joint Scale 比較 (EE4D vs A101 vs HOT3D cache)

**スクリプト**: `compare_hand_scale.py`
**対象**: `*_train_cache_clip21_base15_handroot.pt` (EE4D/A101) + `hot3d_train_cache.pt` (HOT3D)
**手法**: RIC finger part (index 18:63 LH, 63:108 RH) を reshape (T, 15, 3) して wrist相対の L2 ノルムを計算

| Dataset | Clips | LH mean | LH std | LH median | RH mean | RH std | RH median |
|---------|-------|---------|--------|-----------|---------|--------|-----------|
| EE4D    | 268,567 | 0.10487 | 0.02133 | 0.10595 | 0.10321 | 0.02071 | 0.10416 |
| A101    | 122,340 | 0.10921 | 0.02475 | 0.10595 | 0.10566 | 0.02355 | 0.10416 |
| HOT3D   |  32,054 | 0.09549 | 0.02368 | 0.09611 | 0.10344 | 0.02728 | 0.10240 |

Mean hand "size" per clip (avg over 15 joints):

| Dataset | LH mean | LH std | RH mean | RH std |
|---------|---------|--------|---------|--------|
| EE4D    | 0.10487 | 0.00229 | 0.10321 | 0.00219 |
| A101    | 0.10921 | 0.00235 | 0.10566 | 0.00303 |
| HOT3D   | 0.09549 | 0.00457 | 0.10344 | 0.00719 |

**結論**:
- EE4D と A101 はスケールがほぼ一致 (差 ~4%)
- HOT3D LH は EE4D より約 9% 小さい, RH はほぼ同等
- HOT3D は LH/RH の非対称性が大きい (std も大きい) → 個人差 or データ品質
- 全体的に 3 データセットのスケール差は ~10% 以内で概ね揃っており、混合学習に大きな問題はないと判断

---

## 2026-03-11

### Bone Length Loss 追加
- **ファイル**: `src/evaluate/utils.py`, `src/model/vqvae.py`, `src/train/utils.py`, `config/mix_clip21_handroot_tokensep_reg_joints.yaml`
- **内容**: Shape My Moves (CVPR 2025) に倣い、bone length loss を追加。GT と recon の骨長偏差を MSE でペナルティ
- **詳細**:
  - `get_bone_pairs()` / `compute_bone_lengths()` ユーティリティ追加
  - H2VQ に `alpha_bone_length` パラメータ追加、kinematic chain から bone_pairs を `__init__` 時に生成
  - `alpha_joints > 0` or `alpha_bone_length > 0` のとき joints 復元を実行（既存フラグと共有）
  - 52 joints (no tips) / 62 joints (with tips) 両対応

### Fix: recover_from_ric 二重逆回転バグ修正
- **ファイル**: `src/evaluate/utils.py`
- **問題**: `kp3d2motion_rep.py` のエンコードが `r_rot_inv` (R^{-1}) で回転しているのに、`recover_from_ric` / `recover_root_rot_pos` のデコードも `qinv(r_rot_quat)` (R^{-1}) で回転していた → 二重逆回転
- **原因**: オリジナルHumanML3D (`motion_representation.py`) ではエンコードに `r_rot` (R) を使用 → デコードの `qinv` で正しく R^{-1}(R(pos))=pos。`kp3d2motion_rep.py` でエンコードを `r_rot_inv` に変更した際、デコード側を更新し忘れた
- **修正**: `recover_root_rot_pos` と `recover_from_ric` で `qrot(qinv(r_rot_quat), ...)` → `qrot(r_rot_quat, ...)` に変更
- **影響**: joints_loss の学習ではGT/pred両方に同じ関数を通すため相殺されていたが、可視化・推論時の位置復元が壊れていた

### Motion Diffusion: ε-prediction → x0-prediction + Geometric Losses

**Changed files:**
- `src/model/motion_diffusion.py` - `prediction_type` param ("x0"|"eps"), x0-pred forward/sampling, velocity loss, foot contact loss
- `src/evaluate/evaluator_diffusion.py` - `ddim_denoise_from_t()` x0-pred 分岐対応 (旧ckpt互換)
- `src/train/train_diffusion.py` - 新パラメータ渡し、norm stats条件拡張、loss logging追加
- `config/mix3_diffusion_uncond.yaml` - `prediction_type: x0`, `velocity_loss: true`, `foot_contact_loss: true`

## 2026-03-06

### DDPM Diffusion 実装 + Token Separation for Flow/Diffusion

**Changed files:**
- `src/model/vqvae.py` - DDPM実装: cosine/linear beta schedule, ε-prediction学習, DDIM sampling。`_flow_cond_from_ids()` を4-codebook (BR/BL/HR/HL) 対応に拡張。`_forward_flow()` / `sample_from_ids()` を indices dict ベースにリファクタ。`_sample_ddpm()` メソッド追加
- `src/train/utils.py` - `diffusion_timesteps`, `diffusion_schedule` パラメータ追加
- `src/train/train.py` - wandb Video upload (eval vis mp4), 初回eval時にもvis実行, `import glob` 追加
- `src/evaluate/evaluator.py` - `sample_from_ids()` に indices dict を渡すよう変更
- `config/motion_vqvae_mix_clip21_handroot_diffusion.yaml` - 新規: DDPM diffusion config (cosine schedule, 1000 timesteps, DDIM sampling)
- `config/mix_clip21_handroot_tokensep_reg.yaml` - 新規: Regressor + token_sep config
- `config/motion_vqvae_mix_clip21_handroot_flow.yaml` - token_separation 追加
- `config/motion_vqvae_mix_clip21_handroot_flow_sep.yaml` - token_separation 追加

**Training launched (4 jobs):**
- GPU 1: Regressor (dual decoder + token_sep + three_decoders)
- GPU 2: Flow baseline (single FlowDecoder + token_sep)
- GPU 3: Flow decoder_separate (body/hand別FlowDecoder + token_sep)
- GPU 5: Diffusion DDPM (cosine schedule + DDIM sampling + token_sep)

### Decoder汎用化: decoder_type (regressor / flow / diffusion) 統合

**Changed files:**
- `src/model/vqvae.py` - FlowDecoder1D等のコンポーネント追加（timestep_embedding, RotaryEmbedding, FlowCrossAttn, AdaLN, DiTBlock, FlowDecoder1D）。H2VQに `decoder_type` パラメータ追加、flow初期化分岐、`_forward_flow()`, `sample_from_ids()` メソッド追加。既存Attnにrope対応
- `src/train/utils.py` - `build_model_from_args()` にflow関連パラメータ（flow_model_dim, flow_depth等）を追加
- `src/train/train.py` - `_is_flow` フラグで学習ループ分岐、`freeze_encoder` 対応、flow loss ログ追加、ckpt `strict=False` ロード対応
- `src/evaluate/evaluator.py` - flow時に `model.sample_from_ids()` でODEサンプリング復元
- `src/model/vqvae_flow.py` - 非推奨コメント追加

**後方互換:** decoder_type デフォルト "regressor" → 既存ckpt完全互換

### Hand-only / Body-only decoder eval metrics

**Changed files:**
- `src/model/vqvae.py` - eval時（`use_three_decoders=True` & not training）に `_decode_hand_only` / `_decode_body_only` を実行し、`losses` に `recon_hand_only` / `recon_body_only` を追加
- `src/evaluate/evaluator.py` - hand-only decoder の出力から hand joints を復元し PA/WA/W-MPJPE を計算。`EVAL/PA_MPJPE/lh_handonly(mm)` 等の metrics を wandb に追加。body-only も同様。`use_three_decoders=False` 時はスキップ

### Body-relative hand root + Hand Trajectory Token

**Changed files:**
- `src/dataset/kp3d2motion_rep.py` - `_compute_hand_root()` を body-relative に変更。wrist vel は base_idx (head) 基準の相対位置の速度（base-local frame）、wrist rot6d は base joint に対する相対回転
- `src/model/vqvae.py` - `use_hand_traj_token` パラメータ追加。hand_root 9D を hand encoder 入力から分離し、専用の小型 CNN encoder (`encHT`, width=64, depth=2) + 専用 codebook (`qHT`) で量子化。LH/RH shared。traj token をデコーダ入力に結合
- `src/train/utils.py` - `build_model_from_args()` に `use_hand_traj_token` 追加
- `src/train/train.py` - HT codebook の usage/perplexity/commit loss ログ追加

### Data pipeline: clip_len=21, overlap=10 の raw pt 作成

- `data/ee4d_train_raw_clip21_ov10.pt` / `ee4d_val_raw_clip21_ov10.pt` - EE4D 10fps
- `data/assembly101_train_raw_clip21_ov10_ds3.pt` / `assembly101_val_raw_clip21_ov10_ds3.pt` - A101 30fps→10fps (ds3)
- Body-relative handroot cache を precompute 中: `*_cache_clip21_base15_handroot.pt`

### Config

- `config/motion_vqvae_mix_clip41_ov20_tokensep_joints.yaml` - tokensep + three decoders + joints loss の ablation config

---

### Hand Root + Token Separation + Three-Decoder Architecture

**Changed files:**
- `src/dataset/kp3d2motion_rep.py` - Added `_compute_hand_root()` for per-hand 9D root (wrist vel + rot6d). Added `compute_hand_root` param to `process_file()` and `kp3d_to_motion_rep()`. Returns `(data, lh_root, rh_root)` when enabled.
- `src/dataset/dataloader.py` - Added `hand_root: bool` param to `MotionDataset`. When True, prepends hand root (18D) to hand tensor.
- `precompute.py` - Added `--hand-root` CLI flag. Passes through to `MotionPrecomputer` and prepends hand root to cached hand tensors.
- `src/model/vqvae.py` - Major updates:
  - `_build_flip_sign()`: supports hand root 9D flip signs
  - `H2VQ.__init__()`: new params `hand_root`, `use_fuse`, `use_token_separation` (4 codebooks: BR/BL/HR/HL), `use_three_decoders` (body-only + hand-only + full)
  - `_split_hands_input()` / `_reassemble_hand_output()`: handle hand root prefix
  - `_quantize_with_separation()`: 4-codebook quantization with split root/local tokens
  - `_decode()` refactored to `_run_decoder()`, added `_decode_body_only()`, `_decode_hand_only()`
  - `decode_from_ids()`: supports both 2-codebook (idxH/idxB) and 4-codebook (idxBR/idxBL/idxHR/idxHL) modes, with mode param ("full"/"body_only"/"hand_only")
  - `forward()`: three-decoder masking (hand-masked → body dec + full dec; body-masked → hand dec + full dec)
- `src/train/utils.py` - `build_model_from_args()` passes all new params via `getattr` with backward-compat defaults. Auto-computes `hand_in_dim` from `hand_root` flag.
- `src/util/utils.py` - `compute_part_losses()` accepts `hand_root_dim` param for correct hand block offsets
- `src/train/train.py` - Dataset `hand_root` param, variable normalization dims, 4-codebook logging, per-decoder loss logging
- `src/evaluate/utils.py` - `reconstruct_623_from_body_hand()` strips hand root before reassembly via `hand_root_dim` param
- `src/evaluate/evaluator.py` - Variable body/hand dims, 4-codebook stats, hand root dim passthrough
- `config/motion_vqvae_mix_clip41_ov20.yaml` - Added all new params with backward-compat defaults
- `config/motion_vqvae_mix_clip41_ov20_handroot_tokensep.yaml` - New example config with all features enabled

**Backward Compatibility:**
- All new features behind config flags defaulting to existing behavior
- `hand_root: false` → existing 360/480D hand
- `use_token_separation: false` → existing 2 codebooks (qB, qH)
- `use_three_decoders: false` → existing single/dual/tri decoder
- `use_fuse: true` → existing fusion
- Old precomputed `.pt` files work unchanged
- Old configs produce identical behavior

## 2026-03-04

### alpha_hand = 5.0 + Metrics Overlay on MP4

**Changed files:**
- `config/motion_vqvae*.yaml` (8 files) - `alpha_hand: 1.0` -> `alpha_hand: 5.0`
- `src/evaluate/evaluator.py` - Added `_compute_sample_metrics()` for per-sample WA-MPJPE, W-MPJPE (static), PA-MPJPE (dynamic per-frame)
- `src/evaluate/vis.py` - Added `metrics_overlay` param to `visualize_two_motions`; static metrics at top, dynamic at bottom; view-aware (rh view shows only rh metrics)
- `src/evaluate/evaluator_flow.py` - Import `_compute_sample_metrics`, pass overlay to vis calls

**Details:**
- Metrics overlay skipped when `only_gt=True`
- Existing callers (inference_atomic.py etc.) unaffected (default `metrics_overlay=None`)

### World-space Joints L2 Loss 追加

**Changed files:**
- `src/model/vqvae.py` - `H2VQ.__init__()` に `alpha_joints`, `alpha_joints_hand`, `base_idx`, `hand_local` パラメータ追加。`forward()` に world-space joints L2 loss 計算ブロック追加（lazy import で循環参照回避）
- `src/train/utils.py` - `build_model_from_args()` に新4パラメータを `getattr` で渡すよう追加
- `config/*.yaml` (8 files) - 全configに `alpha_joints: 0.0`, `alpha_joints_hand: 0.0` 追加

**Details:**
- `reconstruct_623_from_body_hand` + `recover_from_ric` で recon/target を world-space joints に変換し MSE 計算
- `alpha_joints`: 全 joints の L2、`alpha_joints_hand`: hand joints (22:) の追加 L2
- デフォルト 0.0 で既存挙動と完全互換
- wandb に `joints_loss`, `joints_loss_hand` がログされる
- 勾配フロー確認済み
