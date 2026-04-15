# 2026.4.15
### 本文档是归档目录，保存了旧流程的脚本
  旧流程遗留（应归档）：
  - train.sh, train_fences.sh, train_qwen3vl.sh, train_stage1_grounding.sh, train_stage2_json.sh（被 2_stage_train/ 里的脚本取代）
  - api_annotate_v2.py（被 2_stage_train/api_annotate_stage12.py 取代）
  - prepare_data.py, prepare_finetune_data.py, prepare_stage2_json.py, prepare_train_v2_data.py（旧数据准备流程）
  - convert_labelme_to_grounding.py（旧 LabelMe 流程）
  - find_fence_violations.py, filter_all_fences_for_review.py, extract_compliant_fences.py（旧数据筛选）
  - fix_labelme_json.py（旧标注修复）
  - auto_label_compliant.py（旧合规打标）
  - label_ui.py, ui_review_fences.py, web_ui_review_fences.py（旧标注 UI）
  - strip_reasoning.py（旧数据清洗）
  - test_fence_violations_v1.py, test_regex.py（旧测试）
  - readme.md（内容已过时，被 CLAUDE.md 取代）
  - outputs/（测试输出不应在 scripts/ 下）


# 2026.4.14
# Step 1: 生成 Stage 1 grounding 数据（已完成，78 条）
  # data/grounding_stage1.jsonl ✓

  # Step 2: 生成 Stage 2 JSON 数据（已完成，108 条）
  # data/finetune_stage2_gt.jsonl ✓

  # Step 3: Stage 1 训练 + 自动合并 LoRA
  bash scripts/train_stage1_grounding.sh

  # Step 4: Stage 2 训练
  bash scripts/train_stage2_json.sh

  
  新增文件汇总：
```bash
  ┌─────────────────────────────────────────┬────────────────────────────────────────────────────────────────┐
  │                  文件                   │                              说明                               │
  ├─────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ scripts/convert_labelme_to_grounding.py │ LabelMe → grounding JSONL                                      │
  ├─────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ scripts/prepare_stage2_json.py          │ LabelMe → 真实坐标 JSON JSONL                                   │
  ├─────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ scripts/train_stage1_grounding.sh       │ Stage 1 训练 + LoRA 合并                                        │
  ├─────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ scripts/train_stage2_json.sh            │ Stage 2 训练                                                    │
  ├─────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ data/grounding_stage1.jsonl             │ 78 条 grounding 样本（围栏断口83、临边防护缺失25、围栏倒伏11）      │
  ├─────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ data/finetune_stage2_gt.jsonl           │ 108 条（62违规 + 35合规 + 11 grounding混入）                     │
  └─────────────────────────────────────────┴────────────────────────────────────────────────────────────────┘
```
  有一点值得注意：Fences_noncomlaint/ 里有 19 个文件被跳过（标注了人员/车辆等但没有违规标签），这些图片在 Stage 2
  里也没有被当合规样本使用。如果你想把它们加进来，可以手动补标或者单独处理。
