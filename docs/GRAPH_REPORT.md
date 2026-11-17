# Graph Report - AI_CAR  (2026-11-12)

## Corpus Check
- 6 files · ~22,196 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 240 nodes · 555 edges · 20 communities (8 shown, 12 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5694599`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pipeline.py
- test_pipeline_unit.py
- TestPhase2OptimizationSuite
- is_close_up_shot
- TestPipelinePostProcessing
- resolve_door_position_from_topology
- DamageResult
- map_damages_to_parts
- PartResult
- TestChangeDetectionSuite
- StandardPartStr
- 3. Hướng dẫn Chạy Nhanh (Quick Start)
- determine_car_orientation
- normalize_damage_class

## God Nodes (most connected - your core abstractions)
1. `PartResult` - 50 edges
2. `map_damages_to_parts()` - 47 edges
3. `DamageResult` - 43 edges
4. `TestPipelinePostProcessing` - 30 edges
5. `align_images_and_compare()` - 15 edges
6. `is_close_up_shot()` - 13 edges
7. `draw_annotations()` - 13 edges
8. `filter_parts_with_contextual_rules()` - 12 edges
9. `classify_shot_type()` - 12 edges
10. `TwoStagePipeline` - 12 edges

## Surprising Connections (you probably didn't know these)
- `3.3. Gọi trực tiếp `TwoStagePipeline` bằng Python` --references--> `TwoStagePipeline`  [INFERRED]
  README.md → ai/inference/pipeline.py
- `main()` --calls--> `TwoStagePipeline`  [EXTRACTED]
  random_demo.py → ai/inference/pipeline.py
- `main()` --calls--> `draw_annotations()`  [EXTRACTED]
  random_demo.py → ai/inference/pipeline.py
- `run()` --calls--> `PipelineResponse`  [EXTRACTED]
  ai/tests/test_pipeline_unit.py → ai/inference/pipeline.py

## Import Cycles
- None detected.

## Communities (20 total, 12 thin omitted)

### Community 0 - "pipeline.py"
Cohesion: 0.07
Nodes (17): _bbox_to_polygon(), calculate_center(), _calculate_hsv_red_ratio(), calculate_polygon_overlap_and_iou(), draw_annotations(), filter_parts_with_contextual_rules(), _get_unicode_font(), map_to_standard_part() (+9 more)

### Community 1 - "test_pipeline_unit.py"
Cohesion: 0.12
Nodes (8): align_images_and_compare(), calculate_bbox_iou(), ChangeDetectionResult, DamageComparisonItem, extract_features_and_compute_homography(), load_image_bgr(), visualize_change_detection(), warp_bbox_with_homography()

### Community 2 - "TestPhase2OptimizationSuite"
Cohesion: 0.12
Nodes (3): PipelineResponse, TestPhase2OptimizationSuite, run()

### Community 3 - "is_close_up_shot"
Cohesion: 0.16
Nodes (3): classify_shot_type(), determine_shot_type(), is_close_up_shot()

### Community 5 - "resolve_door_position_from_topology"
Cohesion: 0.15
Nodes (5): calculate_point_to_part_boundary_distance(), fallback_topology_routing(), infer_part_from_spatial_coordinates(), resolve_door_position_from_topology(), resolve_four_side_doors()

### Community 10 - "StandardPartStr"
Cohesion: 0.18
Nodes (3): detect_micro_features(), DisplayLabelStr, StandardPartStr

### Community 11 - "3. Hướng dẫn Chạy Nhanh (Quick Start)"
Cohesion: 0.20
Nodes (9): 1. Cấu trúc Thư mục Dự án (`AI_CAR/`), 2. Cài đặt Môi trường, 3.1. Chạy bộ kiểm thử đơn vị (Unit Tests), 3.2. Chạy trình diễn trên 8 ảnh mẫu (`random_demo.py`), 3.3. Gọi trực tiếp `TwoStagePipeline` bằng Python, 3.4. Khởi chạy FastAPI Gateway Server, 3. Hướng dẫn Chạy Nhanh (Quick Start), 4. Lưu ý Kỹ thuật & Kế hoạch Mở rộng Dataset Stage 2 (+1 more)

### Community 12 - "determine_car_orientation"
Cohesion: 0.29
Nodes (3): determine_car_orientation(), filter_false_positive_parts_by_orientation(), OrientationInfo

## Knowledge Gaps
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `map_damages_to_parts()` connect `map_damages_to_parts` to `pipeline.py`, `test_pipeline_unit.py`, `TestPhase2OptimizationSuite`, `is_close_up_shot`, `TestPipelinePostProcessing`, `resolve_door_position_from_topology`, `DamageResult`, `PartResult`, `StandardPartStr`, `determine_car_orientation`, `.test_semantic_hard_rule_lamp_broken_strictly_belongs_to_headlight`, `.test_physical_guard_dent_forbidden_on_den_xe_and_glass`, `.test_case_3_side_view_front_wheel_and_headlamp_forces_fender`, `.test_case_5_closeup_fender_scratch_context_awareness`, `.test_sanity_override_windshield_forbidden_for_sheet_metal_damages`, `.test_rear_bumper_seam_heuristic_separates_fender_and_rear_bumper`?**
  _High betweenness centrality (0.196) - this node is a cross-community bridge._
- **Should `pipeline.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07441016333938294 - nodes in this community are weakly interconnected._
- **Why does `PartResult` connect `PartResult` to `pipeline.py`, `test_pipeline_unit.py`, `TestPhase2OptimizationSuite`, `is_close_up_shot`, `TestPipelinePostProcessing`, `resolve_door_position_from_topology`, `DamageResult`, `map_damages_to_parts`, `determine_car_orientation`, `.test_semantic_hard_rule_lamp_broken_strictly_belongs_to_headlight`, `.test_physical_guard_dent_forbidden_on_den_xe_and_glass`, `.test_case_3_side_view_front_wheel_and_headlamp_forces_fender`, `.test_case_5_closeup_fender_scratch_context_awareness`, `.test_sanity_override_windshield_forbidden_for_sheet_metal_damages`, `.test_rear_bumper_seam_heuristic_separates_fender_and_rear_bumper`?**
  _High betweenness centrality (0.159) - this node is a cross-community bridge._
- **Should `test_pipeline_unit.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11724137931034483 - nodes in this community are weakly interconnected._
- **Why does `TwoStagePipeline` connect `pipeline.py` to `test_pipeline_unit.py`, `3. Hướng dẫn Chạy Nhanh (Quick Start)`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Should `TestPhase2OptimizationSuite` be split into smaller, more focused modules?**
  _Cohesion score 0.125 - nodes in this community are weakly interconnected._
- **Should `TestPipelinePostProcessing` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._