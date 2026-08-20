"""Evaluation: KITTI-style eval, YOLO metrics, diagnostics, reports."""

from fish_monitoring.eval.kitti_eval import (
    evaluate_kitti_style,
    BoxSet,
    DifficultyResult,
    PointMetrics,
    ClassPrCurve,
    AttributeDifficultyConfig,
    read_yolo_labels,
)
from fish_monitoring.eval.metrics import (
    derive_difficulty_thresholds,
    export_yolo_like_artifacts,
    CurvePoint,
    sweep_point_metrics,
    confusion_matrix_detection,
)
from fish_monitoring.eval.diagnose import (
    Gt,
    diagnose,
    sweep_conf,
    pick_best_conf,
    DiagnoseResult,
    SweepPoint,
    _match_predictions,
    _load_image_size,
)
from fish_monitoring.eval.report import ReportConfig, write_performance_report
from fish_monitoring.eval.spatial import SpatialReportConfig, run_spatial_error_report

__all__ = [
    "evaluate_kitti_style", "BoxSet", "DifficultyResult", "PointMetrics",
    "ClassPrCurve", "AttributeDifficultyConfig", "read_yolo_labels",
    "derive_difficulty_thresholds", "export_yolo_like_artifacts", "CurvePoint",
    "sweep_point_metrics", "confusion_matrix_detection",
    "Gt", "diagnose", "sweep_conf", "pick_best_conf",
    "DiagnoseResult", "SweepPoint", "_match_predictions", "_load_image_size",
    "ReportConfig", "write_performance_report",
    "SpatialReportConfig", "run_spatial_error_report",
]
