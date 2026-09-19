import copy

from docling_ibm_models.tableformer.data_management.matching_post_processor import (
    MatchingPostProcessor,
)
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor


def _cell(cell_id, row, column, bbox, cell_class=2, label="body"):
    return {
        "cell_id": cell_id,
        "row_id": row,
        "column_id": column,
        "bbox": list(bbox),
        "cell_class": cell_class,
        "label": label,
    }


def _processor(monkeypatch, enabled):
    if enabled:
        monkeypatch.setenv("CC_TABLEFORMER_EMPTY_CELL_PRESERVATION", "1")
    else:
        monkeypatch.delenv("CC_TABLEFORMER_EMPTY_CELL_PRESERVATION", raising=False)
    return MatchingPostProcessor({"predict": {"pdf_cell_iou_thres": 0.05}})


def test_feature_is_off_by_default(monkeypatch):
    processor = _processor(monkeypatch, enabled=False)
    aligned = [_cell(1, 0, 0, [10, 10, 20, 20])]
    candidate = _cell(2, 0, 1, [100, 100, 101, 101], cell_class=0, label="ecel")

    retained, retained_ids = processor._retain_supported_empty_cells(
        {2}, aligned + [candidate], aligned
    )

    assert retained == []
    assert retained_ids == []


def test_supported_empty_cell_uses_surviving_pdf_aligned_grid(monkeypatch):
    processor = _processor(monkeypatch, enabled=True)
    aligned = [
        _cell(1, 0, 0, [10, 10, 20, 20]),
        _cell(2, 0, 1, [20, 10, 30, 20]),
        _cell(3, 1, 0, [10, 20, 20, 30]),
    ]
    candidate = _cell(4, 1, 1, [100, 100, 101, 101], cell_class=0, label="ecel")

    retained, retained_ids = processor._retain_supported_empty_cells(
        {4}, aligned + [candidate], aligned
    )

    assert retained_ids == [4]
    assert retained[0]["cell_id"] == 4
    assert retained[0]["bbox"] == [20, 20, 30, 30]
    assert retained[0]["_cc_preserved_empty"] is True


def test_unsupported_rows_and_columns_are_not_restored(monkeypatch):
    processor = _processor(monkeypatch, enabled=True)
    aligned = [_cell(1, 0, 0, [10, 10, 20, 20])]
    unsupported_row = _cell(2, 1, 0, [10, 20, 20, 30], cell_class=0, label="ecel")
    unsupported_column = _cell(3, 0, 1, [20, 10, 30, 20], cell_class=0, label="ecel")

    retained, retained_ids = processor._retain_supported_empty_cells(
        {2, 3}, aligned + [unsupported_row, unsupported_column], aligned
    )

    assert retained == []
    assert retained_ids == []


def test_preserved_response_uses_real_id_and_no_text_match():
    predictor = object.__new__(TFPredictor)
    table_cells = [
        _cell(1, 0, 0, [10, 10, 20, 20]),
        _cell(2, 0, 1, [20, 10, 30, 20], cell_class=0, label="ecel"),
    ]

    responses = predictor._generate_tf_response(
        table_cells,
        {"7": [{"table_cell_id": 1}]},
        [2],
    )

    preserved = next(response for response in responses if response["cell_id"] == 2)
    assert preserved["cell_id"] == 2
    assert preserved["cell_id"] >= 0
    assert preserved["text_cell_bboxes"] == []
    assert preserved["start_row_offset_idx"] == 0
    assert preserved["start_col_offset_idx"] == 1


def test_process_keeps_preserved_cell_inside_normal_post_processing(monkeypatch):
    details = {
        "table_cells": [
            _cell(0, 0, 0, [0, 0, 10, 10]),
            _cell(1, 0, 1, [10, 0, 20, 10], cell_class=0, label="ecel"),
            _cell(2, 1, 0, [0, 10, 10, 20]),
            _cell(3, 1, 1, [10, 10, 20, 20]),
        ],
        "pdf_cells": [
            {"id": 0, "bbox": [0, 0, 10, 10], "text": "A"},
            {"id": 1, "bbox": [0, 10, 10, 20], "text": "B"},
            {"id": 2, "bbox": [10, 10, 20, 20], "text": "C"},
        ],
        "matches": {
            "0": [{"table_cell_id": 0, "iopdf": 1.0}],
            "1": [{"table_cell_id": 2, "iopdf": 1.0}],
            "2": [{"table_cell_id": 3, "iopdf": 1.0}],
        },
    }
    processor = _processor(monkeypatch, enabled=True)

    result = processor.process(copy.deepcopy(details))

    preserved = [cell for cell in result["table_cells"] if cell["cell_id"] == 1]
    assert len(preserved) == 1
    assert preserved[0]["_cc_preserved_empty"] is True
    assert result["preserved_unmatched_cell_ids"] == [1]
    assert preserved[0]["bbox"] == [10, 0, 20, 10]
