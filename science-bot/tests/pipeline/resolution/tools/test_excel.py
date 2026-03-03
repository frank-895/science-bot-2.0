from pathlib import Path

import openpyxl
from science_bot.pipeline.resolution.tools.excel import list_excel_sheets


def test_list_excel_sheets_returns_workbook_sheet_names(tmp_path: Path):
    workbook = openpyxl.Workbook()
    workbook.active.title = "Summary"
    workbook.create_sheet("Sheet2")
    file_path = tmp_path / "results.xlsx"
    workbook.save(file_path)
    workbook.close()

    sheet_names = list_excel_sheets(tmp_path, "results.xlsx")

    assert sheet_names == ["Summary", "Sheet2"]
