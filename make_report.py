"""Создаёт PDF-отчёт по результатам эксперимента с ансамблем."""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
REPO_URL = "https://github.com/sxd993/ensemble-classification"


def main() -> None:
    data = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    rows, baseline = data["ensemble"], data["decision_tree_baseline"]
    pdfmetrics.registerFont(TTFont("TNR", "/System/Library/Fonts/Supplemental/Times New Roman.ttf"))
    pdfmetrics.registerFont(TTFont("TNR-Bold", "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"))
    styles = getSampleStyleSheet(); styles["Title"].fontName = "TNR-Bold"
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="TNR", fontSize=11, leading=16, spaceAfter=8)
    heading = ParagraphStyle("heading", parent=styles["Heading1"], fontName="TNR-Bold", spaceBefore=12, spaceAfter=8)
    cell = ParagraphStyle("cell", parent=body, alignment=1, leading=13, spaceAfter=0)
    cell_bold = ParagraphStyle("cell_bold", parent=cell, fontName="TNR-Bold")
    story = [Paragraph("ОТЧЁТ: ансамблевая классификация Census Income", styles["Title"]), Spacer(1, 8*mm)]
    story += [Paragraph("1. Формулировка задания", heading), Paragraph("Разработать программу классификации набора данных ансамблевой техникой; варьировать количество участников ансамбля от 50 до 100 с шагом 10, визуализировать показатели качества и сравнить их с деревом решений.", body)]
    story += [Paragraph("2. Исходные тексты и данные", heading), Paragraph(f'Исходные тексты, Census Income и результаты размещены в репозитории: <link href="{REPO_URL}"><u>{REPO_URL}</u></link>.', body)]
    story += [Paragraph("3. Методика", heading), Paragraph(f"Использован метод {rows[0]['technique']} при разбиении 80:20, random state 42 и максимальной глубине базовых деревьев {rows[0]['max_depth']}. Положительный класс — доход более $50 000. Для сопоставимости использованы показатели дерева решений из предыдущего задания на том же разбиении.", body)]
    story += [Paragraph("4. Результаты", heading)]
    table_rows = [[Paragraph(value, cell_bold) for value in ["Участников", "Accuracy", "Precision", "Recall", "F1"]]]
    table_rows += [
        [Paragraph(value, cell) for value in [str(r["n_estimators"])] + [f"{r[k]:.4f}" for k in ("accuracy", "precision", "recall", "f1")]]
        for r in rows
    ]
    table_rows.append([Paragraph(value, cell) for value in ["Дерево решений"] + [f"{baseline[k]:.4f}" for k in ("accuracy", "precision", "recall", "f1")]])
    table = Table(table_rows, colWidths=[35*mm, 33.75*mm, 33.75*mm, 33.75*mm, 33.75*mm])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e3f2fd")), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("PADDING", (0,0), (-1,-1), 5)]))
    story += [table, Spacer(1, 5*mm), Image(str(RESULTS / "quality_vs_ensemble_size.png"), width=165*mm, height=91*mm), Paragraph("Рисунок 1. Зависимость показателей качества от количества участников ансамбля; пунктирные линии — результаты дерева решений.", body)]
    best = max(rows, key=lambda r: r["f1"])
    story += [Paragraph("5. Пояснение результатов", heading), Paragraph(f"Наибольшая F-мера ({best['f1']:.4f}) достигнута при {best['n_estimators']} участниках. Ансамбль повышает устойчивость модели: усреднение результатов множества деревьев снижает влияние особенностей отдельных обучающих подвыборок. В сравнении с одиночным деревом особенно важны precision, recall и F-мера, поскольку классы дохода несбалансированы.", body)]
    SimpleDocTemplate(str(RESULTS / "report.pdf"), pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=18*mm, bottomMargin=18*mm, title="Ансамблевая классификация Census Income").build(story)
    print(RESULTS / "report.pdf")


if __name__ == "__main__":
    main()
