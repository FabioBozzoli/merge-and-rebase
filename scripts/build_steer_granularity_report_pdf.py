"""Render the steer granularity Markdown report as a readable, styled PDF."""

from __future__ import annotations

import argparse
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table


@dataclass(frozen=True)
class Line:
    text: str
    kind: str = "body"


@dataclass(frozen=True)
class TableBlock:
    section: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


type DocumentItem = Line | TableBlock


@dataclass(frozen=True)
class TextPage:
    lines: tuple[Line, ...]


@dataclass(frozen=True)
class TablePage:
    section: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    part: int
    parts: int


type Page = TextPage | TablePage


STYLE = {
    "title": {"size": 20.0, "weight": "bold", "family": "DejaVu Sans", "height": 0.047, "width": 58},
    "h2": {"size": 13.5, "weight": "bold", "family": "DejaVu Sans", "height": 0.033, "width": 78},
    "h3": {"size": 10.5, "weight": "bold", "family": "DejaVu Sans", "height": 0.026, "width": 96},
    "body": {"size": 8.4, "weight": "normal", "family": "DejaVu Sans", "height": 0.019, "width": 108},
    "bullet": {"size": 8.4, "weight": "normal", "family": "DejaVu Sans", "height": 0.019, "width": 102},
    "code": {"size": 7.2, "weight": "normal", "family": "DejaVu Sans Mono", "height": 0.017, "width": 104},
    "space": {"size": 1.0, "weight": "normal", "family": "DejaVu Sans", "height": 0.010, "width": 1},
}


def _strip_inline_markdown(value: str) -> str:
    value = re.sub(r"\[([^]]+)]\(([^)]+)\)", r"\1 (\2)", value)
    return value.replace("**", "").replace("__", "").replace("`", "").strip()


def _wrapped(text: str, kind: str, *, prefix: str = "", continuation: str = "") -> list[Line]:
    width = max(20, int(STYLE[kind]["width"]) - len(prefix))
    chunks = textwrap.wrap(
        _strip_inline_markdown(text),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=False,
    ) or [""]
    return [Line((prefix if i == 0 else continuation) + chunk, kind) for i, chunk in enumerate(chunks)]


def _table_cells(raw: str) -> tuple[str, ...]:
    return tuple(_strip_inline_markdown(cell) for cell in raw.strip().strip("|").split("|"))


def _is_table_separator(raw: str) -> bool:
    cells = raw.strip().strip("|").split("|")
    return bool(cells) and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells)


def _parse_text_line(raw: str, *, in_code: bool) -> list[Line]:
    stripped = raw.rstrip()
    if in_code:
        return _wrapped(stripped, "code")
    if not stripped:
        return [Line("", "space")]
    if stripped.startswith("# "):
        return [*_wrapped(stripped[2:], "title"), Line("", "space")]
    if stripped.startswith("## "):
        return [Line("", "space"), *_wrapped(stripped[3:], "h2")]
    if stripped.startswith("### "):
        return [Line("", "space"), *_wrapped(stripped[4:], "h3")]
    if re.match(r"^\s*[-*] ", stripped):
        content = re.sub(r"^\s*[-*] ", "", stripped)
        return _wrapped(content, "bullet", prefix="• ", continuation="  ")
    numbered = re.match(r"^\s*(\d+\.)\s+(.*)$", stripped)
    if numbered:
        prefix = f"{numbered.group(1)} "
        return _wrapped(numbered.group(2), "bullet", prefix=prefix, continuation=" " * len(prefix))
    return _wrapped(stripped, "body")


def parse_markdown(path: Path) -> list[DocumentItem]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    items: list[DocumentItem] = []
    in_code = False
    current_section = "Report"
    index = 0
    while index < len(raw_lines):
        raw = raw_lines[index]
        if raw.startswith("```"):
            in_code = not in_code
            items.append(Line("", "space"))
            index += 1
            continue
        if not in_code and raw.startswith(("## ", "### ")):
            current_section = _strip_inline_markdown(raw.lstrip("# "))
        if not in_code and raw.strip().startswith("|"):
            table_lines: list[str] = []
            while index < len(raw_lines) and raw_lines[index].strip().startswith("|"):
                table_lines.append(raw_lines[index])
                index += 1
            parsed = [_table_cells(line) for line in table_lines if not _is_table_separator(line)]
            if len(parsed) >= 2:
                width = len(parsed[0])
                normalized = [tuple((*row, *("" for _ in range(width - len(row)))))[:width] for row in parsed]
                items.append(TableBlock(current_section, normalized[0], tuple(normalized[1:])))
            continue
        items.extend(_parse_text_line(raw, in_code=in_code))
        index += 1
    return items


def _column_widths(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> tuple[float, ...]:
    raw_weights = []
    for col, header in enumerate(headers):
        longest = max([len(header), *(len(row[col]) for row in rows)], default=len(header))
        raw_weights.append(min(34.0, max(8.0, longest**0.72)))
    total = sum(raw_weights)
    return tuple(weight / total for weight in raw_weights)


def _wrap_table(
    headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], tuple[int, ...], tuple[float, ...]]:
    widths = _column_widths(headers, rows)

    def wrap_row(row: tuple[str, ...]) -> tuple[tuple[str, ...], int]:
        wrapped = []
        max_lines = 1
        for value, fraction in zip(row, widths, strict=True):
            char_width = max(7, int(105 * fraction))
            parts = textwrap.wrap(value, width=char_width, break_long_words=True, break_on_hyphens=True) or [""]
            wrapped.append("\n".join(parts))
            max_lines = max(max_lines, len(parts))
        return tuple(wrapped), max_lines

    wrapped_header, _ = wrap_row(headers)
    wrapped_rows: list[tuple[str, ...]] = []
    row_units: list[int] = []
    for row in rows:
        wrapped, units = wrap_row(row)
        wrapped_rows.append(wrapped)
        row_units.append(units)
    return wrapped_header, tuple(wrapped_rows), tuple(row_units), widths


def _split_table(block: TableBlock, max_units: int = 27) -> list[TablePage]:
    _, _, units, _ = _wrap_table(block.headers, block.rows)
    chunks: list[tuple[tuple[str, ...], ...]] = []
    current: list[tuple[str, ...]] = []
    used = 0
    for row, row_units in zip(block.rows, units, strict=True):
        cost = max(2, row_units)
        if current and used + cost > max_units:
            chunks.append(tuple(current))
            current = []
            used = 0
        current.append(row)
        used += cost
    if current:
        chunks.append(tuple(current))
    return [TablePage(block.section, block.headers, rows, part, len(chunks)) for part, rows in enumerate(chunks, 1)]


def paginate(items: list[DocumentItem]) -> list[Page]:
    pages: list[Page] = []
    current: list[Line] = []
    used = 0.0
    capacity = 0.84

    def flush_text() -> None:
        nonlocal current, used
        if current:
            pages.append(TextPage(tuple(current)))
            current = []
            used = 0.0

    for item in items:
        if isinstance(item, TableBlock):
            flush_text()
            pages.extend(_split_table(item))
            continue
        height = float(STYLE[item.kind]["height"])
        if current and used + height > capacity:
            flush_text()
        current.append(item)
        used += height
    flush_text()
    return pages


def _base_figure(page_number: int, total_pages: int):
    figure = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    figure.text(
        0.06,
        0.968,
        "STEER — REPORT SULLA GRANULARITÀ DEI BLOCCHI",
        fontsize=7.5,
        color="#52606d",
        family="DejaVu Sans",
    )
    figure.text(
        0.94,
        0.030,
        f"Pagina {page_number} / {total_pages}",
        fontsize=7.0,
        color="#52606d",
        family="DejaVu Sans",
        ha="right",
    )
    return figure


def _render_text_page(figure, page: TextPage) -> None:
    y = 0.925
    for line in page.lines:
        style = STYLE[line.kind]
        if line.kind != "space":
            color = "#172b4d" if line.kind in {"title", "h2", "h3"} else "#202124"
            figure.text(
                0.07,
                y,
                line.text,
                fontsize=float(style["size"]),
                fontweight=str(style["weight"]),
                family=str(style["family"]),
                color=color,
                va="top",
            )
        y -= float(style["height"])


def _render_table_page(figure, page: TablePage, table_number: int) -> None:
    suffix = f" — parte {page.part}/{page.parts}" if page.parts > 1 else ""
    figure.text(
        0.06,
        0.925,
        f"Tabella {table_number}: {page.section}{suffix}",
        fontsize=12.0,
        fontweight="bold",
        family="DejaVu Sans",
        color="#172b4d",
        va="top",
    )

    headers, rows, row_units, widths = _wrap_table(page.headers, page.rows)
    header_units = max(cell.count("\n") + 1 for cell in headers) + 0.5
    total_units = header_units + sum(max(1.6, units) for units in row_units)
    table_height = min(0.79, max(0.16, 0.043 * total_units))
    axes = figure.add_axes((0.055, 0.88 - table_height, 0.89, table_height))
    axes.set_axis_off()
    table = Table(axes, bbox=(0, 0, 1, 1))

    header_height = header_units / total_units
    for col, (value, width) in enumerate(zip(headers, widths, strict=True)):
        cell = table.add_cell(0, col, width, header_height, text=value, loc="left")
        cell.set_facecolor("#1f4e78")
        cell.set_edgecolor("#ffffff")
        cell.set_linewidth(0.8)
        cell.PAD = 0.06
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")

    for row_index, (row, units) in enumerate(zip(rows, row_units, strict=True), start=1):
        height = max(1.6, units) / total_units
        background = "#eef4f8" if row_index % 2 else "#ffffff"
        for col, (value, width) in enumerate(zip(row, widths, strict=True)):
            cell = table.add_cell(row_index, col, width, height, text=value, loc="left")
            cell.set_facecolor(background)
            cell.set_edgecolor("#b8c7d1")
            cell.set_linewidth(0.55)
            cell.PAD = 0.055
            cell.get_text().set_color("#202124")

    font_size = 6.2 if len(headers) >= 6 else 6.7 if len(headers) >= 5 else 7.3
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    axes.add_table(table)


def render_pdf(source: Path, output: Path) -> int:
    pages = paginate(parse_markdown(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    table_number = 0
    with PdfPages(output) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "Granularità dei blocchi nel metodo steer"
        metadata["Author"] = "merge-and-rebase"
        metadata["Subject"] = "Implementazione delle granularità linear e model"
        for page_number, page in enumerate(pages, start=1):
            figure = _base_figure(page_number, len(pages))
            if isinstance(page, TextPage):
                _render_text_page(figure, page)
            else:
                if page.part == 1:
                    table_number += 1
                _render_table_page(figure, page, table_number)
            pdf.savefig(figure, bbox_inches=None)
            plt.close(figure)
    return len(pages)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("docs/steer-block-granularity-report.md"))
    parser.add_argument("--output", type=Path, default=Path("docs/steer-block-granularity-report.pdf"))
    args = parser.parse_args()
    page_count = render_pdf(args.source, args.output)
    print(f"Generated {args.output} ({page_count} pages)")


if __name__ == "__main__":
    main()
