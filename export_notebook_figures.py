from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import nbformat
from nbclient import NotebookClient
import plotly.graph_objects as go
import plotly.io as pio

EXPORT_TARGETS = [
    ("01-dataset-size-by-table.svg", 950, 400),
    ("02-monthly-active-users-trend.svg", 950, 350),
    ("03-monthly-event-intensity.svg", 850, 350),
    ("04-total-points-distribution.svg", 1150, 350),
    ("05-lifecycle-stage-mix.svg", 1150, 350),
    ("06-regional-user-footprint-1.svg", 1250, 650),
    ("07-regional-user-footprint-2.svg", 1250, 650),
    ("08-monthly-label-dynamics.svg", 1150, 550),
]
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Plotly figures from an executed notebook and export them "
            "to SVG files with Kaleido."
        )
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("eda.ipynb"),
        help="Path to the executed notebook.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("images"),
        help="Directory where exported images will be written.",
    )
    return parser.parse_args()


def collect_plotly_figures(nb: nbformat.NotebookNode) -> list[go.Figure]:
    figures: list[go.Figure] = []

    for cell in nb.cells:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            plotly_json = data.get("application/vnd.plotly.v1+json")
            if plotly_json is None:
                continue
            figures.append(
                go.Figure(
                    data=plotly_json.get("data", []),
                    layout=plotly_json.get("layout", {}),
                )
            )

    return figures


def extract_plotly_figures(notebook_path: Path) -> list[go.Figure]:
    nb = nbformat.read(notebook_path, as_version=4)
    figures = collect_plotly_figures(nb)
    if figures:
        return figures

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    nbformat.write(nb, notebook_path)

    figures = collect_plotly_figures(nb)
    if figures:
        return figures

    raise ValueError(
        "No Plotly figures found in notebook outputs even after execution."
    )


def round_svg_corners(svg_path: Path, *, width: int, height: int, radius: int) -> None:
    tree = ET.parse(svg_path)
    root = tree.getroot()

    defs = root.find(f"{{{SVG_NS}}}defs")
    if defs is None:
        defs = ET.Element(f"{{{SVG_NS}}}defs")
        root.insert(0, defs)

    clip_id = "rounded-canvas-clip"
    for existing in defs.findall(f"{{{SVG_NS}}}clipPath"):
        if existing.get("id") == clip_id:
            defs.remove(existing)

    clip_path = ET.SubElement(defs, f"{{{SVG_NS}}}clipPath", {"id": clip_id})
    ET.SubElement(
        clip_path,
        f"{{{SVG_NS}}}rect",
        {
            "x": "0",
            "y": "0",
            "width": str(width),
            "height": str(height),
            "rx": str(radius),
            "ry": str(radius),
        },
    )

    background = root.find(f"{{{SVG_NS}}}rect")
    if background is not None:
        background.set("fill", "rgb(255, 255, 255)")
        background.set("fill-opacity", "1")
        background.set("rx", str(radius))
        background.set("ry", str(radius))
        style = background.get("style", "")
        background.set(
            "style",
            style.replace("fill: rgb(0, 0, 0); fill-opacity: 0;", "fill: rgb(255, 255, 255); fill-opacity: 1;"),
        )

    content_group = ET.Element(
        f"{{{SVG_NS}}}g",
        {"clip-path": f"url(#{clip_id})"},
    )
    move_children = [child for child in list(root) if child is not defs]
    for child in move_children:
        root.remove(child)
        content_group.append(child)
    root.append(content_group)

    tree.write(svg_path, encoding="unicode")


def export_figures(
    figures: list[go.Figure],
    output_dir: Path,
    export_targets: list[tuple[str, int, int]],
) -> None:
    if len(figures) != len(export_targets):
        raise ValueError(
            f"Expected {len(export_targets)} Plotly figures, found {len(figures)}. "
            "Re-run the notebook before exporting."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    pio.defaults.default_format = "svg"

    for figure, (filename, width, height) in zip(
        figures, export_targets, strict=True
    ):
        output_path = output_dir / filename
        export_figure = go.Figure(figure)
        export_figure.update_layout(title=None)
        if export_figure.layout.margin is not None:
            export_figure.update_layout(
                margin={
                    "l": export_figure.layout.margin.l or 0,
                    "r": export_figure.layout.margin.r or 0,
                    "b": export_figure.layout.margin.b or 0,
                    "t": 24,
                }
            )
        export_figure.write_image(output_path, width=width, height=height)
        round_svg_corners(output_path, width=width, height=height, radius=24)
        print(f"exported {output_path} ({width}x{height})")


def main() -> None:
    args = parse_args()
    figures = extract_plotly_figures(args.notebook)
    export_figures(
        figures,
        args.output_dir,
        EXPORT_TARGETS,
    )


if __name__ == "__main__":
    main()
