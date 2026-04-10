import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import nbformat
from nbclient import NotebookClient
import plotly.graph_objects as go
import plotly.io as pio


@dataclass(frozen=True)
class ExportJob:
    notebook: Path
    targets: list[str]


EXPORT_JOBS = {
    "eda.ipynb": ExportJob(
        notebook=Path("eda.ipynb"),
        targets=[
            "01-dataset-size-by-table.svg",
            "02-monthly-active-users-trend.svg",
            "03-monthly-event-intensity.svg",
            "04-total-points-distribution.svg",
            "05-lifecycle-stage-mix.svg",
            "06-regional-user-footprint-1.svg",
            "07-regional-user-footprint-2.svg",
            "08-monthly-label-dynamics.svg",
        ],
    ),
    "final.ipynb": ExportJob(
        notebook=Path("final.ipynb"),
        targets=[
            "09-validation-metric-output.svg",
            "10-model-family-check.svg",
            "11-churn-re-engagement-calibration.svg",
            "12-crm-segment-distribution.svg",
        ],
    ),
}
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Plotly figures from the project notebooks and export them "
            "to rounded-corner SVG files."
        )
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=None,
        help=(
            "Optional notebook path. When omitted, export both the EDA and final "
            "notebook figure sets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("images"),
        help="Directory where exported images will be written.",
    )
    return parser.parse_args()


def resolve_jobs(selected_notebook: Path | None) -> list[ExportJob]:
    if selected_notebook is None:
        return [EXPORT_JOBS["eda.ipynb"], EXPORT_JOBS["final.ipynb"]]

    notebook_name = selected_notebook.name
    if notebook_name not in EXPORT_JOBS:
        supported = ", ".join(sorted(EXPORT_JOBS))
        raise ValueError(
            f"Unsupported notebook {
                selected_notebook}. Supported notebooks: {supported}."
        )
    return [ExportJob(notebook=selected_notebook, targets=EXPORT_JOBS[notebook_name].targets)]


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
        f"No Plotly figures found in {
            notebook_path} outputs even after execution."
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
            style.replace(
                "fill: rgb(0, 0, 0); fill-opacity: 0;",
                "fill: rgb(255, 255, 255); fill-opacity: 1;",
            ),
        )

    content_group = ET.Element(
        f"{{{SVG_NS}}}g", {"clip-path": f"url(#{clip_id})"})
    move_children = [child for child in list(root) if child is not defs]
    for child in move_children:
        root.remove(child)
        content_group.append(child)
    root.append(content_group)

    tree.write(svg_path, encoding="unicode")


def infer_figure_dimensions(figure: go.Figure, *, filename: str) -> tuple[int, int]:
    width = figure.layout.width
    height = figure.layout.height
    if width is None or height is None:
        raise ValueError(
            f"Figure for {filename} is missing layout width/height. "
            "Set the dimensions in the notebook before exporting."
        )
    return int(width), int(height)


def export_single_figure(
    figure: go.Figure,
    *,
    filename: str,
    output_dir: Path,
) -> str:
    output_path = output_dir / filename
    export_figure = go.Figure(figure)
    width, height = infer_figure_dimensions(export_figure, filename=filename)
    export_figure.write_image(output_path, width=width, height=height)
    round_svg_corners(output_path, width=width, height=height, radius=24)
    return f"exported {output_path} ({width}x{height})"


def export_figures(
    figures: list[go.Figure],
    output_dir: Path,
    export_targets: list[str],
) -> None:
    if len(figures) != len(export_targets):
        raise ValueError(
            f"Expected {len(export_targets)} Plotly figures, found {
                len(figures)}. "
            "Re-run the notebook before exporting."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pio.defaults.default_format = "svg"

    max_workers = min(len(figures), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for message in executor.map(
            lambda item: export_single_figure(
                item[0],
                filename=item[1],
                output_dir=output_dir,
            ),
            zip(figures, export_targets, strict=True),
        ):
            print(message)


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(args.notebook)

    for job in jobs:
        print(f"exporting figures from {job.notebook}")
        figures = extract_plotly_figures(job.notebook)
        export_figures(figures, args.output_dir, job.targets)


if __name__ == "__main__":
    main()
