# Project Notes

- `data/`: local project datasets used for EDA and modeling inputs.
- `docs/`: project documentation, specs, and presentation/planning notes.
- Graphs must be presentation-friendly (clear labels, readable styling, and clean layout).
- After every code change involving graphs, re-run the relevant Jupyter cell(s), inspect notebook output, and verify every generated graph.

# Python Change Rule

After every Python-related change (including `.py` and `.ipynb` files), run:

- `task lint`
- `task typecheck`

# Python Environment Rule

Use the project's bundled virtual environment (`.venv`) for every Python operation (commands, scripts, tools, linting, and type checking).
