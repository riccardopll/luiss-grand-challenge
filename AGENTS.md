# Project Notes

- `data/`: local project datasets used for EDA and modeling inputs.
- Graphs must be presentation-friendly (clear labels, readable styling, and clean layout).
- After every code change involving graphs, re-run the relevant Jupyter cell(s), inspect notebook output, and verify the graph output.
- Do not run `task export` unless the user explicitly asks for it.

# Python Change Rule

After every Python-related change (including `.py` and `.ipynb` files), run:

- `task lint`
- `task typecheck`

# Python Environment Rule

Use the project's bundled virtual environment (`.venv`) for every Python operation (commands, scripts, tools, linting, and type checking).
