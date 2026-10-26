# How project folders are organized

Each paper or project has its own top-level folder, named with a short slug
(for example `padic-BC/`). It is published at `https://wazuniga.github.io/<slug>/`.

Every project folder has the same layout:

```
<slug>/
  index.html   the project page (title, papers, links to every file)
  README.md    how to build, run and reproduce the results
  code/        source code
  data/        data used in the papers
  latex/       LaTeX source (arXiv package)
```

To start a new project, copy `_template/` to a new folder, fill in the placeholders, and
add an entry for it under "Code and data" in the home page `index.html`.

The file `.nojekyll` tells GitHub Pages to serve every file as it is (for example,
`README.md` as plain text and data files unchanged).
