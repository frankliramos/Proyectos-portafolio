# Contributing to Proyectos-portafolio

Thank you for your interest in this data science portfolio. Contributions, feedback, and suggestions are welcome.

---

## 📋 How to Contribute

### Reporting Issues

If you find a bug, broken link, or inconsistency in any project:

1. Open a [GitHub Issue](https://github.com/frankliramos/Proyectos-portafolio/issues)
2. Use a clear title describing the problem
3. Include the project name (e.g., "Proyecto 1") and the file affected
4. Provide steps to reproduce if applicable

### Suggesting Improvements

Have an idea to improve a model, dashboard, or documentation?

1. Open an issue with the label `enhancement`
2. Describe the proposed change and its benefit
3. Reference any relevant papers, datasets, or tools

### Submitting a Pull Request

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-description`
3. Make your changes following the code style guidelines below
4. Commit with a clear message: `git commit -m "Fix: correct typo in Proyecto 2 README"`
5. Push to your fork: `git push origin feature/your-description`
6. Open a Pull Request describing what was changed and why

---

## 🎨 Code Style Guidelines

### Python

- Follow **PEP 8** conventions
- Use **type hints** where practical
- Write **docstrings** for all public functions and classes (Google style)
- Keep line length ≤ 100 characters
- Format with `black` before committing:
  ```bash
  black .
  ```
- Lint with `flake8`:
  ```bash
  flake8 . --max-line-length=100
  ```

### Notebooks

- Clear all outputs before committing: `Kernel → Restart & Clear Output`
- Use sequential cell execution (no skipped cells)
- Include markdown cells explaining each major step
- Follow naming convention: `01_eda_projectname.ipynb`, `02_modeling.ipynb`

### Documentation

- Use clear, professional English (and Spanish where bilingual docs exist)
- Keep README files up to date with any structural changes
- Add docstrings to any new source module

---

## 🗂️ Project Structure

Each project follows this standard structure:

```
Proyecto N/
├── README.md              # English documentation
├── README_ES.md           # Spanish documentation
├── requirements.txt       # Pinned Python dependencies
├── app.py                 # Entry point for Streamlit dashboard
├── dashboard/             # Dashboard application files
├── data/
│   ├── raw/               # Original, unmodified data
│   └── processed/         # Cleaned and engineered data
├── models/                # Trained model artifacts
├── notebooks/             # Jupyter notebooks (EDA → Modeling → Evaluation)
├── src/                   # Reusable Python modules
└── results/               # Metrics, plots, and output files
```

---

## ✅ Checklist Before Opening a PR

- [ ] Code follows PEP 8 / project style
- [ ] Docstrings added for new functions
- [ ] Notebooks have cleared outputs
- [ ] No sensitive data (credentials, private keys, personal info)
- [ ] README updated if project structure changed
- [ ] Tests pass (for Proyecto 1: `python -m pytest`)

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers this repository.

---

## 📧 Contact

**Franklin Ramos**
- GitHub: [@frankliramos](https://github.com/frankliramos)
- Repository: [Proyectos-portafolio](https://github.com/frankliramos/Proyectos-portafolio)
