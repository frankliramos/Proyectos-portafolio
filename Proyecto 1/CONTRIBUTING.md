# Contributing to Turbofan RUL Prediction Project

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Turbofan Engine RUL Prediction project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## 🤝 Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow:

- **Be respectful**: Treat everyone with respect and kindness
- **Be collaborative**: Work together towards common goals
- **Be professional**: Maintain professional communication
- **Be inclusive**: Welcome contributors from all backgrounds

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Proyectos-portafolio.git
   cd "Proyectos-portafolio/Proyecto 1"
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ How to Contribute

### Types of Contributions

We welcome the following types of contributions:

1. **Bug Fixes**: Fix issues in existing code
2. **New Features**: Add new functionality
3. **Documentation**: Improve or add documentation
4. **Performance**: Optimize existing code
5. **Testing**: Add or improve tests
6. **Refactoring**: Improve code quality

### Contribution Areas

- **Model Improvements**: Better architectures, hyperparameter tuning
- **Dashboard Enhancements**: UI/UX improvements, new visualizations
- **Data Processing**: Better feature engineering, data validation
- **Documentation**: Tutorials, examples, guides
- **Testing**: Unit tests, integration tests
- **Infrastructure**: Docker, CI/CD, deployment

## 💻 Development Setup

### Prerequisites

- Python 3.12+
- Git
- Virtual environment tool (venv, conda)

### Setup Steps

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify setup**:
   ```bash
   python -c "import torch; import streamlit; print('Setup OK!')"
   ```

### Project Structure

```
Proyecto 1/
├── app.py              # Streamlit dashboard
├── src/                # Source code modules
│   ├── config.py       # Configuration
│   ├── data_loading.py # Data utilities
│   ├── features.py     # Feature engineering
│   ├── models.py       # Model architectures
│   └── inference.py    # Inference engine
├── notebooks/          # Jupyter notebooks
├── data/               # Data files
├── models/             # Trained models
└── tests/              # Unit tests (future)
```

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters maximum (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Grouped (standard library, third-party, local)
- **Docstrings**: Google style

### Example Code Style

```python
"""
Module docstring explaining purpose.

Author: Your Name
Date: YYYY-MM-DD
"""

import numpy as np
from typing import Optional

def calculate_rul(
    cycles: np.ndarray, 
    max_cycles: int
) -> np.ndarray:
    """
    Calculate Remaining Useful Life for engine cycles.
    
    Args:
        cycles (np.ndarray): Array of cycle numbers.
        max_cycles (int): Maximum cycle count.
    
    Returns:
        np.ndarray: RUL values for each cycle.
    
    Example:
        >>> cycles = np.array([1, 2, 3])
        >>> calculate_rul(cycles, 10)
        array([9, 8, 7])
    """
    return max_cycles - cycles
```

### Code Quality Tools

Use these tools before submitting:

```bash
# Format code
black app.py src/

# Check style
flake8 app.py src/

# Type checking (optional)
mypy src/
```

## 🧪 Testing

### Running Tests

Currently, tests are minimal. We welcome contributions to improve test coverage!

```bash
# Run existing tests
python test_load_fd001.py
python test_eda_simple.py
```

### Writing Tests

When adding new features, please include tests:

```python
# tests/test_inference.py
import pytest
from src.inference import RULInference

def test_inference_initialization():
    """Test that inference engine initializes correctly."""
    engine = RULInference(project_root=".")
    assert engine is not None
    assert engine.device is not None
```

## 🔄 Pull Request Process

1. **Update documentation**: Ensure README, docstrings are current
2. **Add tests**: Include tests for new functionality
3. **Check code quality**: Run linters and formatters
4. **Update CHANGELOG**: Document your changes
5. **Create PR**: Submit with clear description

### PR Title Format

Use conventional commits format:

- `feat: Add confidence intervals to predictions`
- `fix: Resolve sensor data loading issue`
- `docs: Update installation instructions`
- `refactor: Simplify inference logic`
- `test: Add unit tests for data loading`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring

## Testing
- [ ] Tested locally
- [ ] Added new tests
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Reproduction**: Steps to reproduce
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, OS, dependencies
6. **Screenshots**: If applicable

### Feature Requests

When requesting features, include:

1. **Problem**: What problem does this solve?
2. **Solution**: Proposed solution
3. **Alternatives**: Alternative solutions considered
4. **Use case**: Specific use case/example

## 📚 Documentation Guidelines

### README Updates

- Keep installation instructions current
- Add examples for new features
- Update performance metrics if changed

### Code Documentation

- All public functions need docstrings
- Include type hints
- Provide usage examples
- Document exceptions raised

### Notebook Documentation

- Clear markdown explanations
- Well-commented code cells
- Visualizations with titles/labels
- Conclusion sections

## 🎯 Priority Areas

We're particularly interested in contributions in these areas:

1. **Uncertainty Quantification**: Add confidence intervals to predictions
2. **Model Monitoring**: Implement drift detection
3. **Real-time Integration**: Streaming data support
4. **Additional Datasets**: Support for FD002, FD003, FD004
5. **Testing**: Comprehensive test suite
6. **Docker**: Containerization for easy deployment
7. **CI/CD**: Automated testing and deployment

## ❓ Questions?

If you have questions about contributing:

1. Check existing issues and discussions
2. Create a new issue with the "question" label
3. Reach out to the maintainer

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to make predictive maintenance more accessible!** 🚀
