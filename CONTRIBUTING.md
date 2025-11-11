# Contributing to Optical Neuromorphic Eikonal Solver

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## Ways to Contribute

### 🐛 Bug Reports

Found a bug? Please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, GPU, Python version)
- Error messages or screenshots

### ✨ Feature Requests

Have an idea? Open an issue describing:
- The problem it solves
- Proposed solution
- Potential implementation approach
- Why it would benefit the project

### 📖 Documentation

Improvements to documentation are always welcome:
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve API documentation
- Translate to other languages

### 💻 Code Contributions

#### Before You Start

1. Check existing issues and PRs
2. Discuss major changes in an issue first
3. Fork the repository
4. Create a feature branch

#### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/optical-neuromorphic-eikonal-solver.git
cd optical-neuromorphic-eikonal-solver

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/
```

#### Code Style

- **Python**: Follow PEP 8
- **Line Length**: 100 characters max
- **Docstrings**: Google style
- **Type Hints**: Use where appropriate
- **Comments**: Explain *why*, not *what*

Example:

```python
def estimate_iterations(grid_size: int, complexity: float) -> int:
    """
    Estimate iterations needed for convergence.
    
    Args:
        grid_size: Width/height of the square grid
        complexity: Obstacle density [0, 1]
        
    Returns:
        Estimated iteration count
        
    Note:
        Typical convergence requires 2n iterations where n = grid_size,
        but complex obstacles may require up to 4n.
    """
    base_iters = grid_size * 2
    complexity_factor = 1.0 + complexity
    return int(base_iters * complexity_factor)
```

#### Testing

- Add tests for new features
- Ensure existing tests pass
- Aim for >80% code coverage
- Test on multiple platforms if possible

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_solver.py

# With coverage
python -m pytest --cov=. tests/
```

#### Commit Messages

Use clear, descriptive commit messages:

```
Add support for 3D grids

- Extend texture system to 3D
- Update shaders for volumetric rendering
- Add 3D path extraction
- Include test cases

Closes #42
```

Format:
- First line: imperative mood summary (50 chars)
- Blank line
- Detailed explanation (wrap at 72 chars)
- Reference issues/PRs

#### Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** for new features
3. **Update CHANGELOG.md** with your changes
4. **Ensure CI passes** (if available)
5. **Request review** from maintainers

PR Description Template:

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- List of specific changes
- Another change
- etc.

## Testing
How was this tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Code follows style guidelines
- [ ] All tests pass
```

### 📊 Benchmarks

Contributions to benchmarks are valuable:
- New datasets
- Additional test cases
- Performance optimizations
- Validation against other methods

### 🎨 Visualization

Improvements to visualization:
- Better color schemes
- Additional rendering modes
- Interactive features
- Performance optimizations

## Project Areas

### High Priority

- [ ] 3D volumetric pathfinding
- [ ] Dynamic obstacle handling
- [ ] Multi-agent planning
- [ ] CUDA backend (alternative to OpenGL)
- [ ] Web demo (WebGL/WebGPU)

### Medium Priority

- [ ] Diagonal movement (8-connected grids)
- [ ] Multi-resolution hierarchy
- [ ] Batch query optimization
- [ ] Memory reduction techniques
- [ ] Python typing improvements

### Low Priority (But Welcome!)

- [ ] Additional color schemes
- [ ] Export functionality (paths, fields)
- [ ] Animation recording
- [ ] Plugin system
- [ ] Alternative solvers comparison

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of:
- Experience level
- Gender identity and expression
- Sexual orientation
- Disability
- Personal appearance
- Body size
- Race or ethnicity
- Age
- Religion
- Nationality

### Expected Behavior

- Be respectful and considerate
- Welcome newcomers
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy toward others

### Unacceptable Behavior

- Harassment of any kind
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information
- Other conduct inappropriate in a professional setting

### Enforcement

Violations may result in:
1. Warning
2. Temporary ban
3. Permanent ban

Report violations to: [contact email]

## Questions?

Don't hesitate to ask questions:
- Open an issue with the "question" label
- Join discussions in GitHub Discussions
- Contact maintainers directly

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in academic publications (if applicable)

Thank you for contributing! 🎉

