# 📦 Publishing AetherAI to PyPI & uv

This guide covers how to publish the AetherAI package to PyPI (pip) and ensure compatibility with uv.

## 📋 Pre-requisites

1. **PyPI Account**: Create accounts on:
   - [PyPI](https://pypi.org/account/register/) (production)
   - [TestPyPI](https://test.pypi.org/account/register/) (testing)

2. **API Tokens**: Generate API tokens from your PyPI account settings:
   - Go to Account Settings → API tokens
   - Create a token scoped to the `aetherai` project (or all projects for first upload)

3. **Tools**: Install the build tools:
   ```bash
   pip install --upgrade pip build twine
   ```

---

## 🔨 Manual Publishing

### Step 1: Clean Previous Builds

```bash
# Windows
rmdir /s /q dist build *.egg-info
rmdir /s /q aetherai.egg-info terminal.egg-info

# Linux/macOS
rm -rf dist/ build/ *.egg-info
```

### Step 2: Build the Package

```bash
python -m build
```

This creates:
- `dist/aetherai-1.0.0.tar.gz` (source distribution)
- `dist/aetherai-1.0.0-py3-none-any.whl` (wheel)

### Step 3: Verify the Build

```bash
# Check package metadata
twine check dist/*

# Test installation locally
pip install dist/aetherai-1.0.0-py3-none-any.whl --force-reinstall

# Verify CLI works
aetherai --version
```

### Step 4: Upload to TestPyPI (Recommended First)

```bash
twine upload --repository testpypi dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your TestPyPI API token

### Step 5: Test Installation from TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ aetherai
```

### Step 6: Upload to PyPI (Production)

```bash
twine upload dist/*
```

---

## 🤖 Automated Publishing (GitHub Actions)

### Setup Trusted Publishing (Recommended)

1. Go to PyPI → Your Project → Settings → Publishing
2. Add a new trusted publisher:
   - **Owner**: `KunjShah95`
   - **Repository**: `NEXUS-AI.io`
   - **Workflow name**: `publish-pypi.yml`
   - **Environment**: `pypi`

3. Repeat for TestPyPI with environment `testpypi`

### Trigger a Release

1. **Create a GitHub Release**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
   Then create a release on GitHub → Releases → Draft new release

2. **Manual Dispatch** (for testing):
   - Go to Actions → Publish to PyPI → Run workflow
   - Select target (testpypi or pypi)

---

## 🚀 Installing with uv

After publishing, users can install with uv:

```bash
# Install uv first
pip install uv

# Install aetherai using uv (much faster!)
uv pip install aetherai

# Or install globally
uv pip install aetherai --system
```

### uv in Projects

```bash
# Create a new project with uv
uv init my-project
cd my-project

# Add aetherai as dependency
uv add aetherai
```

---

## 📝 Version Management

When releasing a new version:

1. **Update versions** in:
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `setup.cfg` → `version = X.Y.Z`
   - `aetherai/__init__.py` → `__version__ = "X.Y.Z"`
   - `terminal/__init__.py` → `__version__ = "X.Y.Z"`

2. **Update CHANGELOG.md** with release notes

3. **Create git tag**:
   ```bash
   git add -A
   git commit -m "chore: bump version to X.Y.Z"
   git tag vX.Y.Z
   git push origin main --tags
   ```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `twine check` fails | Ensure README.md is valid markdown |
| Package name taken | Choose a unique name on PyPI |
| Import errors after install | Check `pyproject.toml` packages.find config |
| CLI not found after install | Verify `[project.scripts]` entry points |

### Verify Package Contents

```bash
# Check what's in the wheel
python -m zipfile -l dist/aetherai-1.0.0-py3-none-any.whl

# Check what's in the sdist
tar -tzf dist/aetherai-1.0.0.tar.gz
```

---

## 📊 Post-Publishing Checklist

- [ ] Package appears on PyPI: https://pypi.org/project/aetherai/
- [ ] `pip install aetherai` works
- [ ] `uv pip install aetherai` works
- [ ] `aetherai` command is available
- [ ] `nexus-ai` command is available (alias)
- [ ] README renders correctly on PyPI
- [ ] Version badge in README updates

---

## 🔗 Useful Links

- [PyPI Project Page](https://pypi.org/project/aetherai/)
- [TestPyPI Project Page](https://test.pypi.org/project/aetherai/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Python Packaging Guide](https://packaging.python.org/)
- [uv Documentation](https://github.com/astral-sh/uv)
