# UV Dependency Management Rules

When managing Python dependencies in this project, **always use UV** and follow these patterns to ensure reproducible, maintainable builds.

## Core Principles

- **Project mode is default**: Always use `pyproject.toml` + `uv.lock` for dependency management
- **Lockfile is source of truth**: `uv.lock` must be committed to version control
- **Never manual venv activation**: Use `uv run` for all commands that need the virtual environment
- **Explicit is better**: Always specify dependency groups (`--dev`, `--group`) when adding packages

## Essential Commands

### Project Lifecycle
- `uv init` - Scaffold new project (creates `pyproject.toml` + `.python-version`)
- `uv add <package>` - Add production dependency (updates both `pyproject.toml` and `uv.lock`)
- `uv add --dev <package>` - Add development dependency
- `uv add --group <name> <package>` - Add to custom dependency group
- `uv remove <package>` - Remove dependency from all groups
- `uv sync` - Install exact versions from `uv.lock`
- `uv lock` - Re-resolve and update lockfile
- `uv run <command>` - Execute command in managed virtual environment

### Legacy Compatibility
- `uv export --format requirements-txt > requirements.txt` - Generate pip-compatible requirements for external tools
- `uv pip install -r requirements.txt` - Only use when integrating with legacy systems

## Standard Workflows

### Starting a New Feature
```bash
uv add <new-dependency>
uv run pytest  # Test before committing
git add pyproject.toml uv.lock
```

### CI/CD Installation
```bash
uv sync --no-dev  # Production-only install
uv run python -m build  # Build in isolated environment
```

### Updating Dependencies
```bash
uv lock --upgrade  # Update all to latest compatible versions
uv lock --upgrade-package <pkg>  # Update specific package
uv sync  # Apply updates locally
```

## Critical Pitfalls (AVOID THESE)

- **NEVER** use `uv pip install` for routine development—it doesn't update `pyproject.toml`
- **NEVER** manually edit `uv.lock`—always use `uv add/remove/lock` commands
- **NEVER** commit `.venv` directory—it's auto-managed and platform-specific
- **NEVER** use `python -m pip` or `pip` directly—always prefix with `uv`
- **ALWAYS** run `uv sync` after pulling changes to `uv.lock` from version control

## Project Structure

```
project/
├── pyproject.toml          # Dependency declarations
├── uv.lock                 # Exact version pins (COMMIT THIS)
├── .python-version         # Python version pin
├── .venv/                  # Auto-managed (add to .gitignore)
└── src/
    └── package/
        └── __init__.py
```

## Python Version Management

- `uv python install 3.12` - Install specific Python version
- `uv python pin 3.12` - Pin version for current project (creates `.python-version`)
- UV respects `.python-version` in all commands automatically

---

**Rule Enforcement**: Any dependency-related commands not following these patterns must be flagged and corrected before proceeding.