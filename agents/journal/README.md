# Agent Experiment & Incident Journal

This directory contains daily logs of agent experiments, bug investigations, and operational incidents. Each day gets its own file to keep logs manageable and easy to navigate.

## Structure

- **Daily logs**: `YYYY-MM-DD.md` (e.g., `2025-11-21.md`, `2025-11-22.md`)
- **Index**: This README serves as the entry point and guide

## Logging Rules

1. **File naming**: Use `YYYY-MM-DD.md` format for daily logs
2. **Create on demand**: Create a new file for today when you log the first entry of the day
3. **Reverse chronological within day**: Newest entries first within each daily file
4. **All fields required**: Every entry must have Owner, Agents affected, Git commit, Context, Actions, Result, References
5. **Timestamps**: ALWAYS use system time via `date '+%Y-%m-%d-%H%M'` or similar command - NEVER make up timestamps
6. **Link everything**: Related commits, issues, or external documents for traceability
7. **Git commits**: Reference the commit the work was **based on** (starting point), not the commit it will create
8. **Explicit deviations**: When experiments differ from documented config, describe the delta
9. **Health audits**: If no work occurs for two sprints, log that you performed the scheduled audit
10. **Detail level**: 
    - **Concise entries**: Keep each field to 1-2 sentences for small changes (bug fixes, UI tweaks, minor refactors)
    - **Detailed entries**: Expand fields with sub-bullets for major changes (architectural refactors, breaking changes, new features)

## Entry Template

**All entries use this template** - vary the detail level based on change size:

```markdown
### YYYY-MM-DD-HHMM – Short descriptive title

- **Owner**: AI Agent (or name) + role (e.g., "bug fix", "refactoring", "UI polish")
- **Agents affected**: Which components/modules were touched
- **Git commit(s)**: Based on <commit-hash> (use `git log -1 --format="%H"`)
- **Context**: Brief why/background (1-2 sentences for small changes, paragraph for large)
- **Actions**: What was done (list for small, sub-bullets for large)
- **Result**: Outcome and key notes (1-2 sentences for small, multiple bullets for large)
- **Breaking changes**: (if any - omit if none)
- **Next steps**: (if any - omit if none)
- **Lessons learned**: (optional, omit for trivial changes)
- **References**: File paths or links
```

### Examples

**Small change (bug fix)**:
```markdown
### 2025-11-22-1603 – Fixed horizontal scrolling

- **Owner**: AI Agent (bug fix)
- **Agents affected**: Orchestrator demo TUI
- **Git commit(s)**: Based on 95f3b83f...
- **Context**: Accidental horizontal scrolling broke layout
- **Actions**: Added `overflow-x: hidden` to all panels in CSS
- **Result**: No more horizontal scroll
- **References**: `examples/orchestrator_demo.py` (lines 53-79)
```

**Large change (refactor)**:
```markdown
### 2025-11-22-1534 – Merged ContinuousPDDLIntegration into TaskOrchestrator

- **Owner**: AI Agent (major refactoring)
- **Agents affected**: TaskOrchestrator, continuous_pddl_integration_demo
- **Git commit(s)**: Based on 95f3b83f...
- **Context**: Two similar orchestration classes caused duplication. User requested...
- **Actions**:
  - Removed separate `DetectedObjectRegistry` from orchestrator
  - Use tracker's registry directly for consistency
  - Replaced timer-based auto-save with event-driven saves
  - Updated demo app to use TaskOrchestrator directly
- **Result**: Single source of truth, cleaner separation of concerns
- **Breaking changes**: `ContinuousPDDLIntegration` class removed (not public API)
- **References**: `src/planning/task_orchestrator.py`, `examples/orchestrator_demo.py`
```

## Quick Reference

**To log a new entry:**
1. Open or create `YYYY-MM-DD.md` for today (use `date '+%Y-%m-%d'`)
2. Get the current timestamp: `date '+%Y-%m-%d-%H%M'`
3. Get base commit: `git log -1 --format="%H"`
4. Add entry at top with all required fields (Owner, Agents affected, Git commit, Context, Actions, Result, References)
5. Keep each field brief for small changes, expand for major ones
6. Commit the change

**Field length guidance:**
- **Small changes**: 1-2 sentences per field, simple lists
- **Major changes**: Paragraph for Context, sub-bullets for Actions, multiple Result points

**To find entries:**
- By date: Open the corresponding `YYYY-MM-DD.md` file
- By topic: Use grep/search across journal files
- Recent activity: Check the most recent date files

## Archive Policy

- Keep all daily logs indefinitely (they're small text files)
- If a sprint has no agent activity, create an entry noting the health audit
- Old logs remain valuable for understanding system evolution

## Contributing

When logging experiments or incidents, remember:
- **All required fields, always**: Every entry needs Owner, Agents affected, Git commit, Context, Actions, Result, References
- **Match verbosity to impact**: Bug fix = 1 sentence per field, Major refactor = paragraphs and sub-bullets
- **Optimize for scanning**: Title + Context should give the gist at a glance
- **Link to code**: Always include file paths and line numbers
- **Capture insights**: What you learned, not just what you did
- **Update related docs**: Keep architecture.md and playbook.md in sync with significant changes

