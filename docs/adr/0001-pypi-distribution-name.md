# PyPI distribution name is `pyCaRM-BHE`, not `pyCaRM`

The natural distribution name for this project is `pyCaRM`, matching the GitHub repo
(`BETALAB-team/pyCaRM`) and the import package (`carm`). At the time of the first PyPI
release, both `pyCaRM` and `CaRM` were already registered on PyPI by unrelated projects.
We chose `pyCaRM-BHE` (`pip install pyCaRM-BHE`) to keep the recognizable branding while
being distinct enough to register. The GitHub repo name and the `import carm` package
name are unaffected — only the string users type after `pip install` differs from both.
