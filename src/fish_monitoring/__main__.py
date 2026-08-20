"""Allow ``python -m fish_monitoring`` to invoke the CLI."""
from fish_monitoring.cli import main

raise SystemExit(main())
