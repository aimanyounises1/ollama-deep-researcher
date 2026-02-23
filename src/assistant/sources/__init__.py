"""sources/ — Enterprise data source connectors.

Organized by source system:
- jira_client.py   : Low-level Jira API client
- jira_tool.py     : High-level JiraProject tool with query expansion
- jira_runner.py   : LangGraph node runner for Jira fetch
- jira_patch.py    : Jira session monkey-patch + utilities
- confluence.py    : Confluence search and fetch
- perforce.py      : Perforce node runner
- perforce_client.py: Full P4 client wrapper (PerforceHelper)
- solution_book.py : Solution book knowledge base tool
"""
