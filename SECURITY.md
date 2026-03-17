# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately:

**Email**: virtaava@gmail.com

Please do **not** open a public GitHub issue for security vulnerabilities.

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

I will respond within 72 hours and work with you on a fix before any public disclosure.

## Known Considerations

- **Prompt injection**: Memory content is included in LLM prompts. Stored memories are
  tagged as recollections to mitigate injection, but defense-in-depth depends on the
  underlying LLM's instruction-following robustness.
- **API server**: The OpenAI-compatible proxy binds to localhost only. It has no
  authentication by default and is intended for local development use.
- **Data isolation**: Characters sharing a database are isolated by `character_id`
  filters on all queries. Multi-tenant deployments should use separate database files.
