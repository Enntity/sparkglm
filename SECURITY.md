# Security policy

Do not open a public issue containing credentials, private network details,
model-provider tokens, SSH material, or vulnerabilities that would expose a
running inference appliance.

Before reporting a serving vulnerability, reproduce it against a clean
checkout without private model paths or access tokens. Contact the repository
owner privately through GitHub for issues that require coordinated disclosure.

This research software exposes an HTTP model-serving endpoint. Operators are
responsible for authentication, network isolation, rate limiting, model-license
compliance, and keeping the two-node control network private.
