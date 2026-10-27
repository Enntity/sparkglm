<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Tool-call pattern guard

SparkGLM 0.2.0 and Atlas block recognizable exfiltration patterns in tool calls
by default, in Strict and Spark modes, TP2 and TP4. This is a bounded CPU filter
over parsed arguments. There is no second model or GPU pass. Ordinary assistant
prose and reasoning are not scanned; tool calls wait until their complete
arguments pass.

The guard is part of the common Spark engine SparkGLM shares with SparkQwen
(0.3.3 shipped the same policy). It runs after GLM's tool parser, on the calls
that parser releases. Literal tool XML that GLM writes inside its reasoning stays
text and is never hoisted into an executable call, so it is not screened either.

## What is checked

The guard recognizes these argument fields, case-insensitively:

| Role | Fields |
|---|---|
| Destination | `url`, `uri`, `endpoint`, `host`, `hostname`, `server`, `target`, `destination` |
| Command | `command`, `cmd`, `script`, `shell`, `exec`, `argv`; sibling `args` or `arguments` supplement a command |
| Network payload | `body`, `data`, `json`, `payload`, `query`, `params`, `form`, `content`, `headers` |
| Upload source | `file`, `files`, `file_path`, `filepath`, `source`, `source_path`, `local_path`, `input_file`, `upload_file` |

Destination fields accept literal URLs or bare hosts/authorities, including ports
and bracketed IPv6, plus scheme-relative URLs. Destination objects can wrap
those fields or `href`. Command fields accept strings, argument arrays, or
objects with `program`/`executable` and `args`/`arguments`. Literal
URLs and common `curl`, `wget`, `scp` and `rsync` forms are recognized, including
absolute executable paths, multipart and attached upload flags, and up to three
literal `sh`/`bash`/`zsh`/`dash` `-c`/`-lc` wrappers. This is lexical inspection,
not a shell or program interpreter. Arbitrary numeric command destinations need
an explicit URL; bare numeric forms are recognized only for metadata checks.

The default checks block:

- Recognized literal cloud metadata addresses and names, including HTTP GETs.
- The specific callback hosts `sanmaohe.oss-cn-beijing.aliyuncs.com` and
  `routify-file-proxy-sg.oss-ap-southeast-1.aliyuncs.com`, plus operator denials.
- Recognizable credential-file/environment transfers, private-key blocks,
  selected token prefixes and credential payload fields going to a destination
  without a host exception. Sensitive query strings can be blocked even on GETs.
- Invalid JSON/object arguments, incomplete recognized calls and calls exceeding
  the byte limit.

Credential patterns include `.env` files, common SSH/cloud credential paths,
private-key PEM markers, secret environment-variable names and token prefixes
`ghp_`, `github_pat_`, `xoxb-` and `xoxp-`. This is a finite pattern set.

Destination and payload checks are scoped to the same request object. Nested
request/batch objects are checked independently. Document `content`, `text`,
`code`, notes and descriptions are not interpreted as destinations or commands.
`content` is checked for credential patterns when its object also has a network
destination. Quoting a metadata URL in a local document therefore passes.

**Normal authentication is deliberately allowed:** `Authorization`,
`Proxy-Authorization` and `X-API-Key` token headers do not require a trusted host.
This means the guard does not bind credentials to approved destinations.
Private-key blocks are still treated as sensitive material.

## Reads, writes and limits

| Example | Guard behavior |
|---|---|
| Local credential-file read without a transfer | Allowed |
| GET to a recognized metadata destination | Blocked |
| Recognizable credential upload | Blocked |
| Ordinary private-document upload | Allowed without a recognized credential pattern |
| Ordinary prose or a local document quoting a blocked URL | Allowed |
| Authentication token sent to an arbitrary unblocked host | Allowed by the authentication exception |

Tool names do not exempt arguments. The recognized fields above define coverage;
custom tools with different schemas or argument layouts may not be covered.
A destination and credential in different objects are not associated, even
within one tool call.
Opaque document/payload objects are not recursively interpreted as executable
requests. The guard does not inspect tool results, read files, identify all
private information, or track data between separate calls. It cannot evaluate
arbitrary code, reconstruct encoded/assembled destinations, resolve DNS, follow
redirects, or police a client renderer.

Conservative checks can still reject legitimate login payloads or command fields
that quote a suspicious example. Command scanning can mistake dotted payload
words for additional hosts, including in an otherwise allowlisted upload; use
a structured request tool when this matters. Review the reason and use a narrow host exception
where appropriate. Stronger guarantees require sensitive-file access controls,
credential-to-destination binding and network policy in the tool executor.
There is no provider-wide or country-based blocklist. The
[Cohere report](https://cohere.com/blog/concerning-behaviour-popular-open-weight-ai-model)
does not establish a deliberate model backdoor.

## Configure

Set these in `.env` before starting the pair:

```sh
TOOL_GUARD=on
TOOL_GUARD_MAX_BYTES=1048576
# Optional comma-separated exact hosts:
#TOOL_GUARD_ALLOW_HOSTS=uploads.example.org
#TOOL_GUARD_DENY_HOSTS=blocked.example.org
```

`start.sh` forwards the settings to both ranks; `install/start-node.sh` takes
`--tool-guard`, `--tool-guard-max-bytes`, `--tool-guard-allow-hosts` and
`--tool-guard-deny-hosts`, the same on every rank. Direct container and LLooM
recipe operators use `SPARKGLM_TOOL_GUARD`, `SPARKGLM_TOOL_GUARD_MAX_BYTES`,
`SPARKGLM_TOOL_GUARD_ALLOW_HOSTS`, and `SPARKGLM_TOOL_GUARD_DENY_HOSTS`; unset,
the guard is on with the 1 MiB limit.
Atlas's flags are `--tool-guard`, `--tool-guard-max-bytes`,
`--tool-guard-allow-host` and `--tool-guard-deny-host`. Requests cannot change
server policy; performance profiles cannot override these flags.

Host exceptions are exact, without subdomain wildcards. Explicit denials and
metadata protection take priority. An intentional metadata workflow needs
`TOOL_GUARD=off`; isolate that workflow. Limits are 1 to 67108864 bytes.

## Responses and logs

A block produces a fixed refusal with no blocked tool-call header or arguments.
Chat Completions uses `finish_reason: "content_filter"` and
`guard_stop: "tool_call_guard"`; Anthropic uses `stop_reason: "refusal"`.
Responses returns the fixed notice as `output_text` in a completed response,
without a structured refusal part or guard marker; the call is still withheld.
Earlier safe calls already released in a streamed turn cannot be recalled.

Each blocked response emits a WARN event with a reason code:
`metadata_access`, `blocked_host`, `credential_transfer`, `invalid_arguments`,
`arguments_too_large` or `incomplete_call`. The guard does not include rejected
arguments or credentials in that event. Allowlisted passes have no separate
audit event. Other application logs are outside this guarantee.
