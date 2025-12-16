# Error Codes and Schemas

## Error Response Schema

All non-2xx responses return:

```json
{
  "error": {
    "code": "STRING_IDENTIFIER",
    "message": "Human-readable summary",
    "details": {},
    "correlation_id": "uuid"
  }
}
```

- **code**: Machine-readable constant for client switching.
- **message**: Safe, loggable description.
- **details**: Optional structured extra info.
- **correlation_id**: UUID for tracing.

## Error Codes

| HTTP | Code | Meaning |
|------|------|---------|
| 400 | `BAD_REQUEST` | Malformed request. |
| 400 | `VALIDATION_ERROR` | Body/field validation failed. |
| 401 | `UNAUTHENTICATED` | Missing/invalid API key. |
| 403 | `FORBIDDEN` | Authenticated but not allowed. |
| 404 | `NOT_FOUND` | Resource not found. |
| 409 | `CONFLICT` | Duplicate/conflicting resource. |
| 422 | `SEMANTIC_ERROR` | Input is valid but logically impossible. |
| 429 | `RATE_LIMIT_EXCEEDED` | Rate limit hit. |
| 500 | `INTERNAL_ERROR` | Unexpected server error. |
| 502 | `UPSTREAM_ERROR` | LLM/Qdrant/dependency failed. |
| 503 | `SERVICE_UNAVAILABLE` | Maintenance or overload. |

## Examples

### Validation Error

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request body validation failed.",
    "details": {
      "top_k": "Must be >= 1"
    },
    "correlation_id": "abc-def-123"
  }
}
```

### Upstream Error (OpenAI timeout)

```json
{
  "error": {
    "code": "UPSTREAM_ERROR",
    "message": "Language model request failed.",
    "details": {
      "provider": "openai",
      "reason": "timeout"
    },
    "correlation_id": "xyz-789-456"
  }
}
```

### Authentication Required

```json
{
  "error": {
    "code": "UNAUTHENTICATED",
    "message": "Authentication is required for this endpoint.",
    "details": {},
    "correlation_id": "auth-123-456"
  }
}
```
