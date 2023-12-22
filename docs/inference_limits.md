# Inference limits

Model engines can enforce concurrency limits. This puts a limit to the
number of inferences we can run concurrently on a single model, or the
entire engine.

At the time of writing, only Anthropic limits the concurrent
inferences. We have a larger allowed concurrency for `claude-2.1` than
we do for `claude-2.0`, and the enforced limit across all models is
equal to that of `claude-2.0`. The metrics [for
Anthropic](#anthropic-metrics) are emitted from the application and
need to be kept up-to-date if when we receive increases or start using
new models.

Vertex limits the number of inferences we can run per minute. But has
no limit on the concurrent requests. These metrics are available
through stackdriver and will automatically adjust when we use new
models or have increases.

## Observability of utilization

Like all of the metrics for the AI-gateway. These saturation metrics
are also visible on the [service overview
dashboard](https://dashboards.gitlab.net/d/ai-gateway-main/ai-gateway3a-overview?orgId=1). The
saturation panel on the top right contains all saturation components
that apply to the AI-gateway.

The components regarding limits enforced by model engines are:
- [`max_concurrent_inferences`](https://dashboards.gitlab.net/d/alerts-max_concurrent_inferences/154abead-92ad-5cd7-9112-fd8418ba289b?var-environment=gprd&var-type=ai-gateway&var-stage=main&var-component=max_concurrent_inferences&orgId=1):
  Per model concurrent requests, enforced by Anthropic.
- [`concurrent_inferences_per_engine`](https://dashboards.gitlab.net/d/alerts-max_inferences_per_engine/bd1e5cca-760c-55b0-98fa-4501e273af2a?var-environment=gprd&var-type=ai-gateway&var-stage=main&var-component=max_concurrent_inferences_per_engine&orgId=1):
  Per engine concurrent requests across all models, enforced by
  Anthropic. The limit is the highest number of requests allowed to a
  model. In our case `claude-2.1`.
- [`gcp_quota_limit_vertex_ai`](https://dashboards.gitlab.net/d/alerts-sat_gcp_quota_limit_vertex_ai/d6ff3868-f03d-5dda-bd7c-e0fd406c5cc6?var-environment=gprd&var-type=ai-gateway&var-stage=main&var-component=gcp_quota_limit_vertex_ai&orgId=1):
  Per model rate limit for vertex models.

Each of these dashboards also gives a bit more explanation on how much
we're currently using of our limit presented in a percentage. These
dashboards are also linked from the saturation panel on the service
overview.

These saturation points currently alert, but they will not page the
SRE-oncall because the severity of the saturation points is set lower
than S2.

## Anthropic metrics

Unlike Google, Anthropic does not expose an API that tells us our
limits and current utilization. So we're counting this in the
AI-gateway itself. See the `ModelRequestInstrumentator` in
`ai_gateway/instrumentators/model_requests.py` for these metrics.

| Metric name                       | Labels                       | Explanation                                                                |
|-----------------------------------|------------------------------|----------------------------------------------------------------------------|
| `model_inferences_in_flight`      | `model_engine`, `model_name` | Incremented at the start of a request to an engine, decremented at the end |
| `model_inferences_max_concurrent` | `model_engine`, `model_name` | Set when a model is first used                                             |

The metric for the limits is configured through an environment variable called
`MODEL_ENGINE_CONCURRENCY_LIMITS`. It's currently only used for
`anthropic` but could be used for different model engines. The
variable needs to be set in JSON with this format:

```json
{
  "<engine-name>": { "<model-name>": integer-limit }
}
```

For example for Anthropic (these are not our actual limits):

```json
{ "anthropic": { "claude-2.0": 5, "claude-2.1": 15 } }
```

Because we don't want to share the limits we got from providers, this
environment variable is configured in vault. [Similar to other secrets
in
Runway](https://gitlab.com/gitlab-com/gl-infra/platform/runway/docs/-/blob/master/secrets-management.md?ref_type=heads). So
in this case the variable is available at the following locations:

- Production:
  [`env/production/service/ai-gateway/MODEL_ENGINE_CONCURRENCY_LIMITS`](https://vault.gitlab.net/ui/vault/secrets/runway/kv/env%252Fproduction%252Fservice%252Fai-gateway%252FMODEL_ENGINE_CONCURRENCY_LIMITS/details)
- Staging: [`env/staging/service/ai-gateway/MODEL_ENGINE_CONCURRENCY_LIMITS`](https://vault.gitlab.net/ui/vault/secrets/runway/kv/env%2Fstaging%2Fservice%2Fai-gateway%2FMODEL_ENGINE_CONCURRENCY_LIMITS/details?version=1)

When we introduce new models, and those models have a higher limit due
to performance improvements on Anthropic's side, or a lower limit. We
need to update those environment variables.
