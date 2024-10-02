# Feature Flags

See this [doc](https://docs.gitlab.com/ee/development/ai_features/index.html#push-feature-flags-to-ai-gateway).

## Prevent a feature flag for a specific GitLab realm

You can prevent a feature flag for a specific GitLab realm by the `AIGW_FEATURE_FLAGS__DISALLOWED_FLAGS` setting.

For example, if you want to prevent the `expanded_ai_logging` feature flag on self-managed realm,
[update your application settings file](./application_settings.md#how-to-update-application-settings) as the following:

```shell
AIGW_FEATURE_FLAGS__DISALLOWED_FLAGS='{"self-managed": ["expanded_ai_logging"]}'
```
