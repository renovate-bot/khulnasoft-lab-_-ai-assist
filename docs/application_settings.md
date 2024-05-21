# Application settings

All of the appllication settings should be defined in `example.env`, which is a list of environment variables
with a specific format.

## How to add a new setting

We are using [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
to parse the dotenv file into pydantic objects. The environment variables are formatted in the following way:

```
AIGW_{group-of-setting}__{name-of-setting}
```

- `AIGW_` ... Prefix that must be added to all of the keys.
- `{group-of-setting}__` ... Name of the group that gathers related setting items into one place.
  You can choose an existing one or add a new one. Notice that you need to put _two_ underscores `__`.
- `{name-of-setting}` ... Name of the setting.

These values will be interpreted as pydantic objects (e.g. `Config`) that can be accessed in
dependency setup process powered by [Dependency Injector](https://python-dependency-injector.ets-labs.org/).

Here is an example of process flow:

1. You add a new application setting `AIGW_AWESOME_FEATURE__MAX_TOKENS` in `example.env`.
   `example.env` is the template of application settings, which is copied to `.env` as an actual settings during [the installation process](../README.md#how-to-run-the-server-locally).
1. This is interpreted as `ConfigAwesomeFeature` pydantic group object, which has an field named `max_tokens`.
   You can access to the value via `config.awesome_feature.max_tokens`.
1. This config object is passed to `ContainerAwesomeFeature` that defines how it should initialize your business logic class `AwesomeFeature`
   in application runtime. The simplest example is [FactoryProvider](https://python-dependency-injector.ets-labs.org/providers/factory.html).
1. This provider is defined as one of the FastAPI's endpoint [dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/).
   When a user requests to the endpoint, FastAPI initializes an instance of the class `AwesomeFeature`, which you can use in the endpoint.

A few notes:

- `python-dotenv` will treat any value as a string, so specifying `None` maps to the Python value `'None'`.

## Avoid fetching environment variable directly

In general, we should not fetch an environment variable directly in application runtime.
We should follow the [convention of application setting](#how-to-add-a-new-setting) instead,
so that we can easily track what settings are available by just looking at `example.env`.

For example, avoid the following pattern:

```python
# Bad
awesome_feature = AwesomeFeature(
   max_tokens=os.environ["AIGW_AWESOME_FEATURE__MAX_TOKENS"]
)
```

Instead, you should do:

1. Add the application setting to `example.env` and `.env`.
   ```shell
   AIGW_AWESOME_FEATURE__MAX_TOKENS=100
   ```
1. Add the config group for the feature:
   ```python
   # ai_gateway/config.py
   class ConfigAwesomeFeature(BaseModel):
      max_tokens: int = 0
   ```
1. Declare dependencies for the feature:
   ```python
   # ai_gateway/container.py
   class ContainerAwesomeFeature(containers.DeclarativeContainer):
      config = providers.Configuration(strict=True)

      awesome_client = providers.Factory(
         AwesomeClient,
         max_tokens=config.awesome_client.max_tokens, # This value is fetched from `.env`.
      )
   ```
1. Add an endpoint to access the feature:
   ```python
   # ai_gateway/api/v1/awesome_feature.py
   @router.post("/awesome_feature")
   async def awesome_feature(
      request: Request,
      awesome_client: AwesomeClient = Depends(Provide[ContainerAwesomeFeature.awesome_client]),
   ):
      return awesome_client.predict('Hi, how are you?')
   ```

You can also silence the warning if it's a legitimate usage or to be followed-up in the future:

```python
# RUNWAY_REGION is the variable injected by Runway provisioner.
os.getenv("RUNWAY_REGION", default) # pylint: disable=direct-environment-variable-reference
```
