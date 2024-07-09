# Tests

This project uses [Pytest](https://docs.pytest.org/en/stable/) for testing.

## Unit test

To run the entire unit test suite, you can use the following command:

```shell
make test
```

To run the tests in watch mode, use the following command:

```shell
make test-watch
```

To see test coverage, you can run the following command:

```shell
make test-coverage
```

This will run all the tests, output coverage in the terminal and generate an HTML report.
You can view the HTML report by running:

```shell
open htmlcov/index.html
```

## Integration test

Integration tests verify that the API endpoints correctly handle requests and return appropriate responses.
Unlike unit testing, integration tests send requests to the server as a black-box component without mocking application code.

- Integration tests are located at `integration_tests` directory.
- Integration tests are slower than unit tests. Adding too many test cases could result in slower CI pipelines, which could affect productivity negatively.
- According to [the GitLab testing guide](https://docs.gitlab.com/ee/development/testing_guide/testing_levels.html#white-box-tests-at-the-system-level-formerly-known-as-system--feature-tests),
it's ideal to just cover happy path, unhappy path and regression cases. The rest of the low-level testing should be covered in unit testing.

### Run integration test locally

Update your [application settings](application_settings.md) file:

```shell
# .env

# Enable authentication and authorization by default.
# It can be skipped only when "Bypass-Auth" header is specified. We use it for getting the self-signed JWT at the initial request.
AIGW_AUTH__BYPASS_EXTERNAL=false
AIGW_AUTH__BYPASS_EXTERNAL_WITH_HEADER=true
```

Restart you local AI GW: `poetry run ai_gateway`. You need local AI GW up and running during the test:

```shell
poetry run ai_gateway
```

Open a new terminal and run the following command:

```shell
# Adjust `AI_GATEWAY_URL` for your local AI GW url

export AI_GATEWAY_URL=http://localhost:5052
make test-integration
```

## Code guidelines

- Avoid using [provider overriding](https://python-dependency-injector.ets-labs.org/providers/overriding.html),
since it can lead to divergences between test and runtime behavior. See
[this issue](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/511)
for more details.
