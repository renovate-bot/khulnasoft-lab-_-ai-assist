## What does this merge request do and why?

<!-- Briefly describe what this merge request does and why. -->

%{first_multiline_commit}

## How to set up and validate locally

_Numbered steps to set up and validate the change are strongly suggested._

<!--
Example below:

1. Check out to this merge request's branch.
1. Ensure a local Docker image built successfully.
   ```shell
   docker build --platform linux/amd64 -t ai-gateway:dev .
   ```
1. Run a local service on Docker.
   ```shell
   docker run --platform linux/amd64 --rm \
     -p 5052:5052 \
     -e AIGW_AUTH__BYPASS_EXTERNAL=true \
     -v $PWD:/app -it ai-gateway:dev
   ```
-->

## Merge request checklist

- [ ] Tests added for new functionality. If not, please raise an issue to follow up.
- [ ] Documentation added/updated, if needed.

/label ~"group::ai framework"

<!-- Select a type -->
<!-- /label ~"type::bug" -->
<!-- /label ~"type::feature" -->
<!-- /label ~"type::maintenance" -->

/assign me
