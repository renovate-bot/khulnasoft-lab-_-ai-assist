# Release process for Self-managed AI Gateway

AI Gateway is deployed by the following ways:

- **Self-managed AI Gateway**: Deployed by customers when a new version of AI Gateway is released. The actual deployment method is TBD. See [this epic](https://gitlab.com/groups/gitlab-org/-/epics/13393) for more information.
- **GitLab-managed AI Gateway**: Deployed via Runway when a new commit is merged into `main` branch.

This release process is necessary to ensure compatibility between Self-managed GitLab Monolith and Self-managed AI Gateway.
It's NOT necessary for GitLab-managed AI Gateway as currently Runway deploys the latest SHA.

## Overview

We follow the [Semantic Versioning guideline](https://semver.org/),
which is rendered in [Conventional Commits](https://www.conventionalcommits.org/en) as an actual practice.
To harness the practice, we use [semantic-release](https://github.com/semantic-release/semantic-release) and [commitlint](https://github.com/conventional-changelog/commitlint).

In CI pipelines in AI Gateway:

- On merge requests:
  - `lint:commit` job runs to validate the commits in the feature branch if they are following Conventional Commits.
  - `publish-dryrun` job runs to make sure the commits are releasable via semantic-release.
- On `main` branch:
  - `publish` job can run manually to cut a new release and Git tag. This requires Maintainer+ access in AI Gateway project.
- On Git tags:
  - `release-docker-image:tag` job runs to pushes a new Docker image.
- On Git tags with format `gitlab-*`:
  - `release-docker-hub-image:tag` job runs to push a new Docker image to DockerHub. Requires `$DOCKERHUB_USERNAME` and `$DOCKERHUB_PASSWORD` to be set as CI variables.

In addition, we have [the expectations on backward compatibility](https://docs.gitlab.com/ee/architecture/blueprints/ai_gateway/#basic-stable-api-for-the-ai-gateway).
Tl;dr;

- We keep the API interfaces backward-compatible for the last 2 major versions.
- If a breaking change happens where we don't have a control (e.g. a depended 3rd party model was removed), we try to find a backward-compatible solution otherwise we bump a major version.

## View released versions of AI Gateway

To view released versions of AI Gateway, visit the following links:

- [Releases](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/releases): This page lists the released versions and changelogs.
- [Container Registry](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/container_registry): This page lists the released Docker images e.g. `registry.gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/model-gateway:v1.0.0`
- [DockerHub](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/container_registry): This page lists the released images that match a GitLab version e.g. `docker/gitlab/model-gateway:gitlab-v17.2`

### AI Gateway and GitLab releases compatibility

On each GitLab release, a new tag will be created named `gitlab-{gitlab-release}` on the same commit of the latest 
AI Gateway release before the cutoff. Users on self-hosted environments can use this to download a version of AI Gateway
that is compatible with their GitLab installation. These images are available both on 
[GitLab container registry](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/container_registry/3809284) 
and on [DockerHub](https://hub.docker.com/repository/docker/gitlab/model-gateway/tags).

## Release a new version of AI Gateway

1. Visit [the pipeline list on `main` branch](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/pipelines?page=1&scope=all&ref=main).
1. Select a pipeline that you want to publish.
1. Play `publish` job. This requires Maintainer+ access in AI Gateway project. If you're seeking for a help, ping a maintainer from [the dashboard](https://gitlab-org.gitlab.io/gitlab-roulette/?currentProject=ai-gateway).

This job automatically calculates the next version based on [the commit messages](https://www.conventionalcommits.org/en), cut a new Git tag and create a release.

## Configure release workflow

You can customize the release workflow via the configuration file `.releaserc.yml`. See [configuration page](https://github.com/semantic-release/semantic-release?tab=readme-ov-file#documentation).

If you want to set up maintenance/backport releases, see [this recipe](https://github.com/semantic-release/semantic-release/blob/master/docs/recipes/release-workflow/maintenance-releases.md).
