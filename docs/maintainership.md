# Maintainership

**NOTE: Currently, this project is short-handed. Please consider becoming a maintainer!**

As of today, AI Gateway is a small project with less than 10 internal contributors,
hence, we follow [the common guideline](https://handbook.gitlab.com/handbook/engineering/workflow/code-review/#maintainership-process-for-smaller-projects)
for a new reviewer, trainee maintainer and maintainer.
If you have a question, please reach out one of the maintainers that you can find in the [GitLab Review Workload Dashboard](https://gitlab-org.gitlab.io/gitlab-roulette/?currentProject=ai-gateway).

## Preferred domain knowledge

To maintain AI Gateway project, the following domain knowledge is preferred:

- [Python](https://www.python.org) as the primary programming language.
- [Poetry](https://python-poetry.org) as the package and virtual environment manager.
- [FastAPI](https://fastapi.tiangolo.com/) as the modern web framework.
- [Architectural blueprint](https://docs.gitlab.com/ee/architecture/blueprints/ai_gateway/) to understand how the AI Gateway is integrated with the other components.

In addition, it would be better to familiarize yourself with the following observability services:

- [Metrics](https://dashboards.gitlab.net/d/ai-gateway-main/ai-gateway3a-overview?orgId=1) to understand the application performance on production.
- [Logs](https://log.gprd.gitlab.net/app/r/s/zKEel) to investigate the bugs on production.
- Alerts in `g_mlops-alerts` Slack channel.

## How to become a maintainer

While there is no strict guideline how to become a maintainer, we generally recommend the following activities before submitting the request:

- Author more than or equal to 5 MRs.
- Review more than or equal to 5 MRs as a trainee maintainer.
- Familiarize yourself with the [preferred domain knowledge](#preferred-domain-knowledge).

When it's ready:

- Create a merge request and indicate your role as an `ai-gateway: maintainer` in your [team member entry](https://gitlab.com/gitlab-com/www-gitlab-com/blob/master/doc/team_database.md).
- Assign MR to your manager and AI Gateway maintainers for merge.
- After the MR is merged, ask an Owner of [`@gitlab-org/maintainers/ai-gateway`](https://gitlab.com/groups/gitlab-org/maintainers/ai-gateway/-/group_members?with_inherited_permissions=exclude) group to add you. It will make you one of the [Code Owners](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/.gitlab/CODEOWNERS?ref_type=heads) whose approval is required for an MR to be merged.
- Add your username to the `members` of the `ai-gateway` entry in the [`inventory.json`](https://gitlab.com/gitlab-com/gl-infra/platform/runway/provisioner/-/blob/main/inventory.json?ref_type=heads). It will make you one of the maintainers in [the deployment project](https://gitlab.com/gitlab-com/gl-infra/platform/runway/deployments/ai-gateway), which allows you to deploy via Runway.
