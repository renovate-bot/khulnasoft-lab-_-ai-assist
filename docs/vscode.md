# Enabling Code Suggestions in VSCode

This file contains the step-by-step guide explaining how to enable Code Suggestions (CS) in VSCode with 
the GitLab extension installed post Milestone ["Code Suggestions Gated MVC"](https://gitlab.com/groups/gitlab-org/modelops/applied-ml/code-suggestions/-/epics/2).

1. Install the GitLab VSCode [extension](https://marketplace.visualstudio.com/items?itemName=GitLab.gitlab-workflow). 
   This guide assumes that we're installing a version `>=3.59.2`.
2. For Mac users: open the command palette by pressing `Shift+Command+P` and type `GitLab: Add account to VS Code`.
3. Insert the URL to your GitLab instance and the personal access token with scopes: `read_user`, `read_api`.
4. Go to the GitLab extension settings and click on `Enable code completion`.
5. For Mac users: open the command palette by pressing `Shift+Command+P` and type `Git: Clone`.
6. Insert the repository URL address that you want to clone from GitLab. Please, note that Code Suggestions can be 
   enabled for GitLab projects only at this iteration.
7. Voila! Open the cloned project in VSCode and get code completions in newly created or existing files.
