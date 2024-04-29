require "gitlab-dangerfiles"

Gitlab::Dangerfiles.for_project(self, 'ai-gateway') do |dangerfiles|
  # Import all plugins from the gem
  dangerfiles.import_plugins

  # Import a defined set of danger rules
  dangerfiles.import_dangerfiles(only: %w[roulette type_label subtype_label z_retry_link])
end
