#!/usr/bin/env ruby

# frozen_string_literal: true

puts "clone dir: #{ENV['GITLAB_DOCS_CLONE_DIR']}"

DOC_DIR = ENV['GITLAB_DOCS_CLONE_DIR']
ROOT_URL = ENV.fetch('GITLAB_DOCS_WEB_ROOT_URL')
METADATA_KEYS = %w{title md5sum source source_type, source_url}

require 'json'
require_relative "base_content_parser"
require_relative "docs_content_parser"

def parse(filenames)
  filenames.map do |filename|
    source = filename.sub("#{DOC_DIR}/doc/", '')

    puts "parsing: { filename: #{filename}, source: #{source} }"

    ::Gitlab::Llm::Embeddings::Utils::DocsContentParser.parse_and_split(
      File.read(filename),
      source,
      'doc',
      root_url: ROOT_URL
    )
  end
end

def export(entries)
  log_name = ENV.fetch('GITLAB_DOCS_JSONL_EXPORT_PATH')
  File.delete(log_name) if File.exist?(log_name)
  File.open(log_name, 'w') do |f|
    entries.flatten.each do |entry|
      entry = entry.dup
      entry[:metadata] = entry[:metadata].slice(*METADATA_KEYS)
      f.puts JSON.dump(entry)
    end
  end
end

def execute
  entries = parse(Dir.glob("#{DOC_DIR}/doc/**/*.md"))
  export(entries)
end

execute
