# frozen_string_literal: true

require 'openssl'
require 'yaml'

module Gitlab
  module Llm
    module Embeddings
      module Utils
        MAX_CHARS_PER_EMBEDDING = 1500
        MIN_CHARS_PER_EMBEDDING = 100
        METADATA_REGEX = /\A---(?<metadata>.*?)---/m

        class BaseContentParser
          attr_reader :max_chars_per_embedding, :min_chars_per_embedding, :root_url

          def initialize(root_url)
            @max_chars_per_embedding = MAX_CHARS_PER_EMBEDDING
            @min_chars_per_embedding = MIN_CHARS_PER_EMBEDDING
            @root_url = root_url
          end

          def parse_and_split(content, source_name)
            items = []
            md5sum = ::OpenSSL::Digest::SHA256.hexdigest(content)
            content, metadata = parse_content_and_metadata(content, md5sum, source_name)

            split_by_newline_positions(content) do |text|
              next if text.nil?
              next unless text.match?(/\w/)

              items << {
                content: text,
                metadata: metadata
              }
            end
            items
          end

          def parse_content_and_metadata(content, md5sum, source_name)
            match = content.match(METADATA_REGEX)
            metadata = if match
                         metadata = YAML.safe_load(content.match(METADATA_REGEX)[:metadata])
                         content = match.post_match.strip
                         metadata
                       else
                         {}
                       end

            metadata['title'] = title(content)
            metadata['md5sum'] = md5sum
            metadata['source'] = source_name
            metadata['source_type'] = 'doc'
            metadata['source_url'] = url(source_name)

            [content, metadata]
          end

          def split_by_newline_positions(content)
            if content.length < max_chars_per_embedding && content.length >= min_chars_per_embedding
              yield content
              return
            end

            positions = content.enum_for(:scan, /\n/).map { Regexp.last_match.begin(0) }

            cursor = 0
            while position = positions.select { |s| s > cursor && s - cursor <= max_chars_per_embedding }.max
              if content[cursor...position].length < min_chars_per_embedding
                cursor = position + 1
                next
              end

              yield content[cursor...position]
              cursor = position + 1
            end

            while cursor < content.length
              content[cursor...].chars.each_slice(max_chars_per_embedding) do |slice|
                if slice.length < min_chars_per_embedding
                  yield nil
                  cursor = content.length
                  next
                end

                yield slice.join("")
                cursor += slice.length
              end
            end
          end

          def url(source_name)
            page = source_name.gsub('/doc/', '').gsub('.md', '')
            "#{root_url}/#{page}"
          end

          def title(content)
            return unless content

            match = content.match(/#+\s+(?<title>.+)\n/)

            return unless match && match[:title]

            match[:title].gsub(/\*\*\(.+\)\*\*$/, '').strip
          end
        end
      end
    end
  end
end
