# frozen_string_literal: true

require 'openssl'
require 'yaml'

module Gitlab
  module Llm
    module Embeddings
      module Utils
        class BaseContentParser
          attr_reader :max_chars_per_embedding, :min_chars_per_embedding, :root_url

          def initialize(min_chars_per_embedding, max_chars_per_embedding, root_url:)
            @max_chars_per_embedding = max_chars_per_embedding
            @min_chars_per_embedding = min_chars_per_embedding
            @root_url = root_url
          end

          def parse_and_split(content, source_name, source_type)
            items = []
            md5sum = ::OpenSSL::Digest::SHA256.hexdigest(content)
            content, metadata, url = parse_content_and_metadata(content, md5sum, source_name, source_type)

            split_by_newline_positions(content) do |text|
              next if text.nil?
              next unless text.match?(/\w/)

              items << {
                content: text,
                metadata: metadata,
                url: url
              }
            end
            items
          end

          def parse_content_and_metadata(content, md5sum, source_name, source_type)
            match = content.match(metadata_regex)
            metadata = if match
                         metadata = YAML.safe_load(content.match(metadata_regex)[:metadata])
                         content = match.post_match.strip
                         metadata
                       else
                         {}
                       end

            metadata['title'] = title(content)
            metadata['md5sum'] = md5sum
            metadata['source'] = source_name
            metadata['source_type'] = source_type
            url = url(source_name, source_type)

            [content, metadata, url]
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

          def url(source_name, source_type)
            return unless source_name
            return unless source_type == 'doc'

            page = source_name.gsub('/doc/', '').gsub('.md', '')
            help_page_url(page)
          end

          def title(content)
            return unless content

            match = content.match(/#+\s+(?<title>.+)\n/)

            return unless match && match[:title]

            match[:title].gsub(/\*\*\(.+\)\*\*$/, '').strip
          end

          private

          def metadata_regex
            /\A---(?<metadata>.*?)---/m
          end

          def help_page_url(page)
            "#{root_url}/#{page}"
          end
        end
      end
    end
  end
end
