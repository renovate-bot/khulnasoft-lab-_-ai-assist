# The original string containing escaped characters for newlines and tabs
code_string = r"\n    async def _handle_stream(\n        self, response: AsyncIterator[TextGenModelChunk]\n    ) -> AsyncIterator[CodeSuggestionsChunk]:\n        chunks = []\n        total_tokens = 0\n        try:\n            async for chunk in response:\n                chunk_content = CodeSuggestionsChunk(text=chunk.text)\n                chunks.append(chunk.text)\n                total_tokens += self.tokenization_strategy.estimate_length([chunk.text])\n                yield chunk_content\n        finally:\n            self.snowplow_instrumentator.watch(\n                SnowplowEvent(\n                    context=None,\n                    action=\"tokens_per_user_request_response\",\n                    label=\"code_generation\",\n                    value=total_tokens,\n                )\n            )\n"

# Replace escaped newline and tab characters with actual newlines and tabs
pretty_code = code_string.replace("\\n", "\n").replace("\\t", "\t")

# Print the formatted code
print(pretty_code)
