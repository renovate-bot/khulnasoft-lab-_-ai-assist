__all__ = ["PostProcessor"]


class PostProcessor:
    def process(self, text: str) -> str:
        """Execute a number of post-processing actions on a text.

        Args:
           completion: A text to process

        Returns:
           A resulted output
        """
        return self._strip_whitespaces(text)

    def _strip_whitespaces(self, text: str) -> str:
        return "" if text.isspace() else text
