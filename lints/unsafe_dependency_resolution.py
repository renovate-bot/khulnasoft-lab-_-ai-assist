from astroid import nodes
from pylint.checkers import BaseChecker
from pylint.lint import PyLinter


class UnsafeDependencyResolution(BaseChecker):
    name = "unsafe-dependency-resolution"
    msgs = {
        "W5001": (
            "Unsafe dependency resolution detected.",
            "unsafe-dependency-resolution",
            "See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/606 for more information.",
        )
    }

    def visit_call(self, node: nodes.Call) -> None:
        if (
            hasattr(node, "func")
            and isinstance(node.func, nodes.Name)
            and node.func.name == "Depends"
        ):
            subscript = node.args[0]
            if (
                hasattr(subscript, "value")
                and isinstance(subscript.value, nodes.Name)
                and subscript.value.name == "Provide"
            ):
                self.add_message("unsafe-dependency-resolution", node=node)


def register(linter: "PyLinter") -> None:
    linter.register_checker(UnsafeDependencyResolution(linter))
