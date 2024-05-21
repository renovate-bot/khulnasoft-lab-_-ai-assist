import astroid
import pylint.testutils

from lints import direct_environment_variable_reference


class TestDirectEnvironmentVariableReference(pylint.testutils.CheckerTestCase):
    CHECKER_CLASS = (
        direct_environment_variable_reference.DirectEnvironmentVariableReference
    )

    def test_finds_os_environ(self):
        node = astroid.extract_node("""os.environ['KEY']""")

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="direct-environment-variable-reference", node=node.value
            ),
            ignore_position=True,
        ):
            self.checker.visit_attribute(node.value)

    def test_finds_os_getenv(self):
        node = astroid.extract_node("""os.getenv('KEY')""")

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="direct-environment-variable-reference", node=node.func
            ),
            ignore_position=True,
        ):
            self.checker.visit_attribute(node.func)

    def test_ignores_os_getpid(self):
        node = astroid.extract_node("""os.getppid()""")

        with self.assertNoMessages():
            self.checker.visit_attribute(node)
