import astroid
import pylint.testutils

from lints import unsafe_dependency_resolution


class TestUnsafeDependencyResolution(pylint.testutils.CheckerTestCase):
    CHECKER_CLASS = unsafe_dependency_resolution.UnsafeDependencyResolution

    def test_finds_unsafe_dependency_resolution(self):
        node = astroid.extract_node(
            """
        Factory[SomeModel] = Depends(Provide[Container.some_factory.provider])
        """
        )

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="unsafe-dependency-resolution", node=node.value
            ),
            ignore_position=True,
        ):
            self.checker.visit_call(node.value)

    def test_ignores_depends_async_def_method(self):
        node = astroid.extract_node(
            """
        Factory[SomeModel] = Depends(async_def_method)
        """
        )

        with self.assertNoMessages():
            self.checker.visit_call(node.value)

    def test_ignores_provide_without_depends(self):
        node = astroid.extract_node(
            """
        Factory[SomeModel] = Provide[Container.some_factory.provider]
        """
        )

        with self.assertNoMessages():
            self.checker.visit_call(node.value)
