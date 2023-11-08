import pytest

from ai_gateway.code_suggestions.processing.base import LanguageId
from ai_gateway.prompts.parsers import CodeParser
from ai_gateway.prompts.parsers.comments import BaseCommentVisitor

EMPTY_SOURCE_FILE = ""

C_SOURCE_SAMPLE_COMMENTS = """// foo
/* bar */
"""
C_SOURCE_SAMPLE_MIXED = """
#include<stdio.h>

// foo
int main()
{
    /*
      bar
    */
	return printf("\nHello World!");
}
"""

CPP_SOURCE_SAMPLE_COMMENTS = """
// foo
/* bar */
"""
CPP_SOURCE_SAMPLE_MIXED = """
#include <iostream>

int main() {
  std::cout << "Hello world!\n";
}
"""

CSHARP_SOURCE_SAMPLE_COMMENTS = """
// foo
/* bar */
/// <summary>
///  C# also has XML comments
/// </summary>
"""
CSHARP_SOURCE_SAMPLE_MIXED = """
/// <summary>
///  This is a hello world program.
/// </summary>
namespace HelloWorld
{
    class Program
    {
        static void Main(string[] args)
        {
            System.Console.WriteLine("Hello world!");
        }
    }
}
"""

GO_SOURCE_SAMPLE_COMMENTS = """// foo
// bar
/*
func main(){

}
*/
"""

GO_SOURCE_SAMPLE_MIXED = """// the main package
package main

// The main function
func main() {
    
}
"""

JAVA_SOURCE_SAMPLE_COMMENTS = """
// foo
/* bar */
/**
 *
 * foobar
 *
 */
"""
JAVA_SOURCE_SAMPLE_MIXED = """
public class HelloWorld
{
 // foo
 public static void main(String[] args)
 {
  /*
   bar
  */
  System.out.println("Hello world!");
 }
}
"""

JS_SOURCE_SAMPLE_COMMENTS = """
// foo
/* bar */
"""
JS_SOURCE_SAMPLE_MIXED = """
// writes hello world
document.write("Hello world!");
"""


PYTHON_SOURCE_SAMPLE_COMMENTS = """
# foo
"""
PYTHON_SOURCE_SAMPLE_MIXED = """
# this prints hello world
if __name__=="__main__":
    print(f"Hello world!")
"""

RUBY_SOURCE_SAMPLE_COMMENTS = """
# foo
=begin
multiline comment
=end
"""
RUBY_SOURCE_SAMPLE_MIXED = """
# this says hello world
puts "Hello world!"
"""

RUST_SOURCE_SAMPLE_COMMENTS = """
// foo
// bar
"""
RUST_SOURCE_SAMPLE_MIXED = """
# this says hello world
fn main() {
   println!("Hello world!");
}
"""

SCALA_SOURCE_SAMPLE_COMMENTS = """
// foo
/* bar */
/**Comment start
*
*foobar
*
*comment ends*/
"""
SCALA_SOURCE_SAMPLE_MIXED = """
// foo
println("Hello world!")
"""

TS_SOURCE_SAMPLE_COMMENTS = """
// foo
/* bar */
/**
foobar
*/
"""
TS_SOURCE_SAMPLE_MIXED = """
// foo
let message: string = 'Hello, World!';
console.log(message);
"""

KOTLIN_SOURCE_SAMPLE_COMMENTS = """
// foo
/* bar */
/**
 *
 * foobar
 *
 */
"""
KOTLIN_SOURCE_SAMPLE_MIXED = """
// foo
fun main() {
    println("Hello world!")
}
"""


@pytest.mark.parametrize(
    ("lang_id", "source_code", "expected"),
    [
        (LanguageId.C, C_SOURCE_SAMPLE_COMMENTS, True),
        (LanguageId.C, C_SOURCE_SAMPLE_MIXED, False),
        (LanguageId.CPP, CPP_SOURCE_SAMPLE_COMMENTS, True),
        (LanguageId.CPP, CPP_SOURCE_SAMPLE_MIXED, False),
        (LanguageId.CSHARP, CSHARP_SOURCE_SAMPLE_COMMENTS, True),
        (LanguageId.CSHARP, CSHARP_SOURCE_SAMPLE_MIXED, False),
        (LanguageId.GO, GO_SOURCE_SAMPLE_COMMENTS, True),
        (LanguageId.GO, GO_SOURCE_SAMPLE_MIXED, False),
        (LanguageId.JAVA, JAVA_SOURCE_SAMPLE_COMMENTS, True),
        (LanguageId.JAVA, JAVA_SOURCE_SAMPLE_MIXED, False),
        (LanguageId.JS, JS_SOURCE_SAMPLE_COMMENTS, True),
        (LanguageId.JS, JS_SOURCE_SAMPLE_MIXED, False),
        (LanguageId.PYTHON, PYTHON_SOURCE_SAMPLE_COMMENTS, True),
        (LanguageId.PYTHON, PYTHON_SOURCE_SAMPLE_MIXED, False),
        (LanguageId.RUBY, RUBY_SOURCE_SAMPLE_COMMENTS, True),
        (LanguageId.RUBY, RUBY_SOURCE_SAMPLE_MIXED, False),
        (LanguageId.RUST, RUST_SOURCE_SAMPLE_COMMENTS, True),
        (LanguageId.RUST, RUST_SOURCE_SAMPLE_MIXED, False),
        (LanguageId.SCALA, SCALA_SOURCE_SAMPLE_COMMENTS, True),
        (LanguageId.SCALA, SCALA_SOURCE_SAMPLE_MIXED, False),
        (LanguageId.TS, TS_SOURCE_SAMPLE_COMMENTS, True),
        (LanguageId.TS, TS_SOURCE_SAMPLE_MIXED, False),
        (LanguageId.KOTLIN, KOTLIN_SOURCE_SAMPLE_COMMENTS, True),
        (LanguageId.KOTLIN, KOTLIN_SOURCE_SAMPLE_MIXED, False),
    ],
)
def test_comments_only(lang_id: LanguageId, source_code: str, expected: bool):
    parser = CodeParser.from_language_id(source_code, lang_id)
    output = parser.comments_only()

    assert output == expected
