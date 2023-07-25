import pytest

from codesuggestions.prompts.code_parser import CodeParser
from codesuggestions.suggestions.processing.ops import LanguageId


PYTHON_SOURCE_SAMPLE = """
import os
import time

# more code
# more code
# more code
# more code
# more code
# more code

import random
import pandas as pd

def sum(a, b):
    import numpy as np
    return a + b

def subtract(a, b):
    return a - b

class Calculator:
    def __init__(self):
        self.result = 0

    def calculateSum(self, a, b):
        self.result = sum(a, b)        
"""

C_SOURCE_SAMPLE = """
#include <stdio.h>
#include <stdlib.h>

// Define a struct called 'Person'
struct Person {
    char name[50];
    int age;
};

// Function to initialize a Person struct
void initializePerson(struct Person *person, const char *name, int age) {
    strcpy(person->name, name);
    person->age = age;
}

// Function to print the details of a Person
void printPersonDetails(const struct Person *person) {
    printf("Name: %s\n", person->name);
    printf("Age: %d\n", person->age);
}

int main() {
    struct Person p;
    initializePerson(&p, "John Doe", 25);
    printPersonDetails(&p);
    return 0;
}
"""

JAVASCRIPT_SOURCE_SAMPLE = """
import React, { useState } from "react";
import dateFns from "date-fns";
import { sum } from "mathjs";

const App = () => {
  const [date, setDate] = useState(new Date());
  const [number, setNumber] = useState(0);

  const addNumber = () => {
    setNumber(sum(number, 1));
  };

  const getDateString = () => {
    return dateFns.format(date, "YYYY-MM-DD");
  };

  return (
    <div>
      <h1>Date: {getDateString()}</h1>
      <h1>Number: {number}</h1>
      <button onClick={addNumber}>Add 1</button>
    </div>
  );
};

export default App;
"""

JAVASCRIPT_SAMPLE_SOURCE_2 = """
// Function Declaration 1
function add(a, b) {
  return a + b;
}

// Function Declaration 2
function multiply(a, b) {
  return a * b;
}

// Class Declaration
class Calculator {
  constructor() {
    this.result = 0;
  }

  // Method 1
  calculateSum(a, b) {
    this.result = add(a, b);
  }

  // Method 2
  calculateProduct(a, b) {
    this.result = multiply(a, b);
  }

  // Method 3
  getResult() {
    return this.result;
  }
}

// Generator function 1
function* countNumbers() {
  let i = 1;
  while (true) {
    yield i;
    i++;
  }
}

// Usage
const calculator = new Calculator();

calculator.calculateSum(5, 3);
console.log("Sum:", calculator.getResult()); // an inline comment

calculator.calculateProduct(5, 3);
console.log("Product:", calculator.getResult()); /* and a block comment */
"""

TS_SAMPLE_SOURCE = """
// Importing required modules
import { Calculator } from './calculator';

// Function 1: Add two numbers
function addNumbers(a: number, b: number): number {
  return a + b;
}

// Function 2: Multiply two numbers
function multiplyNumbers(a: number, b: number): number {
  return a * b;
}

// Main class
class MyApp {
  // Function to perform some calculations
  performCalculations(): void {
    const num1 = 5;
    const num2 = 3;

    const sum = addNumbers(num1, num2);
    console.log('Sum:', sum);

    const product = multiplyNumbers(num1, num2);
    console.log('Product:', product);
  }
}

// Instantiating the class and running the app
const myApp = new MyApp();
myApp.performCalculations();

"""

CPP_SAMPLE_SOURCE = """
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// comment 1
int main() {
  // Create a vector of strings.
  vector<string> strings = {"Hello", "World"};

  // Print the vector of strings.
  for (string string : strings) {
    cout << string << endl;
  }

  return 0;
}
"""

CSHARP_SAMPLE_SOURCE = """
using System.Console;

// comment 1
public class Program
{
  public static void Main()
  {
      Console.WriteLine("Hello, world!");
  }
}
"""

GO_SAMPLE_SOURCE = """
package main

// comment 1
import (
    "fmt"
    "log"
    "net/http"
    "os"
)

// comment 2
func main() {
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Hello, world!")
    })

    log.Fatal(http.ListenAndServe(":8080", nil))
}
"""

GO_SAMPLE_SOURCE_2 = """
package main

import "fmt"
import "log"
"""

JAVA_SAMPLE_SOURCE = """
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

// comment 1
@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}

/* block comment 1 */
"""

RUST_SAMPLE_SOURCE = """
use std::io;
use actix_web::{web, App, HttpServer};

// comment 1
fn main() {
    let server = HttpServer::new(|| {
        App::new()
            .service(web::get("/hello").to(hello))
    });

    server.bind("127.0.0.1:8080").unwrap().run().unwrap();
}

/* block comment 1 */
fn hello() -> impl Responder {
    "Hello, world!"
}
"""

SCALA_SAMPLE_SOURCE = """
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class Person

def greet(name: String): Unit = {
  println(s"Hello, $name!")
}

// comment 1
object Main extends App {

  val numbers = ArrayBuffer[Int]()
  for (i <- 1 to 10) {
    numbers += Random.nextInt(100)
  }

  println(numbers.sorted)
}
"""

PHP_SAMPLE_SOURCE = """
<?php

use SomeOtherNamespace\coolFunction;

// Import 1
require_once 'calculator.php';

// Function 1: Add two numbers
function addNumbers($a, $b) {
    return $a + $b;
}

// Function 2: Multiply two numbers
function multiplyNumbers($a, $b) {
    return $a * $b;
}

// Main class
class MyApp {
    // Function to perform some calculations
    public function performCalculations() {
        $num1 = 5;
        $num2 = 3;

        $sum = addNumbers($num1, $num2);
        echo 'Sum: ' . $sum . PHP_EOL;

        $product = multiplyNumbers($num1, $num2);
        echo 'Product: ' . $product . PHP_EOL;
    }
}

// Instantiating the class and running the app
$myApp = new MyApp();
$myApp->performCalculations();
?>
"""


@pytest.mark.parametrize(
    ("lang_id", "source_code", "target_symbols_counts"),
    [
        (LanguageId.PYTHON, PYTHON_SOURCE_SAMPLE, {"imports": 4, "functions": 2, "comments": 6, "classes": 1}),
        (LanguageId.C, C_SOURCE_SAMPLE, {"imports": 2, "functions": 3, "comments": 3}),
        (LanguageId.JS, JAVASCRIPT_SOURCE_SAMPLE, {"imports": 3}),
        (LanguageId.TS, TS_SAMPLE_SOURCE, {"imports": 1, "functions": 2, "comments": 5, "classes": 1}),
        (LanguageId.CPP, CPP_SAMPLE_SOURCE, {"imports": 3, "comments": 1, "functions": 1}),
        (LanguageId.CSHARP, CSHARP_SAMPLE_SOURCE, {"imports": 1, "comments": 1, "classes": 1}),
        (LanguageId.GO, GO_SAMPLE_SOURCE, {"imports": 1, "functions": 1, "comments": 2}),
        (LanguageId.GO, GO_SAMPLE_SOURCE_2, {"imports": 2}),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, {"imports": 2, "comments": 2, "classes": 1}),
        (LanguageId.RUST, RUST_SAMPLE_SOURCE, {"imports": 2, "comments": 2}),
        (LanguageId.SCALA, SCALA_SAMPLE_SOURCE, {"imports": 2, "comments": 1, "classes": 1, "functions": 1}),
        (LanguageId.JS, JAVASCRIPT_SAMPLE_SOURCE_2, {"classes": 1, "functions": 3, "comments": 7}),
        (LanguageId.PHP, PHP_SAMPLE_SOURCE, {"imports": 1, "functions": 2, "comments": 5, "classes": 1}),
    ],
)
def test_symbol_counter(
    lang_id: LanguageId,
    source_code: str,
    target_symbols_counts: set[str],
):
    parser = CodeParser(lang_id)
    output = parser.count_symbols(source_code, target_symbols=target_symbols_counts.keys())

    assert len(output) == len(target_symbols_counts)
    for symbol, expected_count in target_symbols_counts.items():
        assert output[symbol] == expected_count

@pytest.mark.parametrize(
    ("not_supported_lang_id"),
    [
        LanguageId.RUBY,
        LanguageId.KOTLIN,
    ]
)
def test_lang_id_not_supported(not_supported_lang_id: LanguageId):
    with pytest.raises(ValueError):
        CodeParser(not_supported_lang_id)
