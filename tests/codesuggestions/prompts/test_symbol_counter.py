import pytest

from codesuggestions.prompts.code_parser import CodeParser
from codesuggestions.suggestions.processing.ops import LanguageId

PYTHON_SOURCE_SAMPLE = """# Python test
import os
"""

PYTHON_SOURCE_SAMPLE_SPLIT_IMPORTS = """
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
    // Declare a Person struct variable
    struct Person p;

    // Initialize the Person struct using the initializePerson function
    initializePerson(&p, "John Doe", 25);

    // Print the details of the Person struct using the printPersonDetails function
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

TS_SAMPLE_SOURCE = """
import React from "react";
import ReactDOM from "react-dom";
import App from "./App";

ReactDOM.render(<App />, document.getElementById("root"));
"""

CPP_SAMPLE_SOURCE = """
#include <iostream>
#include <string>
#include <vector>

using namespace std;

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

public static void Main()
{
    Console.WriteLine("Hello, world!");
}
"""

GO_SAMPLE_SOURCE = """
package main

import (
    "fmt"
    "log"
    "net/http"
    "os"
)

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

@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
"""

RUST_SAMPLE_SOURCE = """
use std::io;
use actix_web::{web, App, HttpServer};

fn main() {
    let server = HttpServer::new(|| {
        App::new()
            .service(web::get("/hello").to(hello))
    });

    server.bind("127.0.0.1:8080").unwrap().run().unwrap();
}

fn hello() -> impl Responder {
    "Hello, world!"
}
"""

SCALA_SAMPLE_SOURCE = """
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object Main extends App {

  val numbers = ArrayBuffer[Int]()
  for (i <- 1 to 10) {
    numbers += Random.nextInt(100)
  }

  println(numbers.sorted)
}
"""


@pytest.mark.parametrize(
    ("lang_id", "source_code", "target_symbols_counts"),
    [
        (LanguageId.PYTHON, PYTHON_SOURCE_SAMPLE, {"imports": 1}),
        (LanguageId.PYTHON, PYTHON_SOURCE_SAMPLE_SPLIT_IMPORTS, {"imports": 4}),
        (LanguageId.PYTHON, PYTHON_SOURCE_SAMPLE_SPLIT_IMPORTS, {"functions": 2}),
        (LanguageId.C, C_SOURCE_SAMPLE, {"imports": 2, "functions": 3}),
        (LanguageId.JS, JAVASCRIPT_SOURCE_SAMPLE, {"imports": 3}),
        (LanguageId.TS, TS_SAMPLE_SOURCE, {"imports": 3}),
        (LanguageId.CPP, CPP_SAMPLE_SOURCE, {"imports": 3}),
        (LanguageId.CSHARP, CSHARP_SAMPLE_SOURCE, {"imports": 1}),
        (LanguageId.GO, GO_SAMPLE_SOURCE, {"imports": 1}),
        (LanguageId.GO, GO_SAMPLE_SOURCE_2, {"imports": 2}),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, {"imports": 2}),
        (LanguageId.RUST, RUST_SAMPLE_SOURCE, {"imports": 2}),
        (LanguageId.SCALA, SCALA_SAMPLE_SOURCE, {"imports": 2}),
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
