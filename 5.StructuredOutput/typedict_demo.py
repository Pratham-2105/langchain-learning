from typing import TypedDict


class Person(TypedDict):

    name: str
    age: int

new_person: Person = {"name":"Pratham", "age":"21"}

print(new_person)