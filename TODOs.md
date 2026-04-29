**In no particular order:**

* Code quality/Some tools to introduce:
   * tests and formating standards - "pytest", "black", "flake8", "isort", "mypy",
   * consistency with amount of comments, docstrings etc

| Capability         | Fallow | Python ecosystem           |
| ------------------ | ------ | -------------------------- |
| Dead code          | ✅      | Vulture                    |
| Duplication        | ✅      | jscpd / Clone Digger       |
| Complexity         | ✅      | Radon                      |
| Architecture rules | ✅      | (rare / manual)            |
| One unified CLI    | ✅      | ❌ (usually multiple tools) |

   * UV for package management - https://robert-mcdermott.medium.com/saying-goodbye-to-anaconda-91c18ddf89bb
   * https://pydantic.dev/docs/validation/latest/get-started/ for type hinting instead of mypy?
* Bootstrap quantiles
* way more code abstraction:
    * a lot of boilerplate code that makes the code not that readable
    * more early exits rather then if statements
    * reduce deep loops etc, never nesting
* more debug logging
* look for duplicate code and pull out to utils
* fix GSL
* implement more indices
    * sector keywords
    * https://www.climdex.org/learn/indices/    
    * https://climate.copernicus.eu/sectoral-specific-challenges
* document
    * document for developers
    * document usage
    * document algs
* Better metadata handling for netcdf writing
* regression tests
* Test another backend: fortran, c++
* faster io
* logo

