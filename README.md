![](./assets/logo.svg)

[![Unit Testing](https://github.com/rentruewang/inversql/actions/workflows/unittest.yaml/badge.svg)](https://github.com/rentruewang/inversql/actions/workflows/unittest.yaml)
[![Pre Commit Checks](https://github.com/rentruewang/inversql/actions/workflows/precommit.yaml/badge.svg)](https://github.com/rentruewang/inversql/actions/workflows/precommit.yaml)
[![Publish](https://github.com/rentruewang/inversql/actions/workflows/release.yaml/badge.svg)](https://github.com/rentruewang/inversql/actions/workflows/release.yaml)

![PyPI](https://img.shields.io/pypi/v/inversql)
![MIT](https://img.shields.io/badge/license-MIT-blue)
[![](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)][demo]

# <a id="inversql"></a> 🔄 InverSQL

## Generate SQL that match a set of records by decomposing decision trees

### <a id="demo-gif"></a> 🎬 Demo in a GIF

![](./assets/quick-demo.gif)

### Architecture

```

    +--------------+                    +-----------------+
    |  Controller  | <----------------- | Display to User |
    +------+-------+                    +--------+--------+
           |                                     |
    Process Input (CSV)           Returns Processed Data/SQL
           |                                     |
           V                                     |
    +--------------------------------------------+---------+
    |                        MODEL                         |
    |                                                      |
    |               [ Feature Extraction ]                 |
    |               (Joins based on stats)                 |
    |                         |                            |
    |                         |                            |
    |                         V                            |
    |               [ Run Decision Tree ]                  |
    |                         |                            |
    |                         V                            |
    |             [ Decompose Decision Tree ]              |
    |                         |                            |
    |                         V                            |
    |             [ Simplify / Optimize SQL ]              |
    |                                                      |
    +------------------------------------------------------+
```

### 🌟 Like and subscribe!

That's pretty much it! If you have read this far, please consider giving me a star (⭐) or a fork (🍴). This will keep my motivation going!

Or if you have too much cash at hand:

[![BuyMeACoffee](https://raw.githubusercontent.com/pachadotdev/buymeacoffee-badges/main/bmc-yellow.svg)](https://www.buymeacoffee.com/rentruewang)

[demo]: https://inversql.streamlit.app
